# ============================================================
# Qwen3.5 Gateway — OpenRouter-style API Server for Google Colab
# Uses the correct AutoModelForImageTextToText API
# ============================================================

COLAB_CONFIG: dict = {
    "model":                  "4b",   # "2b" or "4b"
    "max_tokens":             8192,
    "max_model_len":          8192,
    "api_key":                "",     # leave empty = no auth
    "rate_limit_rpm":         60,
    "drive_cache_dir":        "/content/drive/MyDrive/qwen35_cache",
    "gpu_memory_utilization": 0.85,
}

import base64, gc, logging, os, re, sys, time, json
import threading, subprocess
from collections import defaultdict, deque
from pathlib import Path

import httpx

PROXY_PORT = 8090
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("qwen-gateway")

print("=" * 60)
print("  Qwen3.5 Gateway  —  Vision + Text  —  OpenRouter-style")
print("=" * 60 + "\n")

# ── 0. Cleanup ────────────────────────────────────────────────
subprocess.run("pkill -9 -f 'cloudflared'", shell=True, stderr=subprocess.DEVNULL)
subprocess.run(f"fuser -k -9 {PROXY_PORT}/tcp", shell=True, stderr=subprocess.DEVNULL)
time.sleep(2)
gc.collect()

# ── 1. Cache ──────────────────────────────────────────────────
def setup_cache() -> Path:
    drive_cache = Path(COLAB_CONFIG["drive_cache_dir"])
    if Path("/content/drive/MyDrive").exists():
        drive_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(drive_cache)
        log.info(f"Drive cache: {drive_cache}")
        return drive_cache
    fallback = Path("/tmp/qwen35_cache")
    fallback.mkdir(exist_ok=True)
    return fallback

cache_root = setup_cache()

MODEL_MAP = {
    "2b": "Qwen/Qwen3.5-2B",
    "4b": "Qwen/Qwen3.5-4B",
}
model_id = MODEL_MAP[COLAB_CONFIG["model"]]

def resolve_model(mid: str):
    slug = mid.replace("/", "--")
    snap_root = cache_root / "hub" / f"models--{slug}" / "snapshots"
    if snap_root.exists():
        snaps = sorted(
            [p for p in snap_root.glob("*/") if p.is_dir()],
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if snaps:
            log.info(f"Using cached model: {snaps[0]}")
            return str(snaps[0]), True
    return mid, False

model_path, from_cache = resolve_model(model_id)

# ── 2. Dependencies ───────────────────────────────────────────
print("[2/4] Installing / upgrading dependencies...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "--upgrade",
    "transformers>=4.52.0",  # Qwen3.5 multimodal needs recent transformers
    "torchvision",
    "accelerate",
    "fastapi",
    "uvicorn[standard]",
    "httpx",
    "PyMuPDF",
])

# ── 3. Load model ─────────────────────────────────────────────
print(f"[3/4] Loading {model_id} ...")
import torch
import fitz  # PyMuPDF
from transformers import AutoProcessor, AutoModelForImageTextToText, TextIteratorStreamer

# Correct class for Qwen3.5 (replaces AutoModelForCausalLM)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
tokenizer = processor.tokenizer  # for TextIteratorStreamer

model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
).eval()

log.info(f"Model device: {next(model.parameters()).device}")
print(f"[3/4] Done.\n")

# ── 4. Rate limiter ───────────────────────────────────────────
_rate_windows: dict = defaultdict(deque)

def is_rate_limited(client_ip: str) -> bool:
    rpm = COLAB_CONFIG["rate_limit_rpm"]
    if rpm <= 0:
        return False
    now = time.time()
    window = _rate_windows[client_ip]
    while window and window[0] < now - 60:
        window.popleft()
    if len(window) >= rpm:
        return True
    window.append(now)
    return False

# ── 5. Media helpers ──────────────────────────────────────────
async def pdf_bytes_to_image_blocks(pdf_bytes: bytes) -> list:
    """Rasterise every PDF page at 150 DPI into image content blocks."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    blocks = []
    for page in doc:
        pix = page.get_pixmap(dpi=150)
        b64 = base64.b64encode(pix.tobytes("png")).decode()
        blocks.append({"type": "image", "url": f"data:image/png;base64,{b64}"})
    doc.close()
    return blocks

async def source_to_content_blocks(source: str) -> list:
    """
    Convert any source string into Qwen3.5 content blocks.
    Qwen3.5 uses  {"type": "image", "url": "..."}  — the HF standard format.
    """
    if source.startswith("data:"):
        if "application/pdf" in source:
            _, encoded = source.split(",", 1)
            return await pdf_bytes_to_image_blocks(base64.b64decode(encoded))
        return [{"type": "image", "url": source}]

    if source.startswith(("http://", "https://")):
        clean = source.split("?")[0].lower()
        ext   = clean.rsplit(".", 1)[-1] if "." in clean else ""
        if ext == "pdf":
            async with httpx.AsyncClient(follow_redirects=True, timeout=60) as cli:
                r = await cli.get(source)
                r.raise_for_status()
            return await pdf_bytes_to_image_blocks(r.content)
        if ext in ("mp4", "mov", "avi", "webm"):
            return [{"type": "video", "url": source}]
        return [{"type": "image", "url": source}]

    return [{"type": "text", "text": source}]

# ── 6. Core inference ─────────────────────────────────────────
def build_inputs(messages: list) -> dict:
    """
    Use processor.apply_chat_template — the correct Qwen3.5 API.
    Handles vision tokens internally; no process_vision_info needed.
    No mm_token_type_ids issue with this path.
    """
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

def run_generation_sync(inputs: dict, max_new_tokens: int, temperature: float) -> str:
    """Blocking generation. Returns decoded text only (no prompt)."""
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0.05,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)

def run_generation_stream(inputs: dict, max_new_tokens: int, temperature: float) -> TextIteratorStreamer:
    """Non-blocking generation. Returns a streamer to iterate over."""
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    threading.Thread(
        target=model.generate,
        kwargs=dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0.05,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        ),
        daemon=True,
    ).start()
    return streamer

# ── 7. FastAPI ────────────────────────────────────────────────
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="Qwen3.5 Gateway", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok", "model": model_id}

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": model_id, "object": "model", "owned_by": "qwen"}],
    }

@app.post("/v1/agent")
async def agent_endpoint(request: Request):
    """Multimodal shortcut: accepts files[] + prompt directly."""
    if is_rate_limited(request.client.host):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    body = await request.json()

    content: list = []
    for s in body.get("files", []):
        content.extend(await source_to_content_blocks(str(s)))
    if body.get("prompt"):
        content.append({"type": "text", "text": body["prompt"]})

    messages = [{"role": "user", "content": content}]
    stream   = body.get("stream", False)
    max_tok  = int(body.get("max_tokens",   COLAB_CONFIG["max_tokens"]))
    temp     = float(body.get("temperature", 0.7))

    inputs = build_inputs(messages)

    if stream:
        streamer = run_generation_stream(inputs, max_tok, temp)

        async def event_gen():
            for chunk in streamer:
                yield f"data: {json.dumps({'choices':[{'delta':{'content':chunk}}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    text = run_generation_sync(inputs, max_tok, temp)
    return JSONResponse({
        "model":   model_id,
        "choices": [{"message": {"role": "assistant", "content": text}}],
    })

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible endpoint. Accepts standard messages[] format."""
    if is_rate_limited(request.client.host):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    body    = await request.json()
    msgs    = body.get("messages", [])
    stream  = body.get("stream", False)
    max_tok = int(body.get("max_tokens",   COLAB_CONFIG["max_tokens"]))
    temp    = float(body.get("temperature", 0.7))

    # Normalise legacy {"image": "..."} blocks to HF standard {"type":"image","url":"..."}
    normalised = []
    for msg in msgs:
        raw = msg.get("content", "")
        if isinstance(raw, list):
            fixed = []
            for block in raw:
                if isinstance(block, dict) and "type" not in block:
                    if "image" in block:
                        block = {"type": "image", "url": block["image"]}
                    elif "video" in block:
                        block = {"type": "video", "url": block["video"]}
                fixed.append(block)
            raw = fixed
        normalised.append({**msg, "content": raw})

    inputs = build_inputs(normalised)

    if stream:
        streamer = run_generation_stream(inputs, max_tok, temp)

        async def event_gen():
            for chunk in streamer:
                yield f"data: {json.dumps({'choices':[{'delta':{'content':chunk}}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    text = run_generation_sync(inputs, max_tok, temp)
    return JSONResponse({
        "id":      f"chatcmpl-{int(time.time())}",
        "object":  "chat.completion",
        "model":   model_id,
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {},
    })

# ── 8. Start ──────────────────────────────────────────────────
print("[4/4] Starting server...")
import uvicorn
import urllib.request as _ur

threading.Thread(
    target=lambda: uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT, access_log=False),
    daemon=True,
).start()
time.sleep(4)

try:
    with _ur.urlopen(f"http://127.0.0.1:{PROXY_PORT}/health", timeout=5) as r:
        log.info(f"Server health: {r.read().decode()}")
except Exception as e:
    log.warning(f"Health check failed (server may still be starting): {e}")

print("\n[Tunnel] Starting cloudflared...")
proc = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", f"http://localhost:{PROXY_PORT}"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
for line in proc.stdout:
    m = re.search(r"https://[a-z0-9\-]+\.trycloudflare\.com", line)
    if m:
        url = m.group(0)
        print("\n" + "=" * 60)
        print(f"  PUBLIC URL : {url}")
        print(f"  HEALTH     : {url}/health")
        print(f"  MODELS     : {url}/v1/models")
        print("=" * 60 + "\n")
        break

try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    proc.terminate()
