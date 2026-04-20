# ============================================================
# Qwen3.5 Gateway — OpenRouter-style API Server for Google Colab
# Uses the correct AutoModelForImageTextToText API
# ============================================================

COLAB_CONFIG: dict = {
    "model":                  "4b",       # "2b" or "4b"
    "max_tokens":             8192,
    "max_model_len":          8192,
    "api_key":                "",         # leave empty = no auth
    "rate_limit_rpm":         60,
    "drive_cache_dir":        "/content/drive/MyDrive/qwen35_cache",
    "gpu_memory_utilization": 0.75,       # lowered — 4B needs headroom on T4
    "clear_cache":            False,      # set True only to force a fresh re-download
    "offload_folder":         "/tmp/qwen_offload",  # disk spill if VRAM full
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

    # Clear old weights if requested (e.g. switching model size)
    if COLAB_CONFIG.get("clear_cache") and drive_cache.exists():
        import shutil
        log.info(f"Clearing old cache at {drive_cache} ...")
        shutil.rmtree(str(drive_cache), ignore_errors=True)
        print(f"[Cache] Deleted {drive_cache} — will re-download fresh.")

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

# GPU check — fail fast with a clear message
HAS_GPU = torch.cuda.is_available()
if not HAS_GPU:
    print("\n" + "!" * 60)
    print("  WARNING: No GPU detected — running on CPU.")
    print("  Inference will be very slow (minutes per response).")
    print("  Fix: Runtime -> Change runtime type -> T4 GPU -> Save")
    print("!" * 60 + "\n")
else:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_gb   = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU: {gpu_name}  ({gpu_gb:.1f} GB VRAM)")

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
tokenizer = processor.tokenizer  # for TextIteratorStreamer

if HAS_GPU:
    real_gpu_gb = int(torch.cuda.get_device_properties(0).total_memory / 1e9)
    load_kwargs = dict(
        device_map     = "auto",
        dtype          = torch.float16,
        offload_folder = COLAB_CONFIG["offload_folder"],
        max_memory     = {
            0:     f"{int(COLAB_CONFIG['gpu_memory_utilization'] * real_gpu_gb)}GiB",
            "cpu": "12GiB",
        },
    )
else:
    # CPU fallback — float32 only (float16 unsupported on CPU)
    load_kwargs = dict(device_map="cpu", dtype=torch.float32)

model = AutoModelForImageTextToText.from_pretrained(
    model_path, trust_remote_code=True, **load_kwargs
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
def normalise_messages(messages: list) -> list:
    """
    Ensure every message's content is a list of typed blocks.
    apply_chat_template requires list[dict], not a plain string.
    Also upgrades legacy {"image":"..."} → {"type":"image","url":"..."}.
    """
    out = []
    for msg in messages:
        raw = msg.get("content", "")
        if isinstance(raw, str):
            # Plain string → single text block
            raw = [{"type": "text", "text": raw}]
        elif isinstance(raw, list):
            fixed = []
            for block in raw:
                if isinstance(block, str):
                    block = {"type": "text", "text": block}
                elif isinstance(block, dict) and "type" not in block:
                    if "image" in block:
                        block = {"type": "image", "url": block["image"]}
                    elif "video" in block:
                        block = {"type": "video", "url": block["video"]}
                fixed.append(block)
            raw = fixed
        out.append({**msg, "content": raw})
    return out

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def build_inputs(messages: list) -> dict:
    """
    Normalise then run processor.apply_chat_template.
    image_grid_thw is kept on CPU to avoid an nvrtc JIT bug on Colab T4
    (libnvrtc-builtins missing triggers a crash in prod() kernel compilation).
    All other tensors go to GPU as normal.
    """
    inputs = processor.apply_chat_template(
        normalise_messages(messages),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    # Move everything to GPU except image_grid_thw — that tensor only drives
    # shape arithmetic inside the vision encoder and must stay on CPU to avoid
    # the nvrtc reduction-kernel JIT crash on Colab T4.
    result = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.cpu() if k == "image_grid_thw" else v.to(model.device)
        else:
            result[k] = v
    return result

def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return THINK_RE.sub("", text).strip()

def _gen_kwargs(inputs: dict, max_new_tokens: int, temperature: float) -> dict:
    return dict(
        **inputs,
        max_new_tokens  = max_new_tokens,
        temperature     = max(temperature, 0.01),
        do_sample       = temperature > 0.05,
        pad_token_id    = tokenizer.eos_token_id,
    )

def run_generation_sync(inputs: dict, max_new_tokens: int, temperature: float) -> str:
    """Blocking generation. Returns decoded text with <think> blocks stripped."""
    with torch.inference_mode():
        out = model.generate(**_gen_kwargs(inputs, max_new_tokens, temperature))
    prompt_len = inputs["input_ids"].shape[-1]
    raw = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    return _strip_think(raw)

def run_generation_stream(inputs: dict, max_new_tokens: int, temperature: float) -> TextIteratorStreamer:
    """Non-blocking generation. Returns a streamer to iterate over."""
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    threading.Thread(
        target=model.generate,
        kwargs=dict(**_gen_kwargs(inputs, max_new_tokens, temperature), streamer=streamer),
        daemon=True,
    ).start()
    return streamer

def stream_filtered(streamer):
    """
    Yield chunks with <think>...</think> blocks suppressed.
    Works across chunk boundaries by buffering while inside a think block.
    """
    buf = ""
    in_think = False
    for chunk in streamer:
        buf += chunk
        while True:
            if in_think:
                end = buf.find("</think>")
                if end == -1:
                    buf = ""
                    break
                buf = buf[end + len("</think>"):]
                in_think = False
            else:
                start = buf.find("<think>")
                if start == -1:
                    yield buf
                    buf = ""
                    break
                if start > 0:
                    yield buf[:start]
                buf = buf[start + len("<think>"):]
                in_think = True
    leftover = buf.strip()
    if leftover and not in_think:
        yield leftover

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
            for chunk in stream_filtered(streamer):
                if not chunk: continue
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

    inputs = build_inputs(msgs)  # normalise_messages() handles all content formats

    if stream:
        streamer = run_generation_stream(inputs, max_tok, temp)

        async def event_gen():
            for chunk in stream_filtered(streamer):
                if not chunk: continue
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

# ── Install cloudflared if missing ────────────────────────────
def ensure_cloudflared():
    import shutil, stat
    if shutil.which("cloudflared"):
        return
    log.info("cloudflared not found — downloading...")
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    dest = Path("/usr/local/bin/cloudflared")
    subprocess.check_call(["wget", "-q", "-O", str(dest), url])
    dest.chmod(dest.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    log.info("cloudflared installed.")

ensure_cloudflared()

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
