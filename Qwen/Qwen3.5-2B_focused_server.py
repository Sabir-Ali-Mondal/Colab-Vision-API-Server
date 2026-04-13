import os
os.environ.setdefault("HF_HOME", "/content/drive/MyDrive/vllm_models")

import sys, subprocess, time, re, base64, threading, shutil, mimetypes, uuid
from pathlib import Path

LOGFILE    = Path("/tmp/server.log")
VLLM_LOG   = Path("/tmp/vllm.log")
UPLOAD_DIR = Path("/tmp/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOGFILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass

# ──────────────────────────────────────────────
# Google Drive mount
# ──────────────────────────────────────────────
def mount_drive():
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        p = Path("/content/drive/MyDrive/vllm_models")
        p.mkdir(parents=True, exist_ok=True)
        for key in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_HUB_CACHE"):
            os.environ[key] = str(p)
        log(f"Drive cache: {p}")
        return p
    except Exception:
        fallback = Path("/tmp/hf_cache")
        fallback.mkdir(parents=True, exist_ok=True)
        for key in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_HUB_CACHE"):
            os.environ[key] = str(fallback)
        log(f"No Drive — using {fallback}")
        return fallback

CACHE_DIR = mount_drive()

# ──────────────────────────────────────────────
# Hardware detection
# ──────────────────────────────────────────────
def detect_hw():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cc   = torch.cuda.get_device_capability(0)
            vram = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            log(f"GPU: {name}  cc={cc}  VRAM={vram}GB")
            return "gpu", name, cc, vram
    except Exception:
        pass
    log("No GPU — CPU mode")
    return "cpu", "", (0, 0), 0

HW, GPU_NAME, CC, VRAM_GB = detect_hw()

def hw_config():
    if HW == "cpu":
        return dict(
            dtype="float32",
            # Keep context small on CPU to avoid OOM
            max_len=4096,
            seqs=2,
            extra=["--device", "cpu", "--block-size", "16", "--swap-space", "4"],
        )
    dtype = "bfloat16" if CC >= (8, 0) else "float16"
    # Scale context window to available VRAM
    # T4=16GB → 32768, A100=40/80GB → 131072, anything smaller → 16384
    if VRAM_GB >= 40:
        max_len = 131072
    elif VRAM_GB >= 16:
        max_len = 32768
    else:
        max_len = 16384
    return dict(
        dtype=dtype,
        max_len=max_len,
        seqs=16,
        extra=["--gpu-memory-utilization", "0.90"],
    )

CFG = hw_config()

# ──────────────────────────────────────────────
# Model selection
# Qwen3.5-2B: natively multimodal (early-fusion
# image+text), 262K context, thinking mode.
# ──────────────────────────────────────────────
DEFAULT_MODEL   = "Qwen/Qwen3.5-2B"
FALLBACK_MODELS = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen2.5-VL-3B-Instruct"]
USER_MODEL      = os.environ.get("MODEL_ID", "").strip()

def pick_model():
    from huggingface_hub import HfApi
    api = HfApi()
    candidates = ([USER_MODEL] if USER_MODEL else []) + [DEFAULT_MODEL] + FALLBACK_MODELS
    for m in candidates:
        if not m:
            continue
        try:
            api.model_info(m, timeout=8)
            log(f"Model: {m}")
            return m
        except Exception as exc:
            log(f"  {m} unreachable: {exc}")
    raise RuntimeError("No model available — set MODEL_ID env var or check network")

MODEL = pick_model()

# ──────────────────────────────────────────────
# Install
# ──────────────────────────────────────────────
def install():
    pkgs = [
        "vllm>=0.9.0",          # Qwen3.5 needs recent vLLM
        "fastapi",
        "uvicorn[standard]",
        "httpx",
        "pillow>=10.4.0",
        "huggingface_hub",
        "python-multipart",     # FastAPI file uploads
        "aiofiles",
        "pdfplumber",           # PDF text extraction
    ]
    log("Installing packages …")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", *pkgs],
        check=True,
    )
    log("Packages ready.")

install()

import asyncio
import httpx
import uvicorn
import aiofiles
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

# ──────────────────────────────────────────────
# vLLM launch  — flags verified against:
#   https://huggingface.co/Qwen/Qwen3.5-2B
#   https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html
# ──────────────────────────────────────────────
def build_vllm_cmd():
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model",            MODEL,
        "--port",             "8001",
        "--host",             "0.0.0.0",
        "--dtype",            CFG["dtype"],
        "--max-model-len",    str(CFG["max_len"]),
        "--max-num-seqs",     str(CFG["seqs"]),
        "--download-dir",     str(CACHE_DIR),
        # Required: parses <think>...</think> tokens in Qwen3.5 responses
        "--reasoning-parser", "qwen3",
        # Enable thinking mode by default; callers can override per-request
        # via extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    ] + CFG["extra"]

    # NOTE: do NOT add --trust-remote-code or --generation-config vllm
    # for Qwen3.5-2B — they cause startup errors with this model family.
    # do NOT add --limit-mm-per-prompt — handled automatically by vLLM
    # for the Qwen3_5ForConditionalGeneration class.
    return cmd

def start_vllm():
    cmd = build_vllm_cmd()
    log("vLLM: " + " ".join(cmd))
    return subprocess.Popen(cmd, stdout=open(VLLM_LOG, "w"), stderr=subprocess.STDOUT)

PROC = start_vllm()

def wait_ready(retries=120, delay=5):
    for i in range(retries):
        try:
            if httpx.get("http://127.0.0.1:8001/health", timeout=3).status_code == 200:
                log("vLLM ready ✓")
                return True
        except Exception:
            pass
        if i % 6 == 0:
            log(f"Waiting for vLLM … ({i * delay}s)")
        time.sleep(delay)
    # Surface failure reason
    try:
        lines = Path(VLLM_LOG).read_text().splitlines()[-50:]
        log("=== vLLM last 50 lines ===")
        for l in lines:
            log("  " + l)
    except Exception:
        pass
    return False

if not wait_ready():
    raise RuntimeError("vLLM failed to start — see logs above.")

# ──────────────────────────────────────────────
# File helpers
# ──────────────────────────────────────────────
ALLOWED_IMAGE_MIME = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_FILE_BYTES     = 50 * 1024 * 1024   # 50 MB

def detect_mime(data: bytes, filename: str = "") -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":               return "image/png"
    if data[:3] == b"\xff\xd8\xff":                     return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":  return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):              return "image/gif"
    if data[:4] == b"%PDF":                             return "application/pdf"
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"

def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"

def read_text_file(path: Path, mime: str) -> str:
    """Extract plain text. Handles PDF via pdfplumber, else UTF-8."""
    if mime == "application/pdf":
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                return "\n".join(pg.extract_text() or "" for pg in pdf.pages)
        except Exception as e:
            log(f"pdfplumber failed: {e} — falling back to raw bytes")
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return path.read_bytes().decode("latin-1", errors="replace")

# ──────────────────────────────────────────────
# In-memory file registry  { uuid -> meta dict }
# ──────────────────────────────────────────────
file_store: dict[str, dict] = {}

# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────
app = FastAPI(title="Qwen3.5 Server", version="4.0")
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
REQ_COUNT = 0

# ── health / metrics ────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL, "hardware": HW,
            "gpu": GPU_NAME, "max_len": CFG["max_len"]}

@app.get("/metrics")
async def metrics():
    return {"requests": REQ_COUNT, "model": MODEL,
            "hardware": HW, "files_stored": len(file_store)}

# ────────────────────────────────────────────
# FILE UPLOAD   POST /upload
# ────────────────────────────────────────────
# Accepts any file. Images are stored and later
# sent as base64 image_url blocks to vLLM.
# Text/PDF/CSV/JSON files are read and injected
# as plain-text context into the prompt.
#
# Returns:
#   { file_id, filename, mime, kind, size, size_fmt }
# ────────────────────────────────────────────
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(413, f"Too large (max {fmt_size(MAX_FILE_BYTES)})")

    mime = detect_mime(raw, file.filename or "")
    kind = "image" if mime in ALLOWED_IMAGE_MIME else "text"
    fid  = str(uuid.uuid4())
    dest = UPLOAD_DIR / fid

    async with aiofiles.open(dest, "wb") as f:
        await f.write(raw)

    meta = {
        "file_id":  fid,
        "filename": file.filename or fid,
        "mime":     mime,
        "kind":     kind,
        "size":     len(raw),
        "size_fmt": fmt_size(len(raw)),
        "path":     str(dest),
    }
    file_store[fid] = meta
    log(f"Upload: {file.filename} [{mime}] {fmt_size(len(raw))} → {fid}")
    return meta

# ── LIST   GET /files ────────────────────────
@app.get("/files")
async def list_files():
    return {"files": list(file_store.values())}

# ── DOWNLOAD   GET /files/{id} ───────────────
@app.get("/files/{file_id}")
async def download_file(file_id: str):
    meta = file_store.get(file_id)
    if not meta:
        raise HTTPException(404, "File not found")
    return FileResponse(path=meta["path"],
                        media_type=meta["mime"],
                        filename=meta["filename"])

# ── DELETE   DELETE /files/{id} ──────────────
@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    meta = file_store.pop(file_id, None)
    if not meta:
        raise HTTPException(404, "File not found")
    Path(meta["path"]).unlink(missing_ok=True)
    log(f"Deleted: {file_id}")
    return {"deleted": file_id}

# ────────────────────────────────────────────
# CHAT   POST /chat
# ────────────────────────────────────────────
# JSON body:
# {
#   "prompt":          "What is in this image?",  ← required
#   "file_ids":        ["<uuid>", ...],            ← optional
#   "max_tokens":      512,                        ← optional
#   "temperature":     0.6,                        ← optional
#   "top_p":           0.9,                        ← optional
#   "stream":          false,                      ← optional
#   "enable_thinking": true                        ← optional (Qwen3.5 thinking mode)
# }
# ────────────────────────────────────────────
@app.post("/chat")
async def chat(req: Request):
    global REQ_COUNT
    REQ_COUNT += 1
    body = await req.json()

    prompt          = body.get("prompt", "").strip()
    file_ids        = body.get("file_ids", [])
    max_tokens      = int(body.get("max_tokens",      512))
    temperature     = float(body.get("temperature",   0.6))
    top_p           = float(body.get("top_p",         0.9))
    stream          = bool(body.get("stream",         False))
    enable_thinking = bool(body.get("enable_thinking", True))

    if not prompt:
        raise HTTPException(400, "prompt is required")

    content = await _build_content(prompt, file_ids)
    return await _call_vllm(content, max_tokens, temperature, top_p,
                             stream, enable_thinking)

# ────────────────────────────────────────────
# CHAT + UPLOAD in one shot   POST /chat/upload
# ────────────────────────────────────────────
# multipart/form-data:
#   prompt           (required)
#   file             (optional)
#   max_tokens       (optional, default 512)
#   temperature      (optional, default 0.6)
#   top_p            (optional, default 0.9)
#   stream           (optional, "true"/"false")
#   enable_thinking  (optional, "true"/"false")
# ────────────────────────────────────────────
@app.post("/chat/upload")
async def chat_with_upload(
    prompt:          str                  = Form(...),
    file:            Optional[UploadFile] = File(None),
    max_tokens:      int                  = Form(512),
    temperature:     float                = Form(0.6),
    top_p:           float                = Form(0.9),
    stream:          str                  = Form("false"),
    enable_thinking: str                  = Form("true"),
):
    global REQ_COUNT
    REQ_COUNT += 1
    file_ids = []
    if file and file.filename:
        meta = await upload_file(file)
        file_ids.append(meta["file_id"])

    content = await _build_content(prompt, file_ids)
    return await _call_vllm(content, max_tokens, temperature, top_p,
                             stream.lower() == "true",
                             enable_thinking.lower() == "true")

# ── internal: build vLLM message content ────
async def _build_content(prompt: str, file_ids: list):
    """
    No files    → plain string prompt
    Text files  → file contents prepended as context, plain string
    Image files → multimodal list [image_url blocks..., text]
    Mixed       → images + text context + prompt as list
    """
    if not file_ids:
        return prompt

    image_blocks = []
    text_parts   = []

    for fid in file_ids:
        meta = file_store.get(fid)
        if not meta:
            log(f"Unknown file_id {fid} — skipping")
            continue

        if meta["kind"] == "image":
            raw = Path(meta["path"]).read_bytes()
            b64 = base64.b64encode(raw).decode()
            image_blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:{meta['mime']};base64,{b64}"},
            })
        else:
            txt = read_text_file(Path(meta["path"]), meta["mime"])
            text_parts.append(
                f"=== {meta['filename']} ===\n{txt}\n=== end ==="
            )

    # Text-only path: just a long string
    if not image_blocks:
        full = ("\n\n".join(text_parts) + "\n\n" + prompt) if text_parts else prompt
        return full

    # Multimodal path: images first, then context, then prompt
    content = image_blocks[:]
    if text_parts:
        content.append({"type": "text", "text": "\n\n".join(text_parts)})
    content.append({"type": "text", "text": prompt})
    return content

# ── internal: call vLLM ──────────────────────
async def _call_vllm(content, max_tokens, temperature, top_p,
                     stream, enable_thinking=True):
    payload = {
        "model":       MODEL,
        "messages":    [{"role": "user", "content": content}],
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "top_p":       top_p,
        "stream":      stream,
        # Qwen3.5 thinking mode toggle — passed via chat_template_kwargs
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }

    if stream:
        async def gen():
            async with httpx.AsyncClient(timeout=300) as c:
                async with c.stream(
                    "POST", "http://127.0.0.1:8001/v1/chat/completions",
                    json=payload
                ) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk
        return StreamingResponse(gen(), media_type="text/event-stream")

    last_err = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=180) as c:
                r = await c.post(
                    "http://127.0.0.1:8001/v1/chat/completions", json=payload
                )
            return JSONResponse(content=r.json(), status_code=r.status_code)
        except Exception as exc:
            last_err = exc
            log(f"vLLM attempt {attempt+1}/3: {exc}")
            await asyncio.sleep(1)

    return JSONResponse({"error": f"Backend failed: {last_err}"}, status_code=500)

# ── OpenAI-compatible transparent proxy ──────
# Preserves streaming, status codes, headers.
# Lets any OpenAI SDK client point at this server.
@app.api_route("/v1/{path:path}", methods=["GET", "POST", "DELETE"])
async def proxy(req: Request, path: str):
    body    = await req.body()
    headers = {k: v for k, v in req.headers.items() if k.lower() != "host"}
    wants_stream = b'"stream":true' in body or b'"stream": true' in body

    if wants_stream:
        async def sg():
            async with httpx.AsyncClient(timeout=300) as c:
                async with c.stream(
                    req.method,
                    f"http://127.0.0.1:8001/v1/{path}",
                    content=body, headers=headers,
                ) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk
        return StreamingResponse(sg(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=300) as c:
        r = await c.request(req.method,
                            f"http://127.0.0.1:8001/v1/{path}",
                            content=body, headers=headers)
    try:
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except Exception:
        return JSONResponse({"error": "non-JSON response"}, status_code=r.status_code)

# ── cloudflared tunnel ───────────────────────
def install_cloudflared():
    if shutil.which("cloudflared"):
        return
    log("Installing cloudflared …")
    deb = ("https://github.com/cloudflare/cloudflared/releases/latest"
           "/download/cloudflared-linux-amd64.deb")
    subprocess.run(
        f"wget -q {deb} -O /tmp/cloudflared.deb && dpkg -i /tmp/cloudflared.deb",
        shell=True, check=True,
    )
    log("cloudflared ready.")

def start_tunnel():
    install_cloudflared()
    while True:
        try:
            p = subprocess.Popen(
                ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8000"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            for line in p.stdout:
                m = re.search(r"https://[a-z0-9\-]+\.trycloudflare\.com", line)
                if m:
                    u = m.group(0)
                    log("=" * 60)
                    log(f"  BASE URL    : {u}")
                    log(f"  health      : GET    {u}/health")
                    log(f"  upload file : POST   {u}/upload")
                    log(f"  list files  : GET    {u}/files")
                    log(f"  download    : GET    {u}/files/<id>")
                    log(f"  delete      : DELETE {u}/files/<id>")
                    log(f"  chat        : POST   {u}/chat")
                    log(f"  chat+upload : POST   {u}/chat/upload  (multipart)")
                    log(f"  OpenAI SDK  : base_url={u}/v1")
                    log("=" * 60)
            p.wait()
        except Exception as exc:
            log(f"Tunnel error: {exc}")
        log("Tunnel exited — restarting in 10 s")
        time.sleep(10)

threading.Thread(target=start_tunnel, daemon=True).start()

# ── watchdog: restart vLLM on crash ─────────
def watchdog():
    global PROC
    while True:
        if PROC.poll() is not None:
            log(f"vLLM exited (code {PROC.returncode}) — restarting …")
            PROC = start_vllm()
            wait_ready()
        time.sleep(20)

threading.Thread(target=watchdog, daemon=True).start()

# ── warmup ───────────────────────────────────
def warmup():
    time.sleep(6)
    try:
        r = httpx.post("http://127.0.0.1:8000/chat",
                       json={"prompt": "Hello.", "max_tokens": 8,
                             "enable_thinking": False},
                       timeout=60)
        log(f"Warmup: {r.status_code}")
    except Exception as exc:
        log(f"Warmup (non-fatal): {exc}")

threading.Thread(target=warmup, daemon=True).start()

# ── start ────────────────────────────────────
log(f"Starting server — model={MODEL}  max_len={CFG['max_len']}  hw={HW}")
uvicorn.run(app, host="0.0.0.0", port=8000)
