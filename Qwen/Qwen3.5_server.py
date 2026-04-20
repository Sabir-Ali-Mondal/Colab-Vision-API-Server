# ============================================================
# Qwen3.5
# ============================================================

COLAB_CONFIG: dict = {
    "model":                  "4b",
    "max_tokens":             8192,
    "thinking":               True,
    "max_model_len":          8192,
    "api_key":                "",
    "rate_limit_rpm":         60,
    "tunnel":                 True,
    "drive_cache_dir":        "/content/drive/MyDrive/qwen35_cache",
    "gpu_memory_utilization": 0.85,
}

import base64
import gc
import io
import logging
import mimetypes
import os
import re
import sys
import time
import threading
import subprocess
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

VLLM_PORT  = 8000
PROXY_PORT = 8090  # 8080 is reserved by Colab

print("=" * 55)
print(" Qwen3.5 Gateway v5.0 starting ...")
print("=" * 55 + "\n")

gc.collect()

# ── Keepalive ─────────────────────────────────────────────────
_ALIVE = True
def _keepalive():
    while _ALIVE:
        try: sys.stderr.write("\x1b[0m"); sys.stderr.flush()
        except Exception: pass
        time.sleep(20)
threading.Thread(target=_keepalive, daemon=True, name="keepalive").start()
print("[Init] Keepalive started")

# ── 1. Model cache ────────────────────────────────────────────
print("\n[1/4] Setting up model cache ...")

def setup_cache() -> Path:
    drive_cache = Path(COLAB_CONFIG["drive_cache_dir"])
    drive_root  = Path("/content/drive/MyDrive")
    local_cache = Path("/tmp/qwen35_cache")
    try:
        if drive_root.exists() and any(drive_root.iterdir()):
            drive_cache.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(drive_cache)
            print(f"  Drive mounted — cache: {drive_cache}")
            return drive_cache
    except Exception:
        pass
    try:
        from google.colab import drive as _d
        _d.mount("/content/drive", force_remount=False)
        time.sleep(3)
        if drive_root.exists():
            drive_cache.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(drive_cache)
            print(f"  Drive mounted — cache: {drive_cache}")
            return drive_cache
    except Exception as e:
        print(f"  Drive unavailable ({e})")
    local_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(local_cache)
    print(f"  Using local cache: {local_cache}")
    return local_cache

cache_root = setup_cache()

MODELS   = {"2b": "Qwen/Qwen3.5-2B", "4b": "Qwen/Qwen3.5-4B"}
model_id = MODELS[COLAB_CONFIG["model"]]

def resolve_model(mid: str) -> tuple[str, bool]:
    slug      = mid.replace("/", "--")
    snap_root = cache_root / "hub" / f"models--{slug}" / "snapshots"
    if snap_root.exists():
        snaps = sorted([p for p in snap_root.glob("*/") if p.is_dir()],
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if snaps:
            gb = sum(f.stat().st_size for f in snaps[0].rglob("*") if f.is_file()) / 1e9
            print(f"  Cached : {snaps[0]}  ({gb:.1f} GB)")
            return str(snaps[0]), True
    print(f"  Not cached — will download: {mid}")
    return mid, False

model_path, from_cache = resolve_model(model_id)
print(f"  Model  : {model_id}")
if from_cache:
    print("  ETA    : ~3-4 min")
else:
    print("  ETA    : ~12-18 min (first run, downloading ~8 GB)")
print("[1/4] Done")

# ── 2. Install deps ───────────────────────────────────────────
print("\n[2/4] Checking dependencies ...")

def pip_install(*pkgs):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *pkgs],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

_PACKAGES = [
    ("fastapi",         "fastapi>=0.110"),
    ("uvicorn",         "uvicorn[standard]"),
    ("httpx",           "httpx"),
    ("PIL",             "Pillow"),
    ("transformers",    "transformers"),
    ("huggingface_hub", "huggingface_hub"),
    ("pypdf",           "pypdf"),
    ("pdfplumber",      "pdfplumber"),
    ("aiofiles",        "aiofiles"),
]

total = len(_PACKAGES) + 1
for i, (imp, pkg) in enumerate(_PACKAGES, 1):
    filled = int(25 * i / total)
    bar    = "█" * filled + "░" * (25 - filled)
    print(f"\r  [{bar}] {int(100*i/total):3d}%  {pkg:<30}", end="", flush=True)
    try: __import__(imp)
    except ImportError: pip_install(pkg)

bar = "█" * 25
print(f"\r  [{bar}] 100%  vllm                          ", flush=True)
try:
    import vllm
    print(f"  vllm {vllm.__version__} already installed")
except ImportError:
    print("\n  Installing vllm nightly (3-5 min first time) ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "--extra-index-url", "https://wheels.vllm.ai/nightly", "vllm"
    ])
    print("  vllm installed")

print("[2/4] Done")

# ── 3. Launch vLLM ───────────────────────────────────────────
print(f"\n[3/4] Starting vLLM ({'from cache' if from_cache else 'downloading'}) ...")

max_ctx       = int(COLAB_CONFIG.get("max_model_len", 8192))
vllm_log_path = Path("/tmp/vllm.log")
vllm_log_path.write_text("")

vllm_cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model",                  model_path,
    "--port",                   str(VLLM_PORT),
    "--host",                   "127.0.0.1",
    "--tensor-parallel-size",   "1",
    "--max-model-len",          str(max_ctx),
    "--gpu-memory-utilization", str(COLAB_CONFIG.get("gpu_memory_utilization", 0.85)),
    "--reasoning-parser",       "qwen3",
    "--enable-auto-tool-choice",
    "--tool-call-parser",       "qwen3_coder",
    "--trust-remote-code",
    "--served-model-name",      model_path,
    "--no-enable-log-requests",
    "--enforce-eager",
    "--limit-mm-per-prompt",    '{"image": 5, "video": 1}',
    "--media-io-kwargs",        '{"video": {"num_frames": -1}}',
]
vllm_env = os.environ.copy()
vllm_env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

print(f"  CMD: {' '.join(vllm_cmd)}\n")

with open(vllm_log_path, "w") as _vf:
    vllm_proc = subprocess.Popen(vllm_cmd, env=vllm_env, stdout=_vf, stderr=_vf)
print(f"  PID: {vllm_proc.pid}")

# Live-tail vLLM log
def _tail_log():
    try:
        with open(vllm_log_path) as f:
            f.seek(0, 2)
            while True:
                line = f.readline()
                if line: print(f"  [vllm] {line.rstrip()}", flush=True)
                else:    time.sleep(0.25)
    except Exception:
        pass
threading.Thread(target=_tail_log, daemon=True, name="vllm-tail").start()

# Poll /health
print("  Waiting for vLLM /health (timeout: 1200s) ...")
deadline = time.time() + 1200
tick     = 0
spinner  = ["|", "/", "-", "\\"]

while time.time() < deadline:
    if vllm_proc.poll() is not None:
        print(f"\n  [ERROR] vLLM crashed rc={vllm_proc.returncode}")
        print("  Last 50 lines of /tmp/vllm.log:")
        for line in vllm_log_path.read_text(errors="replace").splitlines()[-50:]:
            print(f"    {line}")
        raise RuntimeError("vLLM failed to start — see log above")
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{VLLM_PORT}/health", timeout=2
        ):
            print(f"\r  ✓ vLLM READY in {tick*3}s" + " "*40, flush=True)
            break
    except Exception:
        elapsed = tick * 3
        filled  = min(int(elapsed / 1200 * 30), 30)
        bar     = "█" * filled + "░" * (30 - filled)
        print(f"\r  {spinner[tick%4]} [{bar}] {elapsed}s / 1200s  ",
              end="", flush=True)
        time.sleep(3)
        tick += 1
else:
    raise TimeoutError("vLLM did not start within 1200s")

print("[3/4] Done\n")

# ── 4. FastAPI gateway ────────────────────────────────────────
print("[4/4] Building API gateway ...")

_LOG_PATH = Path("/tmp/qwen_server.log")
_LOG_PATH.touch(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(_LOG_PATH))
    ],
)
log = logging.getLogger("qwen")

max_tokens  = COLAB_CONFIG.get("max_tokens", 8192)
thinking    = COLAB_CONFIG.get("thinking", True)
api_key     = COLAB_CONFIG.get("api_key", "")
rpm         = COLAB_CONFIG.get("rate_limit_rpm", 60)
no_tunnel   = not COLAB_CONFIG.get("tunnel", True)
vllm_base   = f"http://127.0.0.1:{VLLM_PORT}"
_start_time = time.time()
_req_log: deque = deque(maxlen=500)
THINK_RE    = re.compile(r"<think>.*?</think>", re.DOTALL)

_rate_hits: dict[str, deque] = defaultdict(deque)
_rate_lock  = threading.Lock()

def rate_check(ip: str) -> tuple[bool, int]:
    if rpm <= 0: return True, 0
    now = time.time()
    with _rate_lock:
        q = _rate_hits[ip]
        while q and now - q[0] > 60.0: q.popleft()
        if len(q) >= rpm: return False, int(60 - (now - q[0])) + 1
        q.append(now); return True, 0

_MAGIC_BYTES = [
    (b"\xff\xd8\xff","image/jpeg"),(b"\x89PNG\r\n","image/png"),
    (b"GIF87a","image/gif"),(b"GIF89a","image/gif"),(b"BM","image/bmp"),
    (b"\x49\x49\x2a\x00","image/tiff"),(b"\x4d\x4d\x00\x2a","image/tiff"),
    (b"RIFF","video/webm"),(b"\x1aE\xdf\xa3","video/webm"),(b"%PDF","application/pdf"),
]

def sniff_mime(raw: bytes) -> str:
    for magic, mime in _MAGIC_BYTES:
        if raw[:len(magic)] == magic: return mime
    if raw[4:8] == b"ftyp": return "video/mp4"
    if raw[:4] in (b"\x00\x00\x00\x18", b"\x00\x00\x00\x1c"): return "video/mp4"
    return "application/octet-stream"

def pdf_to_text(raw: bytes) -> str:
    parts = []
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for pg in pdf.pages:
                t = pg.extract_text()
                if t: parts.append(t)
        if parts: return "\n\n".join(parts)
    except Exception: pass
    try:
        import pypdf
        for pg in pypdf.PdfReader(io.BytesIO(raw)).pages:
            t = pg.extract_text()
            if t: parts.append(t)
        if parts: return "\n\n".join(parts)
    except Exception: pass
    return "[PDF: text extraction failed]"

def to_content_block(source: str) -> dict:
    raw, mime = None, ""
    if source.startswith("data:"):
        try:
            hdr, b64 = source.split(",", 1)
            mime = hdr.split(":")[1].split(";")[0]
            raw  = base64.b64decode(b64)
        except Exception:
            return {"type": "text", "text": "[invalid data URL]"}
    elif source.startswith(("http://","https://")):
        ext = source.split("?")[0].rsplit(".",1)[-1].lower()
        if ext in ("mp4","mov","avi","webm","mkv","flv"):
            return {"type":"video_url","video_url":{"url":source}}
        if ext == "pdf":
            try:
                with urllib.request.urlopen(source, timeout=30) as r:
                    raw = r.read(); mime = "application/pdf"
            except Exception:
                return {"type":"text","text":f"[fetch failed: {source}]"}
        else:
            return {"type":"image_url","image_url":{"url":source}}
    else:
        if source.startswith("file://"): source = source[7:]
        p = Path(source)
        if p.exists():
            raw  = p.read_bytes()
            mime = mimetypes.guess_type(str(p))[0] or sniff_mime(raw)
        else:
            try: raw = base64.b64decode(source); mime = sniff_mime(raw)
            except Exception: return {"type":"text","text":f"[not found: {source}]"}
    if raw is None: return {"type":"text","text":"[empty source]"}
    if not mime:    mime = sniff_mime(raw)
    if mime == "application/pdf":
        return {"type":"text","text":"[PDF extracted text]\n" + pdf_to_text(raw)}
    d_url = f"data:{mime};base64,{base64.b64encode(raw).decode()}"
    if mime.startswith("video/"): return {"type":"video_url","video_url":{"url":d_url}}
    return {"type":"image_url","image_url":{"url":d_url}}

def sampling_params(use_think: bool, task: str = "general") -> dict:
    if use_think:
        if task == "coding":    return dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0)
        if task == "reasoning": return dict(temperature=1.0, top_p=1.0,  top_k=40, min_p=0.0, presence_penalty=2.0)
        return dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5)
    if task == "reasoning":     return dict(temperature=1.0, top_p=1.0,  top_k=40, min_p=0.0, presence_penalty=2.0)
    return dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5)

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
import httpx

app = FastAPI(title="Qwen3.5 Gateway", version="5.0.0")
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def _ip(req: Request) -> str:
    xff = req.headers.get("x-forwarded-for", "")
    return xff.split(",")[0].strip() if xff else (
        req.client.host if req.client else "unknown")

def _guard(req: Request) -> None:
    if api_key:
        tok = req.headers.get("authorization","").removeprefix("Bearer ").strip()
        if tok != api_key: raise HTTPException(401, "Invalid or missing API key")
    ok, retry = rate_check(_ip(req))
    if not ok: raise HTTPException(429, f"Rate limit — retry after {retry}s",
                                   headers={"Retry-After": str(retry)})

@app.middleware("http")
async def _log_mw(req: Request, call_next):
    t0 = time.time(); resp = await call_next(req)
    ms = int((time.time()-t0)*1000)
    _req_log.append({"ts":time.strftime("%H:%M:%S"),"path":req.url.path,
                     "status":resp.status_code,"ms":ms})
    log.info("[API] %s %s -> %d  %dms", req.method, req.url.path, resp.status_code, ms)
    return resp

@app.get("/")
async def root():
    return {"service":"Qwen3.5 Gateway","status":"ok","version":"5.0.0"}

@app.get("/health")
async def health():
    vok = False
    try:
        with urllib.request.urlopen(f"{vllm_base}/health", timeout=2): vok = True
    except Exception: pass
    return {"status":"ok" if vok else "degraded",
            "vllm":"ready" if vok else "unavailable",
            "model":model_id,"thinking":thinking,
            "max_tokens":max_tokens,"max_ctx_len":max_ctx,
            "uptime_s":int(time.time()-_start_time),
            "total_reqs":len(_req_log)}

@app.get("/logs")
async def get_logs(request: Request, n: int = 100):
    if api_key:
        tok = request.headers.get("authorization","").removeprefix("Bearer ").strip()
        if tok != api_key: raise HTTPException(401, "Unauthorized")
    return PlainTextResponse(
        "\n".join(_LOG_PATH.read_text(errors="replace").splitlines()[-n:]))

@app.get("/logs/vllm")
async def get_vllm_logs(request: Request, n: int = 100):
    if api_key:
        tok = request.headers.get("authorization","").removeprefix("Bearer ").strip()
        if tok != api_key: raise HTTPException(401, "Unauthorized")
    lines = (vllm_log_path.read_text(errors="replace").splitlines()[-n:]
             if vllm_log_path.exists() else ["(empty)"])
    return PlainTextResponse("\n".join(lines))

@app.get("/v1/models")
async def list_models(request: Request):
    _guard(request)
    return {"object":"list","data":[{
        "id":model_id,"object":"model",
        "created":int(_start_time),"owned_by":"qwen",
        "context_window":max_ctx,"capabilities":["text","image","video"]}]}

@app.post("/v1/agent")
async def agent_endpoint(request: Request):
    _guard(request)
    try: body = await request.json()
    except Exception: raise HTTPException(400, "Invalid JSON")
    prompt    = body.get("prompt","")
    files     = body.get("files",[])
    system    = body.get("system","You are a helpful AI assistant.")
    use_think = body.get("thinking", thinking)
    task      = body.get("task","general")
    tokens    = int(body.get("max_tokens", max_tokens))
    stream    = bool(body.get("stream", False))
    content   = [to_content_block(str(s)) for s in files]
    if prompt: content.append({"type":"text","text":prompt})
    if not content: raise HTTPException(400, "Provide at least a prompt or file")
    user_c  = (content[0]["text"]
               if len(content)==1 and content[0].get("type")=="text"
               else content)
    payload = {"model":model_id,
               "messages":[{"role":"system","content":system},
                            {"role":"user","content":user_c}],
               "max_tokens":min(tokens,max_ctx),"stream":stream,
               "chat_template_kwargs":{"enable_thinking":use_think},
               **sampling_params(use_think,task)}
    async with httpx.AsyncClient(timeout=600) as cli:
        if stream:
            async def _gen():
                async with cli.stream("POST",
                    f"{vllm_base}/v1/chat/completions", json=payload) as r:
                    async for chunk in r.aiter_bytes(): yield chunk
            return StreamingResponse(_gen(), media_type="text/event-stream")
        r    = await cli.post(f"{vllm_base}/v1/chat/completions", json=payload)
        data = r.json()
        if not use_think and "choices" in data:
            for ch in data["choices"]:
                c = ch.get("message",{}).get("content")
                if c: ch["message"]["content"] = THINK_RE.sub("",c).strip()
        return JSONResponse(data)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _guard(request)
    try: body = await request.json()
    except Exception: raise HTTPException(400, "Invalid JSON")
    body["model"] = model_id
    body.setdefault("max_tokens", max_tokens)
    body["max_tokens"] = min(int(body["max_tokens"]), max_ctx)
    body.setdefault("chat_template_kwargs", {"enable_thinking": thinking})
    for msg in body.get("messages",[]):
        if isinstance(msg.get("content"), list):
            new = []
            for blk in msg["content"]:
                if blk.get("type") == "file":
                    fd  = blk.get("file",{})
                    src = fd.get("file_data") or fd.get("url","")
                    new.append(to_content_block(str(src)))
                else:
                    new.append(blk)
            msg["content"] = new
    stream = bool(body.get("stream", False))
    async with httpx.AsyncClient(timeout=600) as cli:
        if stream:
            async def _gen():
                async with cli.stream("POST",
                    f"{vllm_base}/v1/chat/completions", json=body) as r:
                    async for chunk in r.aiter_bytes(): yield chunk
            return StreamingResponse(_gen(), media_type="text/event-stream")
        r    = await cli.post(f"{vllm_base}/v1/chat/completions", json=body)
        data = r.json()
        if not thinking and "choices" in data:
            for ch in data["choices"]:
                c = ch.get("message",{}).get("content")
                if c: ch["message"]["content"] = THINK_RE.sub("",c).strip()
        return JSONResponse(data, status_code=r.status_code)

@app.api_route("/v1/{path:path}",
               methods=["GET","POST","PUT","DELETE","PATCH","OPTIONS"])
async def passthrough(path: str, request: Request):
    _guard(request)
    url = f"{vllm_base}/v1/{path}"
    try: body_bytes = await request.body()
    except Exception: body_bytes = b""
    fwd = {k:v for k,v in request.headers.items()
           if k.lower() not in ("host","content-length","authorization")}
    async with httpx.AsyncClient(timeout=600) as cli:
        resp = await cli.request(request.method, url,
                                 headers=fwd, content=body_bytes)
        ct = resp.headers.get("content-type","")
        if "text/event-stream" in ct:
            async def _gen():
                async for chunk in resp.aiter_bytes(): yield chunk
            return StreamingResponse(_gen(), media_type="text/event-stream",
                                     status_code=resp.status_code)
        try:    return JSONResponse(resp.json(), status_code=resp.status_code)
        except: return PlainTextResponse(resp.text, status_code=resp.status_code)

print("[4/4] Gateway ready\n")

# ── Start uvicorn ─────────────────────────────────────────────
import uvicorn
_uv_config = uvicorn.Config(app, host="0.0.0.0", port=PROXY_PORT,
                             log_level="warning", access_log=False)
_uv_server = uvicorn.Server(_uv_config)
threading.Thread(target=_uv_server.run, daemon=True, name="uvicorn").start()
time.sleep(2)
print(f"[START] uvicorn on 0.0.0.0:{PROXY_PORT}\n")

# ── Cloudflare tunnel ─────────────────────────────────────────
if not no_tunnel:
    cf_bin = Path("/usr/local/bin/cloudflared")
    if not cf_bin.exists():
        print("[Tunnel] Installing cloudflared ...")
        subprocess.check_call([
            "wget", "-q", "-O", str(cf_bin),
            "https://github.com/cloudflare/cloudflared/releases/latest"
            "/download/cloudflared-linux-amd64"
        ])
        cf_bin.chmod(0o755)

    print("[Tunnel] Starting ...")
    _cf_proc = subprocess.Popen(
        ["cloudflared","tunnel","--url",f"http://localhost:{PROXY_PORT}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    _cf_pat      = re.compile(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com")
    _cf_deadline = time.time() + 90
    _public_url  = ""
    for _line in _cf_proc.stdout:
        if time.time() > _cf_deadline:
            print("[Tunnel] URL not found in 90s"); break
        m = _cf_pat.search(_line)
        if m:
            _public_url = m.group(0)
            Path("/tmp/public_url.txt").write_text(_public_url)
            sep = "=" * 60
            print(sep)
            print(f"  PUBLIC URL           : {_public_url}")
            print(f"  /v1/chat/completions : {_public_url}/v1/chat/completions")
            print(f"  /v1/agent            : {_public_url}/v1/agent")
            print(f"  /v1/models           : {_public_url}/v1/models")
            print(f"  /health              : {_public_url}/health")
            print(sep); break
else:
    print(f"[Tunnel] Disabled — local: http://localhost:{PROXY_PORT}")

# ── vLLM watchdog ─────────────────────────────────────────────
def _watchdog():
    global vllm_proc
    while True:
        time.sleep(20)
        if vllm_proc.poll() is not None:
            rc = vllm_proc.returncode
            log.warning("[watchdog] vLLM exited rc=%d — restarting ...", rc)
            try:
                time.sleep(2)
                with open(vllm_log_path, "w") as f:
                    vllm_proc = subprocess.Popen(
                        vllm_cmd, env=vllm_env, stdout=f, stderr=f)
                log.info("[watchdog] vLLM restarted PID=%d", vllm_proc.pid)
            except Exception as exc:
                log.error("[watchdog] Restart failed: %s", exc)
threading.Thread(target=_watchdog, daemon=True, name="vllm-watchdog").start()

# ── Status ping loop ──────────────────────────────────────────
print("\n[RUNNING] Server live. Interrupt cell to stop.\n")
try:
    while True:
        time.sleep(60)
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{VLLM_PORT}/health", timeout=2):
                status = "OK"
        except Exception:
            status = "DOWN"
        print(f"  [ping {time.strftime('%H:%M:%S')}] vLLM={status}"
              f"  reqs={len(_req_log)}", flush=True)
except KeyboardInterrupt:
    _ALIVE = False
    print("\n[STOP] Done.")
