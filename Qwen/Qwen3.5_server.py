#!/usr/bin/env python3
# fmt: off

print(">>> INITIALIZING SCRIPT: Please wait, loading configuration and modules... <<<")

COLAB_CONFIG: dict = {
    "model":           "4b",
    "max_tokens":      32768,
    "thinking":        True,
    "max_model_len":   32768,
    "api_key":         "",
    "rate_limit_rpm":  60,
    "tunnel":          True,
}

import argparse
import base64
import importlib
import io
import json
import logging
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, AsyncGenerator

_LOG_PATH = Path("/tmp/qwen_server.log")
_LOG_PATH.touch(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(_LOG_PATH)),
    ],
)
log = logging.getLogger("qwen")

MODELS: dict[str, str] = {
    "2b": "Qwen/Qwen3.5-2B",
    "4b": "Qwen/Qwen3.5-4B",
}
VLLM_PORT     = 8000
PROXY_PORT    = 8080
_public_url   = ""
_vllm_proc:   subprocess.Popen | None = None
_cf_proc:     subprocess.Popen | None = None
_server_start = time.time()
_request_log: deque = deque(maxlen=500)

_PACKAGES: list[tuple[str, str]] = [
    ("fastapi",        "fastapi>=0.110"),
    ("uvicorn",        "uvicorn[standard]"),
    ("httpx",          "httpx"),
    ("PIL",            "Pillow"),
    ("transformers",   "transformers"),
    ("huggingface_hub","huggingface_hub"),
    ("pypdf",          "pypdf"),
    ("pdfplumber",     "pdfplumber"),
    ("aiofiles",       "aiofiles"),
]

def _pip(*pkgs: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *pkgs],
        stdout=subprocess.DEVNULL,
    )

def install_dependencies() -> None:
    log.info("1. Installing / verifying dependencies...")
    for import_name, pip_name in _PACKAGES:
        try:
            importlib.import_module(import_name)
            log.info(f"    [OK] {pip_name}")
        except ImportError:
            log.info(f"    [DL] {pip_name}")
            try:
                _pip(pip_name)
                log.info(f"    [OK] {pip_name} installed")
            except subprocess.CalledProcessError:
                log.warning(f"    [WARN] {pip_name} failed (optional)")

    try:
        import vllm
        log.info(f"    [OK] vllm {vllm.__version__}")
    except ImportError:
        log.info("    [DL] vllm nightly (3-5 min first run)...")
        _pip("vllm", "--extra-index-url", "https://wheels.vllm.ai/nightly")
        log.info("    [OK] vllm installed")

    try:
        importlib.import_module("pytesseract")
        log.info("    [OK] pytesseract (OCR)")
    except ImportError:
        try:
            _pip("pytesseract")
        except Exception:
            log.info("    [INFO] pytesseract unavailable - scanned-PDF OCR disabled")

def install_cloudflared() -> None:
    binary = Path("/usr/local/bin/cloudflared")
    if binary.exists():
        return
    log.info("    [DL] cloudflared...")
    url = (
        "https://github.com/cloudflare/cloudflared/releases/latest"
        "/download/cloudflared-linux-amd64"
    )
    subprocess.check_call(["wget", "-q", "-O", str(binary), url])
    binary.chmod(0o755)
    log.info("    [OK] cloudflared installed")

def detect_hardware() -> dict:
    info: dict[str, Any] = {
        "gpu_name":        "cpu",
        "vram_gb":         0.0,
        "cuda_available":  False,
        "recommended_len": 16384,
    }
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            name, mb_str = r.stdout.strip().splitlines()[0].split(",", 1)
            vram_gb = round(float(mb_str.strip()) / 1024, 1)
            info.update({
                "gpu_name":       name.strip(),
                "vram_gb":        vram_gb,
                "cuda_available": True,
                "recommended_len": (
                    262144 if vram_gb >= 70 else
                    131072 if vram_gb >= 35 else
                    65536  if vram_gb >= 20 else
                    32768  if vram_gb >= 14 else
                    16384
                ),
            })
    except Exception as exc:
        log.debug(f"nvidia-smi: {exc}")
    return info

def resolve_model(model_id: str) -> str:
    slug = model_id.replace("/", "--")
    gd = Path(f"/content/drive/MyDrive/hf_models/{slug}")
    if gd.exists():
        log.info(f"    [OK] model in Google Drive: {gd}")
        return str(gd)

    hf_root = Path.home() / ".cache" / "huggingface" / "hub"
    for snap in sorted((hf_root / f"models--{slug}" / "snapshots").glob("*/"),
                       reverse=True):
        if snap.is_dir():
            log.info(f"    [OK] model in HF cache: {snap}")
            return str(snap)

    log.info(f"    [DL] model will download from HuggingFace: {model_id}")
    return model_id

def start_vllm(model_path: str, max_model_len: int) -> subprocess.Popen:
    global _vllm_proc
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model",                  model_path,
        "--port",                   str(VLLM_PORT),
        "--host",                   "127.0.0.1",
        "--tensor-parallel-size",   "1",
        "--max-model-len",          str(max_model_len),
        "--reasoning-parser",       "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser",       "qwen3_coder",
        "--trust-remote-code",
        "--served-model-name",      model_path,
        "--no-enable-log-requests", # Corrected for your vLLM version
    ]
    env = os.environ.copy()
    env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

    log.info(f"    cmd: {' '.join(cmd)}")
    vllm_log = open("/tmp/vllm.log", "w")
    _vllm_proc = subprocess.Popen(cmd, env=env, stdout=vllm_log, stderr=vllm_log)
    log.info(f"    [OK] vLLM started (PID {_vllm_proc.pid})")
    return _vllm_proc

def wait_for_vllm(timeout: int = 900) -> None:
    log.info("    [WAIT] waiting for vLLM...")
    deadline = time.time() + timeout
    ticks = 0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{VLLM_PORT}/health", timeout=2
            ):
                log.info("    [OK] vLLM ready!\n")
                return
        except Exception:
            time.sleep(3)
            ticks += 1
            if ticks % 10 == 0:
                log.info(f"    ... ({ticks*3}s elapsed)")
    raise TimeoutError("vLLM did not become ready in time.")

def start_vllm_watchdog(model_path: str, max_model_len: int) -> None:
    def _loop() -> None:
        global _vllm_proc
        while True:
            time.sleep(15)
            if _vllm_proc and _vllm_proc.poll() is not None:
                rc = _vllm_proc.returncode
                log.warning(f"[WARN] vLLM exited (rc={rc}), restarting...")
                try:
                    _vllm_proc = start_vllm(model_path, max_model_len)
                    wait_for_vllm()
                    log.info("[OK] vLLM restarted")
                except Exception as exc:
                    log.error(f"vLLM restart failed: {exc}")
    threading.Thread(target=_loop, daemon=True, name="vllm-watchdog").start()

def start_cloudflare(port: int) -> str:
    global _public_url, _cf_proc
    install_cloudflared()
    log.info("    [WEB] starting Cloudflare tunnel...")

    _cf_proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    pat = re.compile(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com")
    deadline = time.time() + 90
    for line in _cf_proc.stdout:
        if time.time() > deadline:
            log.warning("    [WARN] tunnel URL not detected within 90 s")
            break
        m = pat.search(line)
        if m:
            _public_url = m.group(0)
            _print_url_banner(_public_url)
            Path("/tmp/public_url.txt").write_text(_public_url)
            break

    def _watch() -> None:
        global _cf_proc
        while True:
            time.sleep(10)
            if _cf_proc and _cf_proc.poll() is not None:
                log.warning("[WARN] Cloudflare tunnel died, restarting...")
                start_cloudflare(port)
                return
    threading.Thread(target=_watch, daemon=True, name="cf-watchdog").start()
    return _public_url

def _print_url_banner(url: str) -> None:
    log.info("\n==================================================================")
    log.info(f"  [LINK] PUBLIC URL       : {url}")
    log.info(f"  chat/completions        : {url}/v1/chat/completions")
    log.info(f"  agent (multimodal)      : {url}/v1/agent")
    log.info(f"  models                  : {url}/v1/models")
    log.info(f"  health                  : {url}/health")
    log.info("==================================================================\n")

_MAGIC_BYTES: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff",        "image/jpeg"),
    (b"\x89PNG\r\n",         "image/png"),
    (b"GIF87a",              "image/gif"),
    (b"GIF89a",              "image/gif"),
    (b"BM",                  "image/bmp"),
    (b"\x49\x49\x2a\x00",   "image/tiff"),
    (b"\x4d\x4d\x00\x2a",   "image/tiff"),
    (b"RIFF",                "video/webm"),
    (b"\x1aE\xdf\xa3",       "video/webm"),
    (b"%PDF",                "application/pdf"),
]

def sniff_mime(raw: bytes) -> str:
    for magic, mime in _MAGIC_BYTES:
        if raw[:len(magic)] == magic:
            return mime
    if raw[4:8] == b"ftyp":
        return "video/mp4"
    if raw[:4] in (b"\x00\x00\x00\x18", b"\x00\x00\x00\x1c"):
        return "video/mp4"
    return "application/octet-stream"

def pdf_to_text(raw: bytes) -> str:
    parts: list[str] = []
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
        if parts:
            return "\n\n".join(parts)
    except Exception as exc:
        log.debug(f"pdfplumber: {exc}")

    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(raw))
        for page in reader.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
        if parts:
            return "\n\n".join(parts)
    except Exception as exc:
        log.debug(f"pypdf: {exc}")

    try:
        import pytesseract
        from PIL import Image as PILImage
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(raw))
        for page in reader.pages:
            for img_obj in page.images:
                img = PILImage.open(io.BytesIO(img_obj.data))
                parts.append(pytesseract.image_to_string(img))
        if parts:
            return "\n\n".join(parts)
    except Exception as exc:
        log.debug(f"OCR: {exc}")

    return "[PDF: text extraction failed]"

def to_content_block(source: str) -> dict:
    raw: bytes | None = None
    mime: str = ""

    if source.startswith("data:"):
        try:
            header, b64part = source.split(",", 1)
            mime = header.split(":")[1].split(";")[0]
            raw  = base64.b64decode(b64part)
        except Exception:
            return {"type": "text", "text": "[invalid data URL]"}

    elif source.startswith("http://") or source.startswith("https://"):
        ext = source.split("?")[0].rsplit(".", 1)[-1].lower()
        if ext in ("mp4", "mov", "avi", "webm", "mkv", "flv"):
            return {"type": "video_url", "video_url": {"url": source}}
        if ext == "pdf":
            try:
                with urllib.request.urlopen(source, timeout=30) as r:
                    raw  = r.read()
                    mime = "application/pdf"
            except Exception:
                return {"type": "text", "text": f"[fetch failed: {source}]"}
        else:
            return {"type": "image_url", "image_url": {"url": source}}

    else:
        if source.startswith("file://"):
            source = source[7:]
        path = Path(source)
        if path.exists():
            raw  = path.read_bytes()
            mime = mimetypes.guess_type(str(path))[0] or sniff_mime(raw)
        else:
            try:
                raw  = base64.b64decode(source)
                mime = sniff_mime(raw)
            except Exception:
                return {"type": "text", "text": f"[not found: {source}]"}

    if raw is None:
        return {"type": "text", "text": "[empty source]"}

    if not mime:
        mime = sniff_mime(raw)

    if mime == "application/pdf":
        text = pdf_to_text(raw)
        return {"type": "text", "text": f"[PDF extracted text]\n{text}"}

    b64   = base64.b64encode(raw).decode()
    d_url = f"data:{mime};base64,{b64}"

    if mime.startswith("video/"):
        return {"type": "video_url", "video_url": {"url": d_url}}

    return {"type": "image_url", "image_url": {"url": d_url}}

class RateLimiter:
    def __init__(self, rpm: int) -> None:
        self.rpm   = rpm
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, ip: str) -> tuple[bool, int]:
        if self.rpm <= 0:
            return True, 0
        now = time.time()
        with self._lock:
            q = self._hits[ip]
            while q and now - q[0] > 60.0:
                q.popleft()
            if len(q) >= self.rpm:
                return False, int(60 - (now - q[0])) + 1
            q.append(now)
            return True, 0

def get_sampling(thinking: bool, task: str = "general") -> dict:
    if thinking:
        if task == "coding":
            return dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0)
        if task == "reasoning":
            return dict(temperature=1.0, top_p=1.0,  top_k=40, min_p=0.0, presence_penalty=2.0)
        return     dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5)
    else:
        if task == "reasoning":
            return dict(temperature=1.0, top_p=1.0,  top_k=40, min_p=0.0, presence_penalty=2.0)
        return     dict(temperature=0.7, top_p=0.8,  top_k=20, min_p=0.0, presence_penalty=1.5)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()

def build_app(
    model_id:    str,
    max_tokens:  int,
    thinking:    bool,
    api_key:     str,
    rate_lim:    RateLimiter,
    max_ctx_len: int,
) -> Any:
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
    import httpx

    app = FastAPI(title="Qwen3.5 Gateway", version="2.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    VLLM = f"http://127.0.0.1:{VLLM_PORT}"

    def _ip(req: Request) -> str:
        xff = req.headers.get("x-forwarded-for", "")
        return xff.split(",")[0].strip() if xff else (
            req.client.host if req.client else "unknown"
        )

    def _guard(req: Request) -> None:
        if api_key:
            token = req.headers.get("authorization", "").removeprefix("Bearer ").strip()
            if token != api_key:
                raise HTTPException(401, "Invalid or missing API key")
        ok, retry = rate_lim.check(_ip(req))
        if not ok:
            raise HTTPException(
                429, f"Rate limit exceeded - retry after {retry}s",
                headers={"Retry-After": str(retry)},
            )

    @app.middleware("http")
    async def _log(req: Request, call_next):
        t0   = time.time()
        resp = await call_next(req)
        ms   = int((time.time() - t0) * 1000)
        entry = {"ts": time.strftime("%H:%M:%S"), "method": req.method,
                 "path": req.url.path, "status": resp.status_code,
                 "ms": ms, "ip": _ip(req)}
        _request_log.append(entry)
        log.info(f"{req.method:6} {req.url.path:<35} {resp.status_code}  {ms}ms  [{_ip(req)}]")
        return resp

    @app.get("/health")
    async def health():
        ok = False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{VLLM_PORT}/health", timeout=2):
                ok = True
        except Exception:
            pass
        return {
            "status":      "ok" if ok else "degraded",
            "vllm":        "ready" if ok else "unavailable",
            "model":       model_id,
            "thinking":    thinking,
            "max_tokens":  max_tokens,
            "max_ctx_len": max_ctx_len,
            "public_url":  _public_url,
            "uptime_s":    int(time.time() - _server_start),
            "total_reqs":  len(_request_log),
        }

    @app.get("/logs")
    async def proxy_logs(request: Request, n: int = 100):
        if api_key:
            token = request.headers.get("authorization","").removeprefix("Bearer ").strip()
            if token != api_key:
                raise HTTPException(401, "Unauthorized")
        lines = _LOG_PATH.read_text(errors="replace").splitlines()[-n:]
        return PlainTextResponse("\n".join(lines))

    @app.get("/logs/vllm")
    async def vllm_logs(request: Request, n: int = 100):
        if api_key:
            token = request.headers.get("authorization","").removeprefix("Bearer ").strip()
            if token != api_key:
                raise HTTPException(401, "Unauthorized")
        p = Path("/tmp/vllm.log")
        lines = (p.read_text(errors="replace").splitlines()[-n:]
                 if p.exists() else ["(empty)"])
        return PlainTextResponse("\n".join(lines))

    @app.get("/v1/models")
    async def list_models(request: Request):
        _guard(request)
        return {
            "object": "list",
            "data": [{
                "id":             model_id,
                "object":         "model",
                "created":        int(_server_start),
                "owned_by":       "qwen",
                "context_window": max_ctx_len,
            }],
        }

    @app.post("/v1/agent")
    async def agent(request: Request):
        _guard(request)

        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "Invalid JSON body")

        prompt     = body.get("prompt", "")
        files      = body.get("files", [])
        system_msg = body.get("system", "You are a helpful AI assistant.")
        use_think  = body.get("thinking", thinking)
        task       = body.get("task", "general")
        tokens     = int(body.get("max_tokens", max_tokens))
        stream     = bool(body.get("stream", False))

        content: list[dict] = []
        for src in files:
            content.append(to_content_block(str(src)))
        if prompt:
            content.append({"type": "text", "text": prompt})
        if not content:
            raise HTTPException(400, "Provide at least a prompt or a file")

        user_content = content if len(content) > 1 else content[0].get("text", prompt)

        sp = get_sampling(use_think, task)
        payload: dict = {
            "model":                    model_id,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_content},
            ],
            "max_tokens":               min(tokens, max_ctx_len),
            "stream":                   stream,
            "chat_template_kwargs":     {"enable_thinking": use_think},
            **sp,
        }

        async with httpx.AsyncClient(timeout=600) as cli:
            if stream:
                async def _gen() -> AsyncGenerator[bytes, None]:
                    async with cli.stream(
                        "POST", f"{VLLM}/v1/chat/completions", json=payload
                    ) as r:
                        async for chunk in r.aiter_bytes():
                            yield chunk
                return StreamingResponse(_gen(), media_type="text/event-stream")

            r = await cli.post(f"{VLLM}/v1/chat/completions", json=payload)
            data = r.json()
            if not use_think and "choices" in data:
                for ch in data["choices"]:
                    c = ch.get("message", {}).get("content")
                    if c:
                        ch["message"]["content"] = strip_think(c)
            return JSONResponse(data)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        _guard(request)
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "Invalid JSON body")

        body["model"] = model_id
        body.setdefault("max_tokens", max_tokens)
        body["max_tokens"] = min(body["max_tokens"], max_ctx_len)
        body.setdefault("chat_template_kwargs", {"enable_thinking": thinking})

        for msg in body.get("messages", []):
            if isinstance(msg.get("content"), list):
                new: list[dict] = []
                for blk in msg["content"]:
                    if blk.get("type") == "file":
                        fd = blk.get("file", {})
                        src = fd.get("file_data") or fd.get("url", "")
                        new.append(to_content_block(src))
                    else:
                        new.append(blk)
                msg["content"] = new

        stream = bool(body.get("stream", False))

        async with httpx.AsyncClient(timeout=600) as cli:
            if stream:
                async def _gen() -> AsyncGenerator[bytes, None]:
                    async with cli.stream(
                        "POST", f"{VLLM}/v1/chat/completions", json=body
                    ) as r:
                        async for chunk in r.aiter_bytes():
                            yield chunk
                return StreamingResponse(_gen(), media_type="text/event-stream")

            r = await cli.post(f"{VLLM}/v1/chat/completions", json=body)
            data = r.json()
            if not thinking and "choices" in data:
                for ch in data["choices"]:
                    c = ch.get("message", {}).get("content")
                    if c:
                        ch["message"]["content"] = strip_think(c)
            return JSONResponse(data, status_code=r.status_code)

    @app.api_route("/v1/{path:path}",
                   methods=["GET","POST","PUT","DELETE","PATCH","OPTIONS"])
    async def passthrough(path: str, request: Request):
        _guard(request)
        url = f"{VLLM}/v1/{path}"
        try:
            body_bytes = await request.body()
        except Exception:
            body_bytes = b""

        fwd = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length", "authorization")}

        async with httpx.AsyncClient(timeout=600) as cli:
            resp = await cli.request(
                request.method, url, headers=fwd, content=body_bytes
            )
            ct = resp.headers.get("content-type", "")
            if "text/event-stream" in ct:
                async def _gen() -> AsyncGenerator[bytes, None]:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
                return StreamingResponse(_gen(), media_type="text/event-stream",
                                         status_code=resp.status_code)
            try:
                return JSONResponse(resp.json(), status_code=resp.status_code)
            except Exception:
                return PlainTextResponse(resp.text, status_code=resp.status_code)

    return app

def _parse_args():
    p = argparse.ArgumentParser(description="Qwen3.5 Colab Server", add_help=False)
    p.add_argument("--model",        choices=["2b","4b"],  default=None)
    p.add_argument("--max-tokens",   type=int,             default=None)
    p.add_argument("--max-ctx-len",  type=int,             default=None)
    p.add_argument("--no-thinking",  action="store_true",  default=False)
    p.add_argument("--no-tunnel",    action="store_true",  default=False)
    p.add_argument("--api-key",                            default=None)
    p.add_argument("--rpm",          type=int,             default=None)
    p.add_argument("--proxy-port",   type=int,             default=PROXY_PORT)
    args, _ = p.parse_known_args()
    return args

def _pick_model() -> str:
    print("\nSelect Qwen3.5 Model:")
    print("  [1] Qwen3.5-2B  (fast,  ~4 GB VRAM)")
    print("  [2] Qwen3.5-4B  (smart, ~8 GB VRAM)")
    while True:
        c = input("Choice [1/2]: ").strip()
        if c == "1": return "2b"
        if c == "2": return "4b"

def main() -> None:
    args = _parse_args()

    model_key  = args.model or COLAB_CONFIG.get("model") or _pick_model()
    model_id   = MODELS[model_key]
    max_tokens = args.max_tokens or COLAB_CONFIG.get("max_tokens", 32768)
    thinking   = (not args.no_thinking) if args.no_thinking else COLAB_CONFIG.get("thinking", True)
    no_tunnel  = args.no_tunnel or (not COLAB_CONFIG.get("tunnel", True))
    api_key    = args.api_key   or COLAB_CONFIG.get("api_key", "")
    rpm        = args.rpm       if args.rpm is not None else COLAB_CONFIG.get("rate_limit_rpm", 60)
    proxy_port = args.proxy_port

    log.info("2. Hardware detection...")
    hw = detect_hardware()
    log.info(f"    GPU: {hw['gpu_name']}  VRAM: {hw['vram_gb']} GB  CUDA: {hw['cuda_available']}")

    cfg_ctx    = args.max_ctx_len or COLAB_CONFIG.get("max_model_len", hw["recommended_len"])
    max_ctx    = min(int(cfg_ctx), 262144)

    log.info("\n--- Server Configuration ---")
    log.info(f"model        : {model_id}")
    log.info(f"max_tokens   : {max_tokens}")
    log.info(f"max_ctx_len  : {max_ctx}")
    log.info(f"thinking     : {str(thinking)}")
    log.info(f"api_key      : {'<set>' if api_key else '<none - open access>'}")
    log.info(f"rate_limit   : {(str(rpm)+' rpm/IP') if rpm else 'disabled'}")
    log.info(f"tunnel       : {str(not no_tunnel)}")
    log.info("----------------------------\n")

    log.info("1. Dependencies")
    install_dependencies()

    log.info("3. Model resolver")
    model_path = resolve_model(model_id)

    log.info("4. vLLM Start")
    start_vllm(model_path, max_ctx)
    wait_for_vllm()
    start_vllm_watchdog(model_path, max_ctx)

    log.info("9. FastAPI proxy")
    rl  = RateLimiter(rpm)
    app = build_app(model_id, max_tokens, thinking, api_key, rl, max_ctx)
    log.info("    [OK] app built")

    if not no_tunnel:
        log.info("5. Cloudflare tunnel (starting after uvicorn binds...)")
        def _start():
            time.sleep(5)
            start_cloudflare(proxy_port)
        threading.Thread(target=_start, daemon=True, name="cf-starter").start()
    else:
        log.info(f"5. Tunnel disabled - local: http://localhost:{proxy_port}")

    import uvicorn
    log.info(f"\n[START] uvicorn 0.0.0.0:{proxy_port} (Ctrl+C to stop)\n")
    uvicorn.run(app, host="0.0.0.0", port=proxy_port,
                log_level="warning", access_log=False)

if __name__ == "__main__":
    main()
