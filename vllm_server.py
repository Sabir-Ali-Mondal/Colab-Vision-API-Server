#!/usr/bin/env python3

# set safe default before vllm import
import os
os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")

# workaround for vllm import issue
try:
    import torch
    import torch._inductor.config
except Exception:
    pass

import sys, subprocess, time, re, io, base64, json, threading, traceback, shutil, signal
from pathlib import Path

LOGFILE  = Path("/tmp/vllm_server.log")
VLLM_LOG = Path("/tmp/vllm.log")
PID_FILE = Path("/tmp/vllm_server.pid")

def log(msg: str, level: str = "INFO") -> None:
    line = f"[{time.strftime('%H:%M:%S')}] [{level}] {msg}"
    print(line, flush=True)
    try:
        with open(LOGFILE, "a") as fh:
            fh.write(line + "\n")
    except Exception:
        pass

# hardware detection
def _detect_hardware() -> dict:
    forced = os.environ.get("VLLM_DEVICE", "").lower()

    # check for gpu
    is_gpu = False
    gpu_name, cc, n_gpu = "", (0, 0), 1
    try:
        if torch.cuda.is_available():
            is_gpu   = True
            gpu_name = torch.cuda.get_device_name(0)
            cc       = torch.cuda.get_device_capability(0)
            n_gpu    = torch.cuda.device_count()
    except Exception:
        if shutil.which("nvidia-smi"):
            is_gpu = True

    # check for tpu
    is_tpu = (
        Path("/dev/accel0").exists()
        or bool(os.environ.get("COLAB_TPU_ADDR"))
        or forced == "tpu"
    )

    # make hardware decision
    if forced == "cpu" or (not is_gpu and not is_tpu):
        hw = "cpu"
    elif forced == "tpu" or is_tpu:
        hw = "tpu"
    else:
        hw = "gpu"

    log(f"Hardware detected: {hw.upper()}"
        + (f"  ({gpu_name}, cc={cc})" if hw == "gpu" else ""))

    # set vllm flags based on hardware
    if hw == "cpu":
        os.environ["VLLM_TARGET_DEVICE"] = "cpu"
        return dict(
            device_args = ["--device", "cpu",
                           "--block-size", "16",
                           "--swap-space", "4"],
            dtype        = "bfloat16",
            max_seqs     = 2,
            max_model_len= 2048,
            extra        = [],
            label        = "CPU",
            timeout      = 1200,
        )

    if hw == "gpu":
        os.environ["VLLM_TARGET_DEVICE"] = "cuda"
        dtype = "float16" if cc < (8, 0) else "bfloat16"
        extra = ["--gpu-memory-utilization", "0.90",
                 "--enable-prefix-caching"]
        if n_gpu > 1:
            extra += ["--tensor-parallel-size", str(n_gpu)]
        return dict(
            device_args = ["--block-size", "16", "--swap-space", "4"],
            dtype        = dtype,
            max_seqs     = 16,
            max_model_len= 4096,
            extra        = extra,
            label        = f"GPU ({gpu_name or 'unknown'})",
            timeout      = 1200,
        )

    os.environ["VLLM_TARGET_DEVICE"] = "tpu"
    n_chips = len(list(Path("/dev").glob("accel*"))) or 1
    return dict(
        device_args = [],
        dtype        = "bfloat16",
        max_seqs     = 8,
        max_model_len= 4096,
        extra        = ["--tensor-parallel-size", str(n_chips)],
        label        = f"TPU v5e ({n_chips} chip{'s' if n_chips>1 else ''})",
        timeout      = 3600,
    )

HW = _detect_hardware()

# install dependencies
_DEPS = [
    ("vllm",            "vllm>=0.6.0,<0.12.0"),
    ("transformers",    "transformers>=4.45.0"),
    ("fastapi",         "fastapi"),
    ("uvicorn",         "uvicorn[standard]"),
    ("httpx",           "httpx"),
    ("PIL",             "Pillow"),
    ("huggingface_hub", "huggingface_hub"),
    ("accelerate",      "accelerate"),
    ("qwen_vl_utils",   "qwen-vl-utils>=0.0.14"),
]

def install_dependencies() -> None:
    log("Installing dependencies")
    missing = [spec for imp, spec in _DEPS if not _try_import(imp)]
    if missing:
        log(f"pip install: {missing}")
        subprocess.run([sys.executable, "-m", "pip", "install",
                        "-q", "--upgrade", *missing], check=False)
    else:
        log("All dependencies satisfied.")

def _try_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False

install_dependencies()

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, ImageDraw

# configuration
VISION_MODELS = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2-VL-2B-Instruct",
]
VISION_PORT   = 8001
PROXY_PORT    = 8000
MODEL_CACHE_DIR: Path = Path(os.environ.get("HF_HOME", "/tmp/hf_cache"))

_proc:        subprocess.Popen | None = None
_tunnel_proc: subprocess.Popen | None = None
TUNNEL_URL:   str  = ""
_active_model: str = VISION_MODELS[0]

# google drive mount
def try_mount_gdrive() -> None:
    global MODEL_CACHE_DIR
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        cache = Path("/content/drive/MyDrive/vllm_server/models")
        cache.mkdir(parents=True, exist_ok=True)
        MODEL_CACHE_DIR = cache
        os.environ["HF_HOME"] = str(cache)
        log(f"Google Drive mounted cache: {cache}")
    except ImportError:
        log("Not in Colab using /tmp/hf_cache")
    except Exception as exc:
        log(f"Drive mount skipped: {exc}", "WARN")

# vllm process management
def _pick_model() -> str:
    from huggingface_hub import HfApi
    api = HfApi()
    for model_id in VISION_MODELS:
        local = MODEL_CACHE_DIR / "--".join(["models"] + model_id.split("/"))
        if local.exists():
            log(f"Cached model: {local}")
            return model_id
        try:
            api.model_info(model_id, timeout=10)
            log(f"Hub model ok: {model_id}")
            return model_id
        except Exception:
            log(f"Not accessible: {model_id}", "WARN")
    raise RuntimeError(f"No model available: {VISION_MODELS}")

def start_vision_server() -> subprocess.Popen:
    global _proc, _active_model
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _active_model = _pick_model()

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model",               _active_model,
        "--port",                str(VISION_PORT),
        "--host",                "0.0.0.0",
        "--dtype",               HW["dtype"],
        "--max-model-len",       str(HW["max_model_len"]),
        "--max-num-seqs",        str(HW["max_seqs"]),
        "--limit-mm-per-prompt", '{"image": 1}',
        "--disable-log-stats",
    ]
    cmd += HW["device_args"]
    cmd += HW["extra"]

    env = {**os.environ, "VLLM_TARGET_DEVICE": os.environ["VLLM_TARGET_DEVICE"]}

    log(f"Starting vLLM [{HW['label']}] model={_active_model}")
    log(f"Tail logs with: !tail -f {VLLM_LOG}")

    _proc = subprocess.Popen(
        cmd,
        stdout=open(VLLM_LOG, "w"),
        stderr=subprocess.STDOUT,
        env=env,
    )
    log(f"vLLM pid={_proc.pid}")
    return _proc

def wait_for_ready() -> bool:
    timeout  = HW["timeout"]
    deadline = time.time() + timeout
    log(f"Waiting for vLLM [{HW['label']}] (up to {timeout//60} min)")
    last_sz  = 0

    while time.time() < deadline:
        if _proc and _proc.poll() is not None:
            log("vLLM died. Last log:", "ERROR")
            try:
                print(VLLM_LOG.read_text(errors="replace")[-4000:], flush=True)
            except Exception:
                pass
            return False
        try:
            if httpx.get(f"http://127.0.0.1:{VISION_PORT}/health",
                         timeout=3).status_code == 200:
                log(f"vLLM ready [{HW['label']}]")
                return True
        except Exception:
            pass

        try:
            txt = VLLM_LOG.read_text(errors="replace")
            if len(txt) > last_sz + 300:
                last = [l for l in txt.splitlines() if l.strip()]
                if last:
                    log(f"  vllm> {last[-1][-120:]}")
                last_sz = len(txt)
        except Exception:
            pass
        time.sleep(5)

    log("vLLM startup timed out.", "ERROR")
    return False

# image loading helpers
def _load_image(src: str) -> tuple[str, str]:
    if src.startswith("data:"):
        m = re.match(r"data:(image/[^;]+);base64,(.+)", src, re.DOTALL)
        if m:
            return m.group(2), m.group(1)
        raise ValueError("Malformed data URI")
    if src.startswith("file://"):
        p   = Path(src[7:])
        ext = p.suffix.lower().lstrip(".")
        return base64.b64encode(p.read_bytes()).decode(), \
               f"image/{'jpeg' if ext == 'jpg' else ext}"
    if src.startswith(("http://", "https://")):
        r  = httpx.get(src, timeout=20, follow_redirects=True)
        r.raise_for_status()
        ct = r.headers.get("content-type", "image/png").split(";")[0].strip()
        return base64.b64encode(r.content).decode(), ct

    peek = base64.b64decode(src[:128] + "==")
    mime = ("image/png"  if peek[:4]   == b"\x89PNG" else
            "image/jpeg" if peek[:2]   == b"\xff\xd8" else
            "image/webp" if peek[8:12] == b"WEBP"    else "image/png")
    return src, mime

# fastapi routing
app  = FastAPI(title="UI-Agent Server")
BASE = f"http://127.0.0.1:{VISION_PORT}"

@app.get("/health")
async def health():
    return {"status": "ok", "hardware": HW["label"], "model": _active_model}

@app.post("/v1/agent")
async def agent_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    image_src  = body.get("image", "")
    prompt     = body.get("prompt", "").strip()
    max_tokens = int(body.get("max_tokens", 512))
    system_msg = body.get("system",
        "You are a UI automation assistant. "
        "Look at the screenshot and answer in plain text. Be concise.")

    if not prompt:
        return JSONResponse({"error": "'prompt' is required"}, status_code=400)

    content: list = []
    if image_src:
        try:
            b64, mime = _load_image(image_src)
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"}})
            log(f"Agent | {mime} | {prompt[:60]!r}")
        except Exception as exc:
            log(f"Image load failed: {exc} text-only", "WARN")
    else:
        log(f"Agent | no image | {prompt[:60]!r}")

    content.append({"type": "text", "text": prompt})

    payload = {
        "model":       _active_model,
        "messages":    [{"role": "system", "content": system_msg},
                        {"role": "user",   "content": content}],
        "max_tokens":  max_tokens,
        "temperature": 0.2,
    }
    try:
        async with httpx.AsyncClient(timeout=180) as c:
            resp = await c.post(f"{BASE}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return JSONResponse({"text": text})
    except Exception as exc:
        log(f"Backend error: {exc}", "ERROR")
        return JSONResponse({"error": str(exc)}, status_code=502)

@app.api_route("/v1/{path:path}", methods=["GET","POST","PUT","DELETE"])
async def passthrough(request: Request, path: str):
    body    = await request.body()
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length")}
    stream  = False
    if body:
        try:
            stream = json.loads(body).get("stream", False)
        except Exception:
            pass
    if stream:
        async def _s():
            async with httpx.AsyncClient(timeout=300) as c:
                async with c.stream(request.method, f"{BASE}/v1/{path}",
                                    content=body, headers=headers) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk
        return StreamingResponse(_s(), media_type="text/event-stream")
    async with httpx.AsyncClient(timeout=300) as c:
        r = await c.request(request.method, f"{BASE}/v1/{path}",
                            content=body, headers=headers)
    return Response(content=r.content, status_code=r.status_code,
                    headers=dict(r.headers))

def start_proxy_thread() -> None:
    log(f"Starting in-memory proxy on port {PROXY_PORT}")
    threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT,
                                   log_level="warning"),
        daemon=True, name="proxy",
    ).start()

# cloudflare tunnel setup
def start_cloudflare_tunnel(port: int) -> str:
    global TUNNEL_URL, _tunnel_proc
    if not shutil.which("cloudflared"):
        arch = subprocess.run("uname -m", shell=True,
                              capture_output=True, text=True).stdout.strip()
        sfx  = "arm64" if arch == "aarch64" else "amd64"
        log(f"Downloading cloudflared ({sfx})")
        subprocess.run(
            f"curl -fsSL https://github.com/cloudflare/cloudflared/releases/"
            f"latest/download/cloudflared-linux-{sfx} "
            f"-o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared",
            shell=True, check=False)

    _tunnel_proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def _drain():
        for _ in _tunnel_proc.stdout:
            pass

    for line in _tunnel_proc.stdout:
        m = re.search(r"https://[a-z0-9\-]+\.trycloudflare\.com", line)
        if m:
            TUNNEL_URL = m.group(0)
            log(f"PUBLIC  : {TUNNEL_URL}/v1/agent")
            log(f"LOCAL   : http://127.0.0.1:{port}/v1/agent")
            threading.Thread(target=_drain, daemon=True).start()
            return TUNNEL_URL

    log("Tunnel URL not detected.", "WARN")
    return ""

# smoke test
def run_smoke_test() -> None:
    log("Smoke test running")
    img  = Image.new("RGB", (280, 120), (236, 233, 216))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 280, 22], fill=(0, 78, 152))
    draw.text((6, 4),   "System Dialog", fill="white")
    draw.text((20, 45), "Save changes?", fill="black")
    for x, label in [(40, "Save"), (120, "Discard"), (200, "Cancel")]:
        draw.rectangle([x, 85, x+60, 108], fill=(212, 208, 200))
        draw.text((x+8, 92), label, fill="black")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    try:
        r = httpx.post(f"http://127.0.0.1:{PROXY_PORT}/v1/agent", json={
            "image": b64,
            "prompt": "List the button labels visible in this dialog.",
            "max_tokens": 64,
        }, timeout=120)
        result = r.json()
        if "text" in result:
            log(f"Smoke test pass result: {result['text']!r}")
        else:
            log(f"Smoke test fail result: {result}", "WARN")
    except Exception as exc:
        log(f"Smoke test error: {exc}", "ERROR")

# watchdog loop
def watchdog_loop() -> None:
    log("Watchdog started.")
    while True:
        time.sleep(30)
        if _proc and _proc.poll() is not None:
            log("vLLM crashed restarting", "WARN")
            start_vision_server()
        if _tunnel_proc and _tunnel_proc.poll() is not None:
            log("Tunnel crashed restarting", "WARN")
            start_cloudflare_tunnel(PROXY_PORT)

# shutdown handling
def shutdown(signum=None, frame=None) -> None:
    log("Shutting down")
    for p in [_proc, _tunnel_proc]:
        if p:
            try:
                p.terminate()
            except Exception:
                pass
    sys.exit(0)

signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)

# main execution
def main() -> None:
    PID_FILE.write_text(str(os.getpid()))

    # port cleanup
    subprocess.run(["fuser", "-k", f"{PROXY_PORT}/tcp"], capture_output=True)
    subprocess.run(["fuser", "-k", f"{VISION_PORT}/tcp"], capture_output=True)
    time.sleep(1)

    log("vllm_server.py starting")
    log(f"Hardware : {HW['label']}")
    log(f"dtype : {HW['dtype']} max_len: {HW['max_model_len']} max_seqs: {HW['max_seqs']}")

    install_dependencies()
    try_mount_gdrive()

    start_proxy_thread()
    time.sleep(1)

    start_vision_server()
    if not wait_for_ready():
        log("vLLM failed to start. Check /tmp/vllm.log", "ERROR")
        shutdown()

    tunnel_url = start_cloudflare_tunnel(PROXY_PORT)
    public_url = tunnel_url or f"http://127.0.0.1:{PROXY_PORT}"

    log(f"ENDPOINT : {public_url}/v1/agent")
    log(f"MODEL : {_active_model}")
    log(f"HARDWARE : {HW['label']}")
    
    time.sleep(2)
    run_smoke_test()

    threading.Thread(target=watchdog_loop, daemon=True, name="watchdog").start()

    log("Server running. Ctrl+C to stop.")
    while True:
        time.sleep(60)
        alive = "alive" if (_proc and _proc.poll() is None) else "DEAD"
        log(f"Heartbeat | vLLM={alive} | hw={HW['label']} | tunnel={'up' if TUNNEL_URL else 'none'}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc(), "FATAL")
        shutdown()
