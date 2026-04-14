import os, sys, subprocess, time, re, threading, shutil, mimetypes, uuid
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
#  1. CONFIGURE YOUR FOLDER & DEPENDENCIES
# ══════════════════════════════════════════════════════════════════════
DRIVE_ROOT = Path("/content/drive/MyDrive/Qwen35_Server")
LOGFILE    = DRIVE_ROOT / "server.log"
UPLOAD_DIR = DRIVE_ROOT / "uploads"
MODEL_DIR  = DRIVE_ROOT / "model_cache"

# Packages required
REQUIRED_PKGS = [
    "huggingface_hub>=0.23.0",
    "transformers>=4.51.0",
    "accelerate",
    "torch==2.4.0",
    "torchvision==0.19.0",
    "pillow>=10.4.0",
    "fastapi",
    "uvicorn[standard]",
    "httpx",
    "python-multipart",
    "aiofiles",
    "pdfplumber",
    "sentencepiece",
    "qwen-vl-utils",
]

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOGFILE, "a") as f:
            f.write(line + "\n")
    except Exception: pass

# ══════════════════════════════════════════════════════════════════════
#  2. INSTALLATION & KERNEL RESTART LOGIC
# ══════════════════════════════════════════════════════════════════════
def check_and_install():
    # We check for a specific version of torchvision to see if we've already updated.
    # Colab's default is usually older than 0.19.0.
    already_installed = False
    try:
        import torchvision
        from importlib.metadata import version
        if version("torchvision") == "0.19.0":
            already_installed = True
    except Exception:
        pass

    if not already_installed:
        log("First run: Installing packages and updating PyTorch...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--upgrade", *REQUIRED_PKGS],
        )
        log("Install complete. Restarting runtime to load new binaries...")
        # Kill the process to force a clean kernel restart
        os.kill(os.getpid(), 9)
    else:
        log("Environment is ready ✓")

# Execute install/restart check
check_and_install()

# ══════════════════════════════════════════════════════════════════════
#  3. SERVER SETUP (Continues after restart)
# ══════════════════════════════════════════════════════════════════════
def setup_drive():
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        log("Drive mounted.")
    except Exception:
        log("Drive not available — using /tmp")
        global DRIVE_ROOT, LOGFILE, UPLOAD_DIR, MODEL_DIR
        DRIVE_ROOT = Path("/tmp/Qwen35_Server")
        LOGFILE    = DRIVE_ROOT / "server.log"
        UPLOAD_DIR = DRIVE_ROOT / "uploads"
        MODEL_DIR  = DRIVE_ROOT / "model_cache"

    for d in [DRIVE_ROOT, UPLOAD_DIR, MODEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(MODEL_DIR)
    log(f"Work directory: {DRIVE_ROOT}")

setup_drive()

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

def detect_hw():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        log(f"GPU: {name} ({vram}GB)")
        return "cuda", vram
    log("Running on CPU (Slow mode)")
    return "cpu", 0

DEVICE, VRAM_GB = detect_hw()

# ══════════════════════════════════════════════════════════════════════
#  4. LOAD MODEL
# ══════════════════════════════════════════════════════════════════════
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-2B").strip()
DTYPE    = torch.bfloat16 if DEVICE == "cuda" else torch.float32

log(f"Loading {MODEL_ID}...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
log("Model loaded successfully ✓")

_infer_lock = threading.Lock()

def run_inference(messages, max_new_tokens, temperature, top_p, enable_thinking):
    with _infer_lock:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )
        inputs = processor(text=[text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p,
                do_sample=temperature > 0,
            )
        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        return processor.decode(new_tokens, skip_special_tokens=True).strip()

# ══════════════════════════════════════════════════════════════════════
#  5. API SERVER
# ══════════════════════════════════════════════════════════════════════
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import aiofiles
import uvicorn

app = FastAPI(title="Qwen3.5 Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

file_store = {}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    fid = str(uuid.uuid4())
    dest = UPLOAD_DIR / fid
    raw = await file.read()
    async with aiofiles.open(dest, "wb") as f:
        await f.write(raw)
    
    # Simple mime check
    mime = mimetypes.guess_type(file.filename or "")[0] or "application/octet-stream"
    meta = {"file_id": fid, "filename": file.filename, "path": str(dest), "mime": mime}
    file_store[fid] = meta
    return meta

@app.post("/chat")
async def chat(req: Request):
    body = await req.json()
    prompt = body.get("prompt", "")
    file_ids = body.get("file_ids", [])
    
    # Build content (supports text + images)
    content_parts = []
    for fid in file_ids:
        if fid in file_store:
            from PIL import Image
            content_parts.append({"type": "image", "image": Image.open(file_store[fid]["path"])})
    
    content_parts.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content_parts}]
    
    loop = asyncio.get_event_loop()
    reply = await loop.run_in_executor(None, run_inference, messages, 
                                       body.get("max_tokens", 512),
                                       body.get("temperature", 0.6),
                                       body.get("top_p", 0.9),
                                       body.get("enable_thinking", False))
    return {"reply": reply}

# ══════════════════════════════════════════════════════════════════════
#  6. CLOUDFLARED TUNNEL
# ══════════════════════════════════════════════════════════════════════
def start_tunnel():
    if not shutil.which("cloudflared"):
        log("Installing cloudflared...")
        subprocess.run("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && dpkg -i cloudflared-linux-amd64.deb", shell=True)
    
    p = subprocess.Popen(["cloudflared", "tunnel", "--url", "http://127.0.0.1:8000"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        m = re.search(r"https://[a-z0-9\-]+\.trycloudflare\.com", line)
        if m:
            log(f"\nPUBLIC URL: {m.group(0)}\n")

import asyncio
threading.Thread(target=start_tunnel, daemon=True).start()

# ══════════════════════════════════════════════════════════════════════
#  7. START
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
