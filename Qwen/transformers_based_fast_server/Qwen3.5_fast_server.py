# ============================================================
# Qwen3.5 Gateway || Server 
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
    "gpu_memory_utilization": 0.82,
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
import json
import uuid
import inspect
import threading
import subprocess
import urllib.request
import httpx  # <--- CRITICAL IMPORT FIX
from collections import defaultdict, deque
from pathlib import Path

PROXY_PORT = 8090

print("=" * 55)
print(" Qwen3.5 Gateway starting (Signature-Filter Edition) ...")
print("=" * 55 + "\n")

# ── 0. Cleanup ────────────────────────────────────────────────
print("[0/4] Freeing GPU VRAM...")
subprocess.run("pkill -9 -f 'cloudflared'", shell=True, stderr=subprocess.DEVNULL)
subprocess.run("fuser -k -9 8090/tcp", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(2)
gc.collect()
if "torch" in sys.modules:
    import torch
    torch.cuda.empty_cache()
print("  ✓ GPU memory flushed.")

# ── Keepalive ─────────────────────────────────────────────────
_ALIVE = True
def _keepalive():
    while _ALIVE:
        try: sys.stderr.write("\x1b[0m"); sys.stderr.flush()
        except Exception: pass
        time.sleep(20)
threading.Thread(target=_keepalive, daemon=True).start()

# ── 1. Cache ──────────────────────────────────────────────────
print("\n[1/4] Setting up model cache ...")
def setup_cache() -> Path:
    drive_cache = Path(COLAB_CONFIG["drive_cache_dir"])
    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists():
        drive_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(drive_cache)
        print(f"  Drive mounted: {drive_cache}")
        return drive_cache
    os.environ["HF_HOME"] = "/tmp/qwen35_cache"
    return Path("/tmp/qwen35_cache")

cache_root = setup_cache()
model_id = {"2b": "Qwen/Qwen3.5-2B", "4b": "Qwen/Qwen3.5-4B"}[COLAB_CONFIG["model"]]

def resolve_model(mid):
    slug = mid.replace("/", "--")
    snap_root = cache_root / "hub" / f"models--{slug}" / "snapshots"
    if snap_root.exists():
        snaps = sorted([p for p in snap_root.glob("*/") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
        if snaps: return str(snaps[0]), True
    return mid, False

model_path, from_cache = resolve_model(model_id)

# ── 2. Deps ───────────────────────────────────────────────────
print("\n[2/4] Checking dependencies (Transformers GitHub Main)...")
def pip_install(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", *pkgs])

_PACKAGES = [
    ("fastapi", "fastapi>=0.110"), ("uvicorn", "uvicorn[standard]"),
    ("httpx", "httpx"), ("PIL", "Pillow"), ("accelerate", "accelerate"),
    ("transformers", "git+https://github.com/huggingface/transformers.git@main"),
    ("qwen_vl_utils", "qwen-vl-utils"), ("torchvision", "torchvision"),
    ("pypdf", "pypdf"), ("pdfplumber", "pdfplumber"),
]

for imp, pkg in _PACKAGES:
    print(f"  Checking {pkg}...", end=" ", flush=True)
    try:
        if imp == "transformers": raise ImportError # Force update
        __import__(imp)
        print("OK")
    except ImportError:
        pip_install(pkg)
        print("INSTALLED")

# ── 3. Load Model ─────────────────────────────────────────────
print(f"\n[3/4] Loading {model_id} ...")
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, TextIteratorStreamer

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
).eval()

FORWARD_PARAMS = set(inspect.signature(model.forward).parameters.keys())
print(f"  ✓ Model ready. Accepted keys: {list(FORWARD_PARAMS)[:10]}...")

# ── 4. API ────────────────────────────────────────────────────
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse

logging.basicConfig(level=logging.INFO)
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
log = logging.getLogger("qwen")
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def to_content_block(source: str) -> dict:
    if source.startswith("data:"):
        return {"type": "image_url", "image_url": {"url": source}}
    elif source.startswith(("http://","https://")):
        ext = source.split("?")[0].rsplit(".",1)[-1].lower()
        if ext in ("mp4","mov","avi","webm"): return {"type":"video_url","video_url":{"url":source}}
        if ext == "pdf": 
            try:
                with urllib.request.urlopen(source) as r:
                    import pdfplumber
                    with pdfplumber.open(io.BytesIO(r.read())) as pdf:
                        return {"type":"text","text":"\n".join(p.extract_text() or "" for p in pdf.pages)}
            except: return {"type":"text","text":"[pdf extraction failed]"}
        return {"type":"image_url","image_url":{"url":source}}
    return {"type":"text","text":f"[file: {source}]"}

@app.post("/v1/agent")
async def agent_endpoint(request: Request):
    try:
        body = await request.json()
    except:
        raise HTTPException(400, "Invalid JSON")
        
    content = [to_content_block(str(s)) for s in body.get("files", [])]
    if body.get("prompt"): content.append({"type": "text", "text": body["prompt"]})
    
    payload = {
        "model": model_id,
        "messages": [{"role": "system", "content": body.get("system", "You are helpful AI.")},
                     {"role": "user", "content": content}],
        "stream": body.get("stream", False),
        "temperature": body.get("temperature", 0.7),
        "max_tokens": body.get("max_tokens", 1024),
        "chat_template_kwargs": {"enable_thinking": body.get("thinking", True)}
    }
    
    async with httpx.AsyncClient(timeout=600) as cli:
        r = await cli.post(f"http://127.0.0.1:{PROXY_PORT}/v1/chat/completions", json=payload)
        if body.get("stream"):
            return StreamingResponse(r.aiter_bytes(), media_type="text/event-stream")
        return JSONResponse(r.json())

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    msgs = body.get("messages", [])
    
    new_msgs = []
    for m in msgs:
        if isinstance(m.get("content"), list):
            new_c = []
            for b in m["content"]:
                if b.get("type") == "image_url": new_c.append({"type": "image", "image": b["image_url"]["url"]})
                elif b.get("type") == "video_url": new_c.append({"type": "video", "video": b["video_url"]["url"]})
                else: new_c.append(b)
            new_msgs.append({"role": m["role"], "content": new_c})
        else: new_msgs.append(m)

    try:
        from qwen_vl_utils import process_vision_info
        text = processor.apply_chat_template(new_msgs, tokenize=False, add_generation_prompt=True)
        img, vid = process_vision_info(new_msgs)
        inputs = processor(text=[text], images=img, videos=vid, padding=True, return_tensors="pt").to(model.device)
    except:
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to(model.device)

    # Filtering inputs for the specific model version's accepted keys
    safe_inputs = {k: v for k, v in inputs.items() if k in FORWARD_PARAMS}
    
    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = {
        **safe_inputs,
        "max_new_tokens": body.get("max_tokens", 1024),
        "temperature": max(body.get("temperature", 0.7), 0.01),
        "do_sample": body.get("temperature", 0.7) > 0,
        "streamer": streamer,
        "pad_token_id": processor.tokenizer.eos_token_id
    }

    def _gen():
        try: model.generate(**gen_kwargs)
        except Exception as e: 
            streamer.text_queue.put(f"\n[SERVER ERROR] {e}")
            streamer.text_queue.put(None)

    threading.Thread(target=_gen).start()
    
    if body.get("stream"):
        async def event_gen():
            for t in streamer:
                yield f"data: {json.dumps({'choices':[{'delta':{'content':t}}]})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(event_gen(), media_type="text/event-stream")
    
    full = "".join(list(streamer))
    if not body.get("chat_template_kwargs", {}).get("enable_thinking", True):
        full = THINK_RE.sub("", full).strip()
    return JSONResponse({"choices": [{"message": {"role": "assistant", "content": full}}]})

# ── Final Startup ─────────────────────────────────────────────
import uvicorn
threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT, access_log=False), daemon=True).start()
time.sleep(5)

cf_bin = Path("/usr/local/bin/cloudflared")
if not cf_bin.exists():
    subprocess.run(["wget", "-q", "-O", str(cf_bin), "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"])
    cf_bin.chmod(0o755)

print("\n[Tunnel] Starting...")
proc = subprocess.Popen(["cloudflared", "tunnel", "--url", f"http://localhost:{PROXY_PORT}"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    m = re.search(r"https://[a-z0-9\-]+\.trycloudflare\.com", line)
    if m:
        print("\n" + "="*60 + f"\n  URL: {m.group(0)}\n" + "="*60 + "\n")
        break

try:
    while True: time.sleep(60)
except KeyboardInterrupt: print("Stop.")
