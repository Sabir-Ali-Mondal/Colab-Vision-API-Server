#!/usr/bin/env python3
"""
Full-featured client for the Qwen3.5 vLLM server.
Covers every capability the server exposes.
"""

import httpx
import base64
import json
import os
import sys
from pathlib import Path

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────
def get_api_url() -> str:
    url = input("\nEnter the server base URL (from Colab logs): ").strip()
    if not url:
        print("URL cannot be empty.")
        sys.exit(1)
    return url.rstrip("/")

def print_json(label: str, data: dict):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(json.dumps(data, indent=2, ensure_ascii=False))

def extract_reply(response: dict) -> str:
    """Pull assistant text out of a standard OpenAI-style response."""
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return response.get("error", str(response))

def post(base: str, path: str, **kwargs) -> dict:
    r = httpx.post(f"{base}{path}", timeout=180, **kwargs)
    r.raise_for_status()
    return r.json()

def get(base: str, path: str) -> dict:
    r = httpx.get(f"{base}{path}", timeout=30)
    r.raise_for_status()
    return r.json()

def delete(base: str, path: str) -> dict:
    r = httpx.delete(f"{base}{path}", timeout=30)
    r.raise_for_status()
    return r.json()

# ──────────────────────────────────────────────
# 1. Health check
# ──────────────────────────────────────────────
def demo_health(base: str):
    print("\n[1] HEALTH CHECK — GET /health")
    data = get(base, "/health")
    print_json("Server Info", data)

# ──────────────────────────────────────────────
# 2. Metrics
# ──────────────────────────────────────────────
def demo_metrics(base: str):
    print("\n[2] METRICS — GET /metrics")
    data = get(base, "/metrics")
    print_json("Metrics", data)

# ──────────────────────────────────────────────
# 3. Plain text chat  (thinking mode ON)
# ──────────────────────────────────────────────
def demo_text_chat(base: str):
    print("\n[3] PLAIN TEXT CHAT — thinking mode ON")

    payload = {
        "prompt": "What is the capital of France? Be brief.",
        "max_tokens": 128,
        "temperature": 0.6,
        "top_p": 0.9,
        "enable_thinking": True,       # model reasons before answering
        "stream": False
    }

    print("  Request JSON:")
    print(json.dumps(payload, indent=4))

    data = post(base, "/chat", json=payload)
    print_json("Response", data)
    print("\n  Reply:", extract_reply(data))

# ──────────────────────────────────────────────
# 4. Fast chat  (thinking mode OFF)
# ──────────────────────────────────────────────
def demo_fast_chat(base: str):
    print("\n[4] FAST CHAT — thinking mode OFF")

    payload = {
        "prompt": "List 3 planets in our solar system.",
        "max_tokens": 64,
        "temperature": 0.3,
        "top_p": 0.9,
        "enable_thinking": False,      # faster, no internal reasoning
        "stream": False
    }

    print("  Request JSON:")
    print(json.dumps(payload, indent=4))

    data = post(base, "/chat", json=payload)
    print("\n  Reply:", extract_reply(data))

# ──────────────────────────────────────────────
# 5. Streaming chat
# ──────────────────────────────────────────────
def demo_streaming(base: str):
    print("\n[5] STREAMING CHAT — words appear as they generate")

    payload = {
        "prompt": "Write a two-sentence poem about the ocean.",
        "max_tokens": 100,
        "temperature": 0.8,
        "enable_thinking": False,
        "stream": True                 # server sends SSE chunks
    }

    print("  Request JSON:")
    print(json.dumps(payload, indent=4))
    print("\n  Streamed reply: ", end="", flush=True)

    with httpx.stream("POST", f"{base}/chat", json=payload, timeout=180) as r:
        for line in r.iter_lines():
            if line.startswith("data:") and "[DONE]" not in line:
                chunk = json.loads(line[5:].strip())
                delta = chunk["choices"][0]["delta"].get("content", "")
                print(delta, end="", flush=True)
    print()

# ──────────────────────────────────────────────
# 6. Upload an image file
# ──────────────────────────────────────────────
def demo_upload_image(base: str) -> str:
    print("\n[6] UPLOAD IMAGE — fetch a dog photo and upload it")

    # Fetch a random dog image from public API
    print("  Fetching dog image from dog.ceo …")
    r = httpx.get("https://dog.ceo/api/breeds/image/random", timeout=30)
    image_url = r.json()["message"]
    print(f"  Image URL: {image_url}")

    img_bytes = httpx.get(image_url, timeout=30).content

    # Save temporarily so we can upload as a real file
    tmp = Path("/tmp/demo_dog.jpg")
    tmp.write_bytes(img_bytes)

    with open(tmp, "rb") as f:
        resp = httpx.post(f"{base}/upload",
                          files={"file": ("demo_dog.jpg", f, "image/jpeg")},
                          timeout=60)
    resp.raise_for_status()
    meta = resp.json()

    print_json("Upload Response", meta)
    return meta["file_id"]

# ──────────────────────────────────────────────
# 7. Chat with uploaded image
# ──────────────────────────────────────────────
def demo_chat_with_image(base: str, file_id: str):
    print("\n[7] CHAT WITH IMAGE — use uploaded image file_id")

    payload = {
        "prompt": "Describe the dog in this image. What breed might it be?",
        "file_ids": [file_id],         # reference the uploaded image by id
        "max_tokens": 256,
        "temperature": 0.6,
        "enable_thinking": True,
        "stream": False
    }

    print("  Request JSON:")
    print(json.dumps(payload, indent=4))

    data = post(base, "/chat", json=payload)
    print("\n  Reply:", extract_reply(data))

# ──────────────────────────────────────────────
# 8. Upload a text file
# ──────────────────────────────────────────────
def demo_upload_text(base: str) -> str:
    print("\n[8] UPLOAD TEXT FILE — create and upload a .txt document")

    # Create a sample text file on the fly
    content = (
        "Project: Mars Mission Alpha\n"
        "Budget: $4.2 billion\n"
        "Launch date: 2031\n"
        "Crew size: 6 astronauts\n"
        "Mission duration: 900 days\n"
        "Primary objective: Collect rock samples and test life-support systems.\n"
        "Risk level: High\n"
        "Status: Pre-launch planning phase\n"
    )
    tmp = Path("/tmp/demo_report.txt")
    tmp.write_text(content)

    with open(tmp, "rb") as f:
        resp = httpx.post(f"{base}/upload",
                          files={"file": ("mars_report.txt", f, "text/plain")},
                          timeout=60)
    resp.raise_for_status()
    meta = resp.json()

    print_json("Upload Response", meta)
    return meta["file_id"]

# ──────────────────────────────────────────────
# 9. Chat with text file (document Q&A)
# ──────────────────────────────────────────────
def demo_chat_with_text(base: str, file_id: str):
    print("\n[9] CHAT WITH TEXT FILE — ask questions about the document")

    payload = {
        "prompt": "What is the budget and how long is the mission?",
        "file_ids": [file_id],         # server prepends file content as context
        "max_tokens": 128,
        "temperature": 0.3,
        "enable_thinking": False,
        "stream": False
    }

    print("  Request JSON:")
    print(json.dumps(payload, indent=4))

    data = post(base, "/chat", json=payload)
    print("\n  Reply:", extract_reply(data))

# ──────────────────────────────────────────────
# 10. Upload + chat in one shot  (multipart)
# ──────────────────────────────────────────────
def demo_chat_upload_oneshot(base: str):
    print("\n[10] CHAT + UPLOAD ONE SHOT — POST /chat/upload (multipart)")

    # Create a quick JSON file to upload
    data_file = Path("/tmp/demo_data.json")
    data_file.write_text(json.dumps({
        "product": "Wireless Headphones X200",
        "price": 89.99,
        "rating": 4.7,
        "reviews": 1523,
        "features": ["Noise cancellation", "30hr battery", "Bluetooth 5.3"]
    }, indent=2))

    # Everything in one multipart request — no separate upload step
    with open(data_file, "rb") as f:
        resp = httpx.post(
            f"{base}/chat/upload",
            data={
                "prompt":          "Summarize this product in one sentence.",
                "max_tokens":      "100",
                "temperature":     "0.5",
                "enable_thinking": "false",
                "stream":          "false",
            },
            files={"file": ("product.json", f, "application/json")},
            timeout=120,
        )
    resp.raise_for_status()
    data = resp.json()

    print("  Multipart fields sent: prompt + file (product.json)")
    print("\n  Reply:", extract_reply(data))

# ──────────────────────────────────────────────
# 11. Multiple file_ids in one chat
# ──────────────────────────────────────────────
def demo_multi_file_chat(base: str, image_id: str, text_id: str):
    print("\n[11] CHAT WITH MULTIPLE FILES — image + text doc together")

    payload = {
        "prompt": (
            "I have attached a document and an image. "
            "From the document, what is the mission objective? "
            "From the image, describe what you see."
        ),
        "file_ids": [image_id, text_id],   # both files referenced together
        "max_tokens": 300,
        "temperature": 0.6,
        "enable_thinking": True,
        "stream": False
    }

    print("  Request JSON:")
    print(json.dumps(payload, indent=4))

    data = post(base, "/chat", json=payload)
    print("\n  Reply:", extract_reply(data))

# ──────────────────────────────────────────────
# 12. List all files on server
# ──────────────────────────────────────────────
def demo_list_files(base: str):
    print("\n[12] LIST FILES — GET /files")
    data = get(base, "/files")
    print_json("All Files on Server", data)

# ──────────────────────────────────────────────
# 13. Download a file back
# ──────────────────────────────────────────────
def demo_download_file(base: str, file_id: str):
    print(f"\n[13] DOWNLOAD FILE — GET /files/{file_id}")

    r = httpx.get(f"{base}/files/{file_id}", timeout=30)
    r.raise_for_status()

    save_path = Path(f"/tmp/downloaded_{file_id[:8]}.bin")
    save_path.write_bytes(r.content)
    print(f"  Downloaded {len(r.content)} bytes → {save_path}")

# ──────────────────────────────────────────────
# 14. Creativity control demo
# ──────────────────────────────────────────────
def demo_creativity(base: str):
    print("\n[14] CREATIVITY CONTROL — same prompt, different temperature")

    prompt = "Describe the color blue in one sentence."

    for label, temp in [("Factual (temp=0.1)", 0.1),
                         ("Creative (temp=1.2)", 1.2)]:
        payload = {
            "prompt":          prompt,
            "max_tokens":      60,
            "temperature":     temp,
            "enable_thinking": False,
            "stream":          False
        }
        data = post(base, "/chat", json=payload)
        print(f"\n  [{label}]")
        print(f"  Request JSON: {json.dumps(payload)}")
        print(f"  Reply: {extract_reply(data)}")

# ──────────────────────────────────────────────
# 15. OpenAI SDK compatible  /v1/chat/completions
# ──────────────────────────────────────────────
def demo_openai_compat(base: str):
    print("\n[15] OpenAI-COMPATIBLE ENDPOINT — POST /v1/chat/completions")
    print("  (Any OpenAI SDK client works by setting base_url to this server)")

    # Raw OpenAI-style payload — exactly what the openai Python library sends
    payload = {
        "model":    "Qwen/Qwen3.5-2B",
        "messages": [
            {"role": "system",  "content": "You are a helpful assistant."},
            {"role": "user",    "content": "What is 7 multiplied by 8?"}
        ],
        "max_tokens":  64,
        "temperature": 0.3,
        "stream":      False
    }

    print("  Request JSON:")
    print(json.dumps(payload, indent=4))

    data = post(base, "/v1/chat/completions", json=payload)
    print("\n  Reply:", extract_reply(data))

# ──────────────────────────────────────────────
# 16. Delete a file
# ──────────────────────────────────────────────
def demo_delete_file(base: str, file_id: str):
    print(f"\n[16] DELETE FILE — DELETE /files/{file_id}")
    data = delete(base, f"/files/{file_id}")
    print_json("Delete Response", data)

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  Qwen3.5 Local Client — Full Capability Demo")
    print("="*55)

    BASE = get_api_url()

    try:
        # ── server status ──
        demo_health(BASE)
        demo_metrics(BASE)

        # ── text only ──
        demo_text_chat(BASE)
        demo_fast_chat(BASE)
        demo_streaming(BASE)

        # ── file operations ──
        image_id = demo_upload_image(BASE)
        text_id  = demo_upload_text(BASE)
        demo_list_files(BASE)

        # ── chat with files ──
        demo_chat_with_image(BASE, image_id)
        demo_chat_with_text(BASE, text_id)
        demo_multi_file_chat(BASE, image_id, text_id)

        # ── one-shot upload+chat ──
        demo_chat_upload_oneshot(BASE)

        # ── generation controls ──
        demo_creativity(BASE)

        # ── openai compat ──
        demo_openai_compat(BASE)

        # ── download then cleanup ──
        demo_download_file(BASE, image_id)
        demo_delete_file(BASE, image_id)
        demo_delete_file(BASE, text_id)

        # ── final metrics ──
        demo_metrics(BASE)

        print("\n" + "="*55)
        print("  All demos completed successfully.")
        print("="*55)

    except httpx.HTTPStatusError as e:
        print(f"\n[HTTP ERROR] {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        print(f"\n[CONNECTION ERROR] {e}")
    except KeyboardInterrupt:
        print("\nAborted.")

if __name__ == "__main__":
    main()
