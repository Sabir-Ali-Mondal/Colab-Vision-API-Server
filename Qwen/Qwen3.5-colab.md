```markdown
# Qwen3.5 Server Setup & API Quick Start Guide

Welcome to your personal Qwen3.5 server. This guide covers how to start your server in Google Colab, how to get your public URL, and how to use the OpenAI-compatible API and Multimodal Agent.

---

### 1. Setup & Running the Server

**Requirements**
- Google Colab with **T4 GPU** runtime (free tier works)
- Google Drive mounted (for model caching — saves ~14 min on every restart after first run)

**Steps**
1. Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.
2. Go to **Runtime → Change runtime type → T4 GPU → Save**.
3. Paste the full `server.py` script into a cell.
4. Edit the `COLAB_CONFIG` block at the top:

```python
COLAB_CONFIG: dict = {
    "model":                  "4b",   # "2b" (~4 GB) or "4b" (~8 GB)
    "max_tokens":             8192,
    "thinking":               True,   # True = reasoning mode, False = fast
    "max_model_len":          8192,   # safe for T4 15 GB VRAM
    "api_key":                "",     # set a password to secure your server
    "rate_limit_rpm":         60,     # requests per minute per IP
    "tunnel":                 True,   # False = local only, no public URL
    "drive_cache_dir":        "/content/drive/MyDrive/qwen35_cache",
    "gpu_memory_utilization": 0.85,
}
```

5. Run the cell. Progress prints step by step.

**Expected startup times**

| Run | Time |
|---|---|
| First run (downloading model) | ~14–18 min |
| Second run onwards (cached) | ~3–4 min |

> **Why so long first time?** The 4B model is ~8 GB. vLLM downloads it, loads it into GPU VRAM, and warms up the inference engine. This is a one-time cost — Google Drive caches the model for all future runs.

6. When ready, the output shows:

```
============================================================
  PUBLIC URL           : https://xxxx-xxxx.trycloudflare.com
  /v1/chat/completions : https://xxxx-xxxx.trycloudflare.com/v1/chat/completions
  /v1/agent            : https://xxxx-xxxx.trycloudflare.com/v1/agent
  /v1/models           : https://xxxx-xxxx.trycloudflare.com/v1/models
  /health              : https://xxxx-xxxx.trycloudflare.com/health
============================================================
```

**To view raw vLLM logs:**
```python
!cat /tmp/vllm.log
```

---

### 2. Important Notes for Colab Free Tier

- **Do NOT kill port 8080** — Colab uses it internally. The server runs on port `8090` to avoid conflicts.
- **Do NOT run `fuser -k` or `kill -9`** on any port — it will disconnect your runtime.
- **GPU session lasts ~4–5 hours** on free tier. After expiry, re-run the cell — the model loads from Drive cache in ~3–4 min.
- If the runtime disconnects unexpectedly, check **Runtime → View resources** to confirm GPU is still allocated before debugging anything else.

---

### 3. Getting Your Link and Key

Scroll to the bottom of the cell output after startup. You will see the public URL banner shown above.

- **Base URL:** `https://<YOUR_URL>.trycloudflare.com/v1`
- **API Key:** whatever you set in `COLAB_CONFIG["api_key"]` (leave empty string `""` for open access)

> The Cloudflare URL changes every time you restart the server. Save the new URL after each restart.

---

### 4. Standard Chat Completions (OpenAI Compatible)

**Python** (`pip install openai`)
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<YOUR_URL>.trycloudflare.com/v1",
    api_key="your_api_key_here"  # use "empty" if no key set
)

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-4B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Write a Python script to reverse a string."}
    ],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

**Node.js** (`npm install openai`)
```javascript
import OpenAI from 'openai';

const openai = new OpenAI({
  baseURL: 'https://<YOUR_URL>.trycloudflare.com/v1',
  apiKey: 'your_api_key_here'
});

async function main() {
  const res = await openai.chat.completions.create({
    model: 'Qwen/Qwen3.5-4B',
    messages: [{ role: 'user', content: 'What is the capital of France?' }],
  });
  console.log(res.choices[0].message.content);
}
main();
```

**cURL**
```bash
curl "https://<YOUR_URL>.trycloudflare.com/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key_here" \
  -d '{
    "model": "Qwen/Qwen3.5-4B",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Quick test (no dependencies)**
```python
import urllib.request, json

url  = "https://<YOUR_URL>.trycloudflare.com/v1/chat/completions"
body = json.dumps({
    "messages": [{"role": "user", "content": "Hello! Who are you?"}],
    "max_tokens": 200
}).encode()

req  = urllib.request.Request(
    url, data=body, headers={"Content-Type": "application/json"})
data = json.loads(urllib.request.urlopen(req, timeout=60).read())
print(data["choices"][0]["message"]["content"])
```

---

### 5. Multimodal Agent (Images, Videos & PDFs)

The `/v1/agent` endpoint accepts any combination of files and a text prompt. The server handles downloading, decoding, and PDF text extraction automatically.

**Python**
```python
import requests

response = requests.post(
    "https://<YOUR_URL>.trycloudflare.com/v1/agent",
    headers={"Authorization": "Bearer your_api_key_here"},
    json={
        "prompt": "Read this invoice and extract the total amount.",
        "files":  ["https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"],
        "task":   "general"   # "general" | "coding" | "reasoning"
    }
)
print(response.json()["choices"][0]["message"]["content"])
```

**Node.js**
```javascript
const res = await fetch("https://<YOUR_URL>.trycloudflare.com/v1/agent", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": "Bearer your_api_key_here"
  },
  body: JSON.stringify({
    prompt: "Analyze these files and summarize them.",
    files: [
      "https://example.com/chart.png",       // image URL
      "https://example.com/report.pdf",      // PDF URL
      "data:video/mp4;base64,AAAAIGZ0eX..."  // base64 video
    ],
    task: "reasoning",
    stream: false
  })
});
const data = await res.json();
console.log(data.choices[0].message.content);
```

**Supported file inputs**

| Type | Example |
|---|---|
| Image URL | `https://example.com/photo.png` |
| PDF URL | `https://example.com/report.pdf` |
| Video URL | `https://example.com/clip.mp4` |
| Base64 data URL | `data:image/jpeg;base64,/9j/4AAQ...` |
| Local file path | `/content/myfile.pdf` |

---

### 6. Endpoints Reference

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI-compatible chat — text, image, streaming |
| `/v1/agent` | POST | Multimodal agent — images, PDFs, video, URLs |
| `/v1/models` | GET | List available models |
| `/health` | GET | Server status, uptime, VRAM, request count |
| `/logs` | GET | Last 100 proxy log lines (requires auth if key set) |
| `/logs/vllm` | GET | Last 100 raw vLLM engine log lines |

---

### 7. Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| Runtime disconnects on startup | Port conflict (8080 used by Colab) | Server already uses 8090 — do not kill any ports |
| `nvidia-smi not found` | CPU runtime | Runtime → Change runtime type → T4 GPU |
| vLLM crashes immediately | Wrong flag name in old vLLM | Already fixed — `--no-enable-log-requests` |
| Timeout after 900s | Old timeout value | Already fixed — timeout is 1200s |
| Cloudflare URL not showing | Tunnel took >90s | Re-run cell — usually resolves itself |
| Model slow on second run | Drive not mounted | Make sure Drive mounts at step 1 |
```
