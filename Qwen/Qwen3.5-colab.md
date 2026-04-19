### `server.py` — 969 lines, 10 sections

| Section | Feature |
|---------|---------|
| **§1 Deps** | Auto-installs fastapi, uvicorn, httpx, Pillow, pypdf, pdfplumber, pytesseract, aiofiles + vLLM nightly. Failures on optional packages are warnings, not crashes. |
| **§2 Hardware** | `nvidia-smi` probe → auto-selects `max_model_len` (16K→262K based on VRAM). T4=32K, A100-40G=131K, A100-80G=262K. |
| **§3 Model cache** | Checks Google Drive → HF snapshot cache → downloads. Picks newest HF snapshot automatically. |
| **§4 vLLM** | Launches with `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`. Watchdog restarts on crash. |
| **§5 Cloudflare** | Auto-installs binary, extracts public URL, watchdog restarts tunnel on crash. |
| **§6 Media** | PDF→text via pdfplumber → pypdf → pytesseract OCR fallback. Image/video/URL/base64/file-path all normalised to OpenAI content blocks. |
| **§7 Rate limiter** | Per-IP sliding-window (configurable rpm, 0=off). Returns `Retry-After` header. |
| **§8 Sampling** | All 4 official Qwen3.5 presets: thinking+general, thinking+coding, thinking+reasoning, instruct modes. |
| **§9 FastAPI** | Auth middleware, access logger, `/health`, `/logs`, `/logs/vllm`, `/v1/models`, `/v1/agent`, `/v1/chat/completions`, catch-all passthrough. |
| **§10 Main** | CLI wins > `COLAB_CONFIG` > defaults. `parse_known_args` silences Jupyter's `-f kernel.json`. |

### Node.js client usage (your OpenRouter-style snippet)

```js
const res = await fetch(`${QWEN_URL}/v1/agent`, {
  method: "POST",
  headers: { "Content-Type": "application/json",
             "Authorization": `Bearer ${API_KEY}` },
  body: JSON.stringify({
    prompt: "Analyse all inputs",
    files:  [base64Video, base64PDF, base64Image],  // data URLs or http URLs
    task:   "general",
    stream: false,
  })
});
```
