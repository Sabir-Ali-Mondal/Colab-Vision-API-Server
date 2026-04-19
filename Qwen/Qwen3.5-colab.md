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

## Here are the instructions to run it based on your environment:

---

### Option 1: Running in Google Colab (Recommended)

**Step 1: Open Colab and Enable a GPU**
1. Go to [Google Colab](https://colab.research.google.com/).
2. Create a New Notebook.
3. Go to **Runtime > Change runtime type**.
4. Select a **T4 GPU** (or A100/V100 if you have Colab Pro) and click Save.

**Step 2: Paste the Code**
Copy the entire script you provided and paste it into the first cell of your Colab notebook.

**Step 3: Configure (Optional)**
Look at the top of the script for the `COLAB_CONFIG` block. You can change:
* `"model": "4b"` (Change to `"2b"` if you want it to run faster/use less memory).
* `"api_key": "your_password"` (Highly recommended to prevent others from using your server).

**Step 4: Run the Cell**
Click the **Play** button on the cell. 
* *Note: The first time you run it, it will take 5–10 minutes to install vLLM and download the AI model.*
* Once it is ready, scroll to the bottom of the output. Look for a box that says:
  `🔗 PUBLIC URL : https://some-random-words.trycloudflare.com`

---

### Option 2: Running on a Local Linux PC or Cloud VM (RunPod, AWS, etc.)

*Prerequisites: You must be on Linux (Ubuntu recommended), have Python 3.10+, and have an NVIDIA GPU with CUDA installed. (vLLM does not run natively on Windows/Mac).*

**Step 1: Save the file**
Open a terminal, create a file named `server.py`, and paste the code inside.
```bash
nano server.py
# Paste the code, then press Ctrl+O, Enter, Ctrl+X to save and exit
```

**Step 2: Make it executable**
```bash
chmod +x server.py
```

**Step 3: Run it**
You can run it simply by typing:
```bash
./server.py
```
*It will prompt you to choose a model, install dependencies automatically, download the model, and generate a public Cloudflare URL.*

**Alternative: Run with CLI Arguments**
You can bypass the interactive prompts by passing arguments (as mentioned in the script's docstring):
```bash
./server.py --model 4b --api-key mysecretkey
```

---

### How to use your new API

Once the server is running and prints the Cloudflare `PUBLIC URL`, you can connect to it exactly like the OpenAI API. 

Here is a test you can run from **any other computer** using your terminal:

```bash
curl -X POST "https://YOUR-CLOUDFLARE-URL.trycloudflare.com/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer YOUR_API_KEY_IF_YOU_SET_ONE" \
     -d '{
       "model": "4b",
       "messages": [
         {"role": "user", "content": "Write a haiku about artificial intelligence."}
       ]
     }'
```

**Using it in Python (with the official OpenAI library):**
```python
# pip install openai
from openai import OpenAI

client = OpenAI(
    base_url="https://YOUR-CLOUDFLARE-URL.trycloudflare.com/v1",
    api_key="your_api_key" # Or "empty" if you didn't set one
)

response = client.chat.completions.create(
    model="4b",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True # The server supports streaming!
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```
