# Qwen3.5 Server Setup & API Quick Start Guide

Welcome to your personal Qwen3.5 server. This guide covers how to start your server in Google Colab (or locally), how to get your connection credentials, and how to use the OpenAI-compatible API and Multimodal Agent.

---

### 1. Setup & Running the Server

**Running in Google Colab (Recommended)**
1. Go to Google Colab and create a New Notebook.
2. Go to **Runtime > Change runtime type** and select a **T4 GPU**. Click Save.
3. Paste the entire `server.py` script into the first cell.
4. Near the top of the script, find the `COLAB_CONFIG` block. Set your desired API key here to secure your server:
   ```python
   COLAB_CONFIG: dict = {
       "model":           "4b",
       "api_key":         "your_secret_password_here", # Set your key here
       # ... other settings
   }
   ```
5. Click the **Play** button on the cell to run it. 
6. The first run will take 5-10 minutes to install dependencies (vLLM) and download the Qwen model.

**Running on a Local Linux PC / Cloud VM**
1. Save the code to a file named `server.py`.
2. Make it executable: `chmod +x server.py`
3. Run it via terminal, passing your API key as an argument:
   ```bash
   ./server.py --model 4b --api-key your_secret_password_here
   ```

---

### 2. Getting Your Link and Key

Once the server has finished downloading and starting, it will generate a secure public tunnel via Cloudflare. Scroll to the bottom of the output logs. 

You are looking for a banner that looks like this:
```text
==================================================================
  [LINK] PUBLIC URL       : https://random-words-here.trycloudflare.com
  chat/completions        : https://random-words-here.trycloudflare.com/v1/chat/completions
  agent (multimodal)      : https://random-words-here.trycloudflare.com/v1/agent
==================================================================
```

*   **Your Base URL** is the `PUBLIC URL` link with `/v1` added to the end.
*   **Your API Key** is whatever you typed into the `COLAB_CONFIG` or passed via the `--api-key` argument.

---

### 3. Connection Details

To connect your existing apps, scripts, or UI clients (like Chatbox or AnythingLLM), use these details:

* **Base URL:** `https://<YOUR_CLOUDFLARE_URL>.trycloudflare.com/v1`
* **API Key:** `<YOUR_API_KEY>` *(or leave empty if you didn't set one)*
* **Available Models:** `"4b"` or `"2b"`

---

### 4. Standard Chat Completions (OpenAI Compatible)

Because the server uses standard formatting, you can use the official OpenAI SDKs without changing your code logic.

**Python** (`pip install openai`)
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<YOUR_CLOUDFLARE_URL>.trycloudflare.com/v1",
    api_key="your_api_key_here" # Use "empty" if no key is set
)

response = client.chat.completions.create(
    model="4b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a python script to reverse a string."}
    ],
    stream=True # Set to True for real-time typewriter effect
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

**Node.js** (`npm install openai`)
```javascript
import OpenAI from 'openai';

const openai = new OpenAI({
  baseURL: 'https://<YOUR_CLOUDFLARE_URL>.trycloudflare.com/v1',
  apiKey: 'your_api_key_here' // Use "empty" if no key is set
});

async function main() {
  const completion = await openai.chat.completions.create({
    model: '4b',
    messages: [{ role: 'user', content: 'What is the capital of France?' }],
  });
  console.log(completion.choices[0].message.content);
}
main();
```

**cURL**
```bash
curl "https://<YOUR_CLOUDFLARE_URL>.trycloudflare.com/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key_here" \
  -d '{
    "model": "4b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

### 5. Multimodal Agent (Images, Videos, & PDFs)

Your server includes a custom `/v1/agent` endpoint that acts like OpenRouter. You can pass raw URLs, base64 strings, or file paths, and the server will automatically extract the text (from PDFs) or frames (from Video/Images) before passing it to Qwen3.5.

**Node.js (Fetch)**
```javascript
const response = await fetch("https://<YOUR_CLOUDFLARE_URL>.trycloudflare.com/v1/agent", {
  method: "POST",
  headers: { 
      "Content-Type": "application/json",
      "Authorization": "Bearer your_api_key_here" 
  },
  body: JSON.stringify({
    prompt: "Analyze these files and summarize them.",
    files: [
        "https://example.com/chart.png",           // Image URL
        "https://example.com/report.pdf",          // PDF URL
        "data:video/mp4;base64,AAAAIGZ0eX..."      // Base64 Video
    ],
    task: "reasoning", // Options: "general", "coding", "reasoning"
    stream: false
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);
```

**Python (Requests)**
```python
import requests

url = "https://<YOUR_CLOUDFLARE_URL>.trycloudflare.com/v1/agent"
headers = {"Authorization": "Bearer your_api_key_here"}

payload = {
    "prompt": "Read this invoice and extract the total amount.",
    "files": ["https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"],
    "task": "general"
}

response = requests.post(url, json=payload, headers=headers)
print(response.json()["choices"][0]["message"]["content"])
```

---

### 6. Available Endpoints Overview

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/v1/chat/completions` | `POST` | Standard OpenAI text/vision chat logic. |
| `/v1/agent` | `POST` | Universal file handler (Images, PDFs, Video, URLs). |
| `/v1/models` | `GET` | Returns list of available models (useful for UI dropdowns). |
| `/health` | `GET` | Check server uptime, VRAM status, and total requests handled. |
| `/logs` | `GET` | Fetch the last 100 lines of the proxy server logs (requires Auth). |
