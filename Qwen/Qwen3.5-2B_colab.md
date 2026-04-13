| # | Feature | What it does | How to use |
|---|---|---|---|
| 1 | **Text Chat** | Send a prompt, get a response from Qwen3.5-2B | `POST /chat` with `{"prompt": "your question"}` |
| 2 | **Thinking Mode** | Model thinks step-by-step before answering (more accurate) | Add `"enable_thinking": true` in chat request |
| 3 | **Fast Mode** | Skip thinking, get faster response | Add `"enable_thinking": false` in chat request |
| 4 | **Streaming** | Get response word-by-word as it generates (like ChatGPT typing effect) | Add `"stream": true` in chat request |
| 5 | **Upload Image** | Save an image on the server for later use | `POST /upload` with image file |
| 6 | **Chat with Image** | Send an image + question, model describes/analyzes it | Upload image → get `file_id` → use in `POST /chat` with `"file_ids"` |
| 7 | **Upload Text File** | Upload `.txt`, `.csv`, `.json`, `.md` files | `POST /upload` with text file |
| 8 | **Upload PDF** | Upload a PDF, server extracts the text automatically | `POST /upload` with PDF file |
| 9 | **Chat with File** | Ask questions about an uploaded document or PDF | Upload file → get `file_id` → use in `POST /chat` with `"file_ids"` |
| 10 | **Upload + Chat in One Step** | Upload a file AND send a prompt in a single request | `POST /chat/upload` with multipart form (file + prompt) |
| 11 | **Multiple Files** | Attach multiple files (images + docs) to one chat message | Pass multiple IDs: `"file_ids": ["id1", "id2", "id3"]` |
| 12 | **List Files** | See all files currently stored on the server | `GET /files` |
| 13 | **Download File** | Download any previously uploaded file back | `GET /files/<file_id>` |
| 14 | **Delete File** | Remove a file from the server | `DELETE /files/<file_id>` |
| 15 | **Control Response Length** | Set how long or short the answer should be | Add `"max_tokens": 200` (short) or `"max_tokens": 2048` (long) |
| 16 | **Control Creativity** | Make responses more creative or more factual | Add `"temperature": 0.9` (creative) or `"temperature": 0.1` (factual) |
| 17 | **OpenAI SDK Compatible** | Use any OpenAI Python/JS library by just changing the base URL | Set `base_url=<tunnel_url>/v1` in OpenAI client |
| 18 | **Public URL** | Exposes your Colab server to the internet via Cloudflare tunnel | Auto-printed in logs when server starts |
| 19 | **Auto Restart** | If the AI model crashes, server restarts it automatically | Runs silently in background |
| 20 | **GPU Auto-Detection** | Detects your GPU and picks best settings automatically | Happens at startup, no action needed |
| 21 | **VRAM-Aware Context** | Sets max conversation length based on your GPU memory | T4 16GB → 32K tokens, A100 → 131K tokens |
| 22 | **Google Drive Cache** | Saves downloaded model to Drive so it doesn't re-download | Auto-mounts Drive at startup |
| 23 | **PDF Text Extraction** | Reads text out of PDF pages before sending to model | Happens automatically when you upload a PDF |
| 24 | **MIME Detection** | Figures out if a file is image/PDF/text by reading the actual file bytes, not just the extension | Happens automatically on every upload |
| 25 | **Health Check** | See if server is running and which model is loaded | `GET /health` |
| 26 | **Usage Stats** | See how many requests have been handled and files stored | `GET /metrics` |
| 27 | **CORS Enabled** | Any website or app can call this server without browser blocks | Built in, no config needed |
| 28 | **50MB File Limit** | Protects server from huge files crashing it | Auto-enforced, returns error if exceeded |
| 29 | **Retry on Failure** | If model call fails, automatically retries up to 3 times | Happens silently on every request |
| 30 | **Tunnel Auto-Restart** | If Cloudflare tunnel drops, it reconnects automatically | Runs silently in background |
