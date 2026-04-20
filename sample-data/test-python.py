import base64
import requests
import mimetypes
from pathlib import Path

# =========================
# CONFIG
# =========================
BASE_URL = "https://<YOUR_URL>.trycloudflare.com/v1/agent"
API_KEY  = "your_api_key_here"   # or "" if open

# =========================
# HELPERS
# =========================
def encode_file_to_base64(path: str):
    path = Path(path)
    mime, _ = mimetypes.guess_type(path)

    if mime is None:
        raise ValueError(f"Cannot detect MIME type for {path}")

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    return f"data:{mime};base64,{encoded}"


# =========================
# MAIN FUNCTION
# =========================
def run_agent(
    prompt: str,
    files: list = None,
    urls: list = None,
    task: str = "general",
    stream: bool = False
):
    """
    files: local file paths
    urls: direct URLs (image/pdf/video)
    """

    payload_files = []

    # ---- Local files (converted to base64)
    if files:
        for f in files:
            payload_files.append(encode_file_to_base64(f))

    # ---- Remote URLs (used directly)
    if urls:
        payload_files.extend(urls)

    headers = {
        "Content-Type": "application/json",
    }

    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    body = {
        "prompt": prompt,
        "files": payload_files,
        "task": task,
        "stream": stream
    }

    response = requests.post(BASE_URL, headers=headers, json=body, timeout=300)

    if response.status_code != 200:
        raise Exception(f"Error {response.status_code}: {response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"]


# =========================
# EXAMPLE USAGE
# =========================
if __name__ == "__main__":

    result = run_agent(
        prompt="Analyze all inputs and give a combined summary.",
        
        # Local files (auto base64)
        files=[
            "sample.pdf",
            "image.jpg",
            "video.mp4"
        ],

        # OR remote URLs
        urls=[
            "https://example.com/chart.png",
            "https://example.com/report.pdf"
        ],

        task="reasoning"
    )

    print("\n=== RESPONSE ===\n")
    print(result)
