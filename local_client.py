# local_client.py
import httpx
import base64

api_url_input = input("Enter the API URL from Colab: ")
if not api_url_input.strip():
    print("API URL cannot be empty.")
    exit()

API_URL = api_url_input.strip()

try:
    print("Fetching random dog image...")
    dog_api = "https://dog.ceo/api/breeds/image/random"
    r = httpx.get(dog_api, timeout=30)
    r.raise_for_status()
    image_url = r.json()["message"]
    print(f"Image URL: {image_url}")
except Exception as e:
    print(f"Failed to fetch dog image: {e}")
    exit()

try:
    img_data = httpx.get(image_url, timeout=30).content
except Exception as e:
    print(f"Failed to download image: {e}")
    exit()

b64 = base64.b64encode(img_data).decode()

prompt = input("Enter your prompt (e.g. describe the dog): ")
if not prompt.strip():
    prompt = "describe the dog"

try:
    print("Sending to AI...")
    res = httpx.post(
        API_URL,
        json={
            "image": b64,
            "prompt": prompt
        },
        timeout=120
    )
    res.raise_for_status()
    result = res.json()
    print("\nAI Response:")
    print(result.get("text", "No response text found."))
except Exception as e:
    print(f"Error: {e}")
