# Guide: How to Use Colab Vision API Server

This guide provides instructions to set up the AI server on Google Colab and run the client on your local machine.

## Part 1: Run the AI Server on Google Colab

1.  **Open Google Colab** and upload the `vllm_server.py` script.
2.  **Run the Server**: In a Colab cell, run the script.
    ```bash
    !python vllm_server.py
    ```
3.  **Get the Public URL**: After a few moments, the script will print a public URL ending in `.trycloudflare.com`. It will look similar to this: `ENDPOINT : https://your-unique-name.trycloudflare.com/v1/agent`.
4.  **Copy this URL** and keep the Colab notebook tab open.

```
Note on Startup Time: The first time you run the server, it takes about 10 minutes to download and install the large AI model. Subsequent runs will be much faster (around 2 to 3 minutes) because the model is cached and ready to use.
```

## Part 2: Run the Client on Your Local PC

1.  **Save the Client Script**: Save the Python code provided above as `local_client.py` on your computer.
2.  **Open Your Terminal**: Open a terminal or command prompt.
3.  **Install Library**: If `httpx` is not installed, run:
    ```bash
    pip install httpx
    ```
4.  **Run the Script**: Execute the script from your terminal.
    ```bash
    python local_client.py
    ```
5.  **Paste the URL**: The script will prompt you to enter the API URL. Paste the URL you copied from Colab and press Enter.
6.  **Enter a Prompt**: The script will then ask for your prompt. Type a question and press Enter.
7.  **Get the Result**: The script will send the request to the AI on Colab and print the response.

## Compatible AI Models

The server script uses `vLLM`, which supports a growing number of vision-language models. To use a different one, edit the `VISION_MODELS` list in `vllm_server.py` and place your desired model's Hugging Face ID at the top of the list.

**Important**: The free Google Colab tier has limited GPU memory (VRAM). Models larger than ~7 billion parameters may fail to load. Start with smaller models first.

Below is a categorized list of models known to be compatible.

### Recommended for Free Colab Tier (Smaller Models)

*   **Qwen2-VL** (Balanced performance)
    *   `Qwen/Qwen2.5-VL-3B-Instruct`
    *   `Qwen/Qwen2-VL-2B-Instruct`
*   **MiniCPM-V** (Excellent small models)
    *   `openbmb/MiniCPM-V-2`
    *   `openbmb/MiniCPM-Llama3-V-2_5`
*   **Phi-3.5-Vision** (Strong performance from Microsoft)
    *   `microsoft/Phi-3.5-vision-instruct`
*   **InternVL2** (Powerful and efficient)
    *   `OpenGVLab/InternVL2-4B`

### LLaVA Family (Popular & Widely Supported)

*   **LLaVA 1.5**
    *   `llava-hf/llava-1.5-7b-hf`
*   **LLaVA 1.6**
    *   `llava-hf/llava-v1.6-mistral-7b-hf`
    *   `llava-hf/llava-v1.6-vicuna-7b-hf`

### Other High-Performance Models (May require Colab Pro)

*   **Fuyu**
    *   `adept/fuyu-8b`
*   **BLIP-2**
    *   `Salesforce/blip2-opt-2.7b`
*   **Larger LLaVA Models**
    *   `llava-hf/llava-1.5-13b-hf`
*   **Larger InternVL2 Models**
    *   `OpenGVLab/InternVL2-8B`
