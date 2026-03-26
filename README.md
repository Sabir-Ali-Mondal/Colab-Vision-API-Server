# Colab Vision API Server

This project allows you to run a vision-language model on a free Google Colab GPU and expose it as a public, OpenAI-compatible API using a Cloudflare Tunnel. This enables you to send images and text prompts from your local computer and receive responses from the AI running in the cloud.

## Goal

The primary objective is to bridge powerful, GPU-intensive AI models with local development environments. By hosting the model on Colab, you can leverage free GPU resources for tasks like image analysis without needing powerful local hardware.

## How It Works

1.  **Server on Colab**: A Python script using `vLLM` serves a vision-language model on a Google Colab instance.
2.  **Public Tunnel**: The script initiates a secure Cloudflare Tunnel, exposing the Colab server to the internet with a public URL.
3.  **Local Client**: A separate Python script on your local machine sends API requests, containing an image and a prompt, to this public URL.
4.  **AI Response**: The model on Colab processes the request and sends the generated text back to your local machine.

## Features

*   **Free GPU Usage**: Operates on Google Colab's free tier.
*   **Public API**: Creates a temporary public API endpoint for your model.
*   **OpenAI Compatible**: The server endpoint is structured for compatibility with standard OpenAI API requests.
*   **Simple Setup**: Requires minimal configuration to get started.

## Getting Started

Follow the `GUIDE.md` for a step-by-step tutorial on how to set up the server and run the client.

## Search Keywords

github run llm on google colab with cloudflare tunnel, expose colab gpu as api, vllm server on google colab with public url, run openai compatible api on colab free gpu, list of vision models compatible with vLLM
