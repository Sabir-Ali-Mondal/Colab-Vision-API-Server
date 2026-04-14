# Colab Vision API Server

Run vision AI models on a free Google Colab GPU (or any Linux machine) and use them from your local PC through a public API.

---

## Overview

This project lets you run a vision-language model in the cloud and access it locally.

It uses:

* vLLM to serve the model
* Cloudflare Tunnel to create a public URL
* OpenAI-compatible API for easy integration

You can send images and text from your local machine and get AI responses from Colab.

![Colab Vision API Demo](https://github.com/Sabir-Ali-Mondal/Colab-Vision-API-Server/blob/0fc3ec3c31914f0183df4ac8cbcf352469301a4d/colab-vision-api-demonstration.png?raw=true)

---

## Goal

The goal is to make powerful AI models easy to use without needing a strong local computer.

Run the model in the cloud and access it locally through a simple API.

---

## How It Works

1. A Python script runs a vision model using vLLM on Colab or Linux
2. A Cloudflare Tunnel exposes the server as a public URL
3. Your local PC sends requests (image + prompt)
4. The model processes and returns a response

---

## Features

* Works on Colab, local Linux, or cloud GPU
* Uses free Colab GPU
* OpenAI-compatible API
* Supports image and text input
* Public API via Cloudflare Tunnel
* Automatic hardware detection (CPU / GPU / TPU)
* Simple setup

---

## Run Anywhere

This project is not limited to Colab.

You can run it on:

* Google Colab
* Local Linux (WSL2 supported)
* Cloud GPU servers

It automatically detects your hardware and configures itself.

---

## Getting Started

Follow the instructions in **GUIDE.md** to:

* Run the server
* Get the public API URL
* Send requests from your local PC

---

## Search Keywords

google colab ai, run llm on colab, free gpu ai, vllm server, openai compatible api, cloudflare tunnel api, expose colab gpu as api, multimodal ai, vision language model

---

## Summary

Turn free Google Colab GPU into your own OpenAI-style AI server using a single script file. It combines vLLM, Cloudflare tunnel, and vision models into a simple workflow that runs on Colab, local Linux, or cloud GPU with no setup. The script installs dependencies, detects hardware (CPU/GPU/TPU), runs the model, exposes a public API, and lets you send image and text requests from your local PC—no GPU needed, no complex setup, just a clean and practical AI backend.
