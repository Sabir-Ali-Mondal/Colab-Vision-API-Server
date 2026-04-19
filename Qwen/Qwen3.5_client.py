#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║   Qwen3.5 Local Client  ·  talks to the Colab server                ║
╠══════════════════════════════════════════════════════════════════════╣
║  QUICK START                                                         ║
║    export QWEN_URL="https://xxxx.trycloudflare.com"                 ║
║    export QWEN_KEY="mysecret"   # only if api_key was set in server ║
║                                                                      ║
║  USAGE                                                               ║
║    python client.py "Hello!"                                        ║
║    python client.py "Describe" --files photo.jpg                    ║
║    python client.py "Summarise" --files doc.pdf                     ║
║    python client.py "Analyse" --files clip.mp4 chart.png doc.pdf   ║
║    python client.py --stream "Write me a story"                     ║
║    python client.py --no-thinking "Quick answer pls"                ║
║    python client.py --task coding "Write a quicksort in Python"     ║
║    python client.py --interactive                                   ║
║    python client.py --health                                        ║
║                                                                      ║
║  AS A LIBRARY                                                        ║
║    from client import QwenClient                                    ║
║    c = QwenClient("https://xxxx.trycloudflare.com", key="secret")  ║
║    print(c.chat("Hello!"))                                          ║
║    print(c.agent("Describe", files=["image.jpg", "doc.pdf"]))       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
#  CLIENT CLASS
# ══════════════════════════════════════════════════════════════════════════════

class QwenClient:
    def __init__(self, base_url: str, key: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self._headers = {"Content-Type": "application/json"}
        if key:
            self._headers["Authorization"] = f"Bearer {key}"

    # ── convenience wrappers ──────────────────────────────────────────────────

    def health(self) -> dict:
        return self._get("/health")

    def models(self) -> list[str]:
        data = self._get("/v1/models")
        return [m["id"] for m in data.get("data", [])]

    def chat(
        self,
        messages: "list[dict] | str",
        *,
        max_tokens: int = 32768,
        thinking:   bool = True,
        task:       str  = "general",
        stream:     bool = False,
    ) -> str:
        """Standard OpenAI-style chat."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        models = self.models()
        model  = models[0] if models else "Qwen/Qwen3.5-4B"
        sp = _sampling(thinking, task)
        payload = {
            "model":      model,
            "messages":   messages,
            "max_tokens": max_tokens,
            "stream":     stream,
            "chat_template_kwargs": {"enable_thinking": thinking},
            **sp,
        }
        if stream:
            return self._stream("/v1/chat/completions", payload)
        return _extract(self._post("/v1/chat/completions", payload))

    def agent(
        self,
        prompt:     str,
        *,
        files:      "list[str] | None" = None,
        system:     str  = "You are a helpful AI assistant.",
        thinking:   bool = True,
        task:       str  = "general",
        max_tokens: int  = 32768,
        stream:     bool = False,
    ) -> str:
        """Multimodal /v1/agent endpoint."""
        payload = {
            "prompt":     prompt,
            "files":      [_encode(f) for f in (files or [])],
            "system":     system,
            "thinking":   thinking,
            "task":       task,
            "max_tokens": max_tokens,
            "stream":     stream,
        }
        if stream:
            return self._stream("/v1/agent", payload)
        return _extract(self._post("/v1/agent", payload))

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        req  = urllib.request.Request(
            self.base_url + path, data=data,
            headers=self._headers, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise RuntimeError(f"HTTP {e.code}: {body}")

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(
            self.base_url + path, headers=self._headers
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def _stream(self, path: str, payload: dict) -> str:
        """SSE streaming — prints tokens live, returns full text."""
        data = json.dumps(payload).encode()
        req  = urllib.request.Request(
            self.base_url + path, data=data,
            headers=self._headers, method="POST",
        )
        parts: list[str] = []
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                for raw_line in r:
                    line = raw_line.decode(errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        obj   = json.loads(chunk)
                        delta = obj["choices"][0]["delta"].get("content", "")
                        if delta:
                            print(delta, end="", flush=True)
                            parts.append(delta)
                    except Exception:
                        pass
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode()}")
        print()
        return "".join(parts)


# ── helpers ───────────────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def _extract(data: dict) -> str:
    if "error" in data:
        return f"[ERROR] {data['error']}"
    try:
        return _THINK_RE.sub("", data["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError):
        return json.dumps(data, indent=2)


def _encode(source: str) -> str:
    """Encode local file to data URL; pass URLs/data URLs through."""
    if source.startswith(("http://", "https://", "data:")):
        return source
    path = Path(source)
    if not path.exists():
        print(f"[warn] file not found: {source}", file=sys.stderr)
        return source
    raw  = path.read_bytes()
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    b64  = base64.b64encode(raw).decode()
    return f"data:{mime};base64,{b64}"


def _sampling(thinking: bool, task: str) -> dict:
    if thinking:
        if task == "coding":
            return dict(temperature=0.6, top_p=0.95)
        return dict(temperature=1.0, top_p=0.95, presence_penalty=1.5)
    return dict(temperature=0.7, top_p=0.8, presence_penalty=1.5)


# ══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE CHAT
# ══════════════════════════════════════════════════════════════════════════════

def interactive(client: QwenClient, thinking: bool, task: str) -> None:
    history: list[dict] = []
    models  = client.models()
    model   = models[0] if models else "Qwen/Qwen3.5-4B"
    print(f"\n🤖  Qwen3.5 Chat  |  model={model}  thinking={thinking}  task={task}")
    print("    Type 'quit' or press Ctrl+C to exit.\n")

    while True:
        try:
            user = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not user:
            continue
        if user.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        history.append({"role": "user", "content": user})
        print("Assistant: ", end="", flush=True)

        try:
            reply = client.chat(history, thinking=thinking, task=task, stream=True)
        except Exception as exc:
            print(f"\n[Error] {exc}")
            history.pop()
            continue

        clean = _THINK_RE.sub("", reply).strip()
        history.append({"role": "assistant", "content": clean})
        print()


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse():
    p = argparse.ArgumentParser(
        description="Qwen3.5 client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("prompt",         nargs="?",    default=None)
    p.add_argument("--url",          default=os.getenv("QWEN_URL", ""))
    p.add_argument("--key",          default=os.getenv("QWEN_KEY", ""))
    p.add_argument("--files",        nargs="+",    default=[], metavar="FILE")
    p.add_argument("--system",       default="You are a helpful AI assistant.")
    p.add_argument("--max-tokens",   type=int,     default=32768)
    p.add_argument("--task",         default="general",
                   choices=["general","coding","reasoning"])
    p.add_argument("--no-thinking",  action="store_true")
    p.add_argument("--stream",       action="store_true")
    p.add_argument("--interactive",  action="store_true")
    p.add_argument("--health",       action="store_true")
    p.add_argument("--models",       action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse()

    # resolve URL
    if not args.url:
        p = Path("/tmp/public_url.txt")
        if p.exists():
            args.url = p.read_text().strip()
    if not args.url:
        print("ERROR: set --url or QWEN_URL=https://xxxx.trycloudflare.com")
        sys.exit(1)

    client   = QwenClient(args.url, key=args.key)
    thinking = not args.no_thinking

    if args.health:
        print(json.dumps(client.health(), indent=2))
        return

    if args.models:
        for m in client.models():
            print(m)
        return

    if args.interactive:
        interactive(client, thinking, args.task)
        return

    if not args.prompt and not args.files:
        print(__doc__)
        sys.exit(0)

    prompt = args.prompt or ""

    if args.files:
        result = client.agent(
            prompt,
            files      = args.files,
            system     = args.system,
            thinking   = thinking,
            task       = args.task,
            max_tokens = args.max_tokens,
            stream     = args.stream,
        )
    else:
        result = client.chat(
            prompt,
            thinking   = thinking,
            task       = args.task,
            max_tokens = args.max_tokens,
            stream     = args.stream,
        )

    if not args.stream:
        print(result)


if __name__ == "__main__":
    main()
