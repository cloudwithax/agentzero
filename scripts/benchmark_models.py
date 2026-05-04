#!/usr/bin/env python3
"""Benchmark TTFT (Time To First Token) for all NVIDIA API models in MODEL_CATALOG."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

# Load .env before importing handler (model resolution happens at import time)
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
_env_path = _project_root / ".env"
load_dotenv(_env_path)

sys.path.insert(0, str(_project_root))

from handler import MODEL_CATALOG

BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
API_KEY = os.environ.get(
    "NVIDIA_API_KEY",
    "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",
)

# Override OPENAI_CLIENT env vars so handler doesn't route to non-NVIDIA endpoint
if os.environ.get("OPENAI_CLIENT_BASE_URL"):
    os.environ.pop("OPENAI_CLIENT_BASE_URL")
    os.environ.pop("OPENAI_CLIENT_API_KEY", None)

PROMPT = (
    "Write a one-sentence summary of quantum entanglement "
    "suitable for a curious 12-year-old."
)

STAGGER_SECONDS = 2.0  # Stay under 40 RPM
MAX_TOKENS = 128
TIMEOUT_SECONDS = 60


async def benchmark_one(
    session: aiohttp.ClientSession,
    model_id: str,
    name: str,
) -> dict:
    """Stream a chat completion and measure TTFT."""
    payload = {
        "model": model_id,
        "temperature": 0.6,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "messages": [{"role": "user", "content": PROMPT}],
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    t0 = time.monotonic()
    ttft_ms: float | None = None
    total_tokens = 0
    full_text = ""
    error: str | None = None

    try:
        async with session.post(
            BASE_URL, json=payload, headers=headers, timeout=TIMEOUT_SECONDS
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                error = f"HTTP {resp.status}: {body[:200]}"
                return {
                    "model_id": model_id,
                    "name": name,
                    "ttft_ms": None,
                    "total_s": None,
                    "tokens": 0,
                    "text": "",
                    "error": error,
                }

            async for line in resp.content:
                line_str = line.decode("utf-8", errors="replace").strip()
                if not line_str or not line_str.startswith("data: "):
                    continue
                data_str = line_str[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    if ttft_ms is None:
                        ttft_ms = (time.monotonic() - t0) * 1000
                    full_text += content
                    total_tokens += 1

        if ttft_ms is None:
            error = "No tokens received in stream"

    except asyncio.TimeoutError:
        error = "Timeout"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    total_s = time.monotonic() - t0
    return {
        "model_id": model_id,
        "name": name,
        "ttft_ms": round(ttft_ms) if ttft_ms else None,
        "total_s": round(total_s, 2),
        "tokens": total_tokens,
        "text": full_text.strip()[:120],
        "error": error,
    }


async def main():
    models = [(m["id"], m["name"]) for m in MODEL_CATALOG]
    results: list[dict] = []

    print(f"Benchmarking {len(models)} models (NVIDIA API)\n")
    print(f"Prompt: \"{PROMPT}\"")
    print(f"Stagger: {STAGGER_SECONDS}s between requests\n")

    async with aiohttp.ClientSession() as session:
        for i, (model_id, name) in enumerate(models):
            print(f"[{i+1}/{len(models)}] {name} ({model_id}) ...", flush=True)
            if i > 0:
                await asyncio.sleep(STAGGER_SECONDS)

            result = await benchmark_one(session, model_id, name)
            results.append(result)

            if result["error"]:
                print(f"  ❌ {result['error']}")
            else:
                print(
                    f"  ✅ TTFT: {result['ttft_ms']}ms | "
                    f"Total: {result['total_s']}s | "
                    f"Tokens: {result['tokens']}"
                )

    # --- Ranking ---
    print("\n" + "=" * 72)
    print("RANKING — by Time To First Token (ascending)")
    print("=" * 72)

    ranked = sorted(results, key=lambda r: r["ttft_ms"] if r["ttft_ms"] is not None else float("inf"))

    print(f"{'#':>3} {'Model':<44} {'TTFT':>7} {'Total':>7} {'Tok':>4}")
    print("-" * 72)

    for i, r in enumerate(ranked, 1):
        if r["error"]:
            print(f"{i:>3} {r['name']:<44} {'FAIL':>7}  {'—'}  {'—':>4}  {r['error'][:60]}")
        else:
            ttft = f"{r['ttft_ms']}ms"
            total = f"{r['total_s']}s"
            print(
                f"{i:>3} {r['name']:<44} {ttft:>7} {total:>7} {r['tokens']:>4}"
            )

    # --- Summary stats ---
    ok = [r for r in results if r["ttft_ms"] is not None]
    if ok:
        fastest = min(ok, key=lambda r: r["ttft_ms"])
        slowest = max(ok, key=lambda r: r["ttft_ms"])
        avg_ttft = sum(r["ttft_ms"] for r in ok) / len(ok)
        print(f"\n⏱  Fastest: {fastest['name']} ({fastest['ttft_ms']}ms)")
        print(f"🐢 Slowest: {slowest['name']} ({slowest['ttft_ms']}ms)")
        print(f"📊 Avg TTFT: {avg_ttft:.0f}ms across {len(ok)} models")
        print(f"❌ Errors: {len(results) - len(ok)}")

    return 0 if not any(r["error"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
