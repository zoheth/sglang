#!/usr/bin/env python3
"""
Test script for Qwen2.5-VL-7B with EAGLE3 speculative sampling.
Based on the training data format with image-text conversations.
"""

import asyncio
import argparse
import json
import os
import time
import base64
from pathlib import Path
from typing import List, Dict, Any

import aiohttp
import numpy as np

TIMEOUT = 6 * 60 * 60
MIN_DIVISOR = 1e-6


def get_headers():
    """Get HTTP headers for API requests."""
    headers = {"Content-Type": "application/json"}
    if api_key := os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 data URL.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded data URL string
    """
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Detect image format from extension
    ext = Path(image_path).suffix.lower()
    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")

    return f"data:{mime_type};base64,{image_data}"


def load_training_data(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    Load training data from JSONL file.

    Expected format:
    {
        "id": "1",
        "image": "/path/to/image.jpg",
        "conversations": [
            {"role": "user", "content": "What color is this image?"},
            {"role": "assistant", "content": "This image shows a solid red color..."}
        ]
    }

    Args:
        jsonl_path: Path to the JSONL file

    Returns:
        List of processed data items
    """
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            item = json.loads(line)

            # Extract user question from conversations
            user_content = None
            for conv in item["conversations"]:
                if conv["role"] == "user":
                    user_content = conv["content"]
                    break

            if user_content is None:
                continue

            # Convert local image path to base64 data URL
            image_path = item["image"]
            if os.path.exists(image_path):
                image_data_url = encode_image_to_base64(image_path)
            else:
                print(f"Warning: Image not found: {image_path}, skipping...")
                continue

            data.append({
                "id": item["id"],
                "content": user_content,
                "image_data": [image_data_url],
                "video_data": [],
            })

    return data


def build_content(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build OpenAI API compatible content format.

    Args:
        item: Data item with content and image_data

    Returns:
        List of content blocks (text + images)
    """
    content = [{"type": "text", "text": item["content"]}]

    for img in item.get("image_data", []):
        content.append({
            "type": "image_url",
            "image_url": {"url": img}
        })

    for vid in item.get("video_data", []):
        content.append({
            "type": "video_url",
            "video_url": {"url": vid}
        })

    return content


async def request_one(session, semaphore, url, body):
    """
    Send a single request to the OpenAI-compatible API.

    Args:
        session: aiohttp ClientSession
        semaphore: asyncio Semaphore for concurrency control
        url: API endpoint URL
        body: JSON request body

    Returns:
        Dict with timing and token statistics
    """
    async with semaphore:
        st = time.perf_counter()
        ttft, prompt_tokens, completion_tokens = 0.0, 0, 0
        response_text = ""

        async with session.post(url, data=body, headers=get_headers()) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(f"HTTP {resp.status}: {error_text}")

            async for line in resp.content:
                line = line.strip()
                if not line:
                    continue

                chunk = line.decode().removeprefix("data: ")
                if chunk == "[DONE]":
                    break

                try:
                    data = json.loads(chunk)
                except json.JSONDecodeError:
                    continue

                if data.get("choices") and ttft == 0.0:
                    if data["choices"][0].get("delta", {}).get("content"):
                        ttft = time.perf_counter() - st

                # Accumulate response text
                if data.get("choices"):
                    delta_content = data["choices"][0].get("delta", {}).get("content", "")
                    response_text += delta_content

                if usage := data.get("usage"):
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)

        latency = time.perf_counter() - st

    return {
        "ttft": ttft,
        "latency": latency,
        "decode_time": latency - ttft,
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "response_text": response_text,
    }


def stats(arr):
    """Calculate statistics for an array of values."""
    arr = np.array(arr)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
    }


async def bench_batch(url, model, prompts, concurrency, verbose=False, **kwargs):
    """
    Benchmark a batch of prompts.

    Args:
        url: Base URL of the API server
        model: Model name
        prompts: List of prompt items
        concurrency: Maximum concurrent requests
        verbose: Print individual responses
        **kwargs: Additional sampling parameters

    Returns:
        Dict with benchmark results
    """
    bodies = [
        json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": build_content(p)}],
            "stream": True,
            "stream_options": {"include_usage": True},
            **kwargs
        })
        for p in prompts
    ]

    semaphore = asyncio.Semaphore(concurrency or len(bodies))
    timeout = aiohttp.ClientTimeout(total=TIMEOUT)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        wall_start = time.perf_counter()
        results = await asyncio.gather(*[
            request_one(session, semaphore, f"{url}/v1/chat/completions", b)
            for b in bodies
        ], return_exceptions=True)
        wall_time = time.perf_counter() - wall_start

    # Filter out exceptions
    successful_results = []
    for i, r in enumerate(results):
        if isinstance(r, dict):
            successful_results.append(r)
            if verbose:
                print(f"\n{'='*60}")
                print(f"Request {i+1} (ID: {prompts[i].get('id', 'N/A')})")
                print(f"Prompt: {prompts[i]['content']}")
                print(f"Response: {r['response_text'][:200]}...")
                print(f"Tokens: {r['input_tokens']} in, {r['output_tokens']} out")
                print(f"TTFT: {r['ttft']:.3f}s, E2E: {r['latency']:.3f}s")
        elif isinstance(r, Exception):
            print(f"\n❌ Request {i+1} failed: {r}")

    results = successful_results
    if not results:
        return None

    total_input = sum(r["input_tokens"] for r in results)
    total_output = sum(r["output_tokens"] for r in results)
    total_tokens = total_input + total_output

    per_req_prefill_tps = [
        r["input_tokens"] / max(r["ttft"], MIN_DIVISOR)
        for r in results if r["ttft"] > 0
    ]
    per_req_decode_tps = [
        r["output_tokens"] / max(r["decode_time"], MIN_DIVISOR)
        for r in results if r["decode_time"] > 0
    ]

    return {
        "num_requests": len(results),
        "wall_time": wall_time,
        "throughput": {
            "prefill_tps": stats(per_req_prefill_tps) if per_req_prefill_tps else None,
            "decode_tps": stats(per_req_decode_tps) if per_req_decode_tps else None,
            "overall_tps": total_tokens / max(wall_time, MIN_DIVISOR),
        },
        "latency": {
            "ttft": stats([r["ttft"] for r in results]),
            "decode_time": stats([r["decode_time"] for r in results]),
            "e2e": stats([r["latency"] for r in results]),
        },
        "tokens": {
            "input": stats([r["input_tokens"] for r in results]),
            "output": stats([r["output_tokens"] for r in results]),
            "total_input": total_input,
            "total_output": total_output,
        },
    }


def print_report(result):
    """Print a formatted benchmark report."""
    def fmt_stats(s, unit=""):
        return (
            f"{s['mean']:.2f}{unit} "
            f"(std={s['std']:.2f}, p50={s['p50']:.2f}, "
            f"p90={s['p90']:.2f}, p99={s['p99']:.2f})"
        )

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS - Qwen2.5-VL-7B + EAGLE3")
    print("=" * 60)

    print(f"\n[Summary]")
    print(f"  Requests: {result['num_requests']}")
    print(f"  Wall time: {result['wall_time']:.2f}s")
    print(f"  Total tokens: {result['tokens']['total_input']} in + "
          f"{result['tokens']['total_output']} out")

    print(f"\n[Throughput]")
    tp = result["throughput"]
    if tp["prefill_tps"]:
        print(f"  Prefill:  {fmt_stats(tp['prefill_tps'], ' tok/s')}")
    if tp["decode_tps"]:
        print(f"  Decode:   {fmt_stats(tp['decode_tps'], ' tok/s')}")
    print(f"  Overall:  {tp['overall_tps']:.2f} tok/s")

    print(f"\n[Latency]")
    lat = result["latency"]
    print(f"  TTFT:     {fmt_stats(lat['ttft'], 's')}")
    print(f"  Decode:   {fmt_stats(lat['decode_time'], 's')}")
    print(f"  E2E:      {fmt_stats(lat['e2e'], 's')}")

    print(f"\n[Tokens per request]")
    tok = result["tokens"]
    print(f"  Input:    {fmt_stats(tok['input'])}")
    print(f"  Output:   {fmt_stats(tok['output'])}")
    print("=" * 60 + "\n")


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen2.5-VL-7B with EAGLE3 speculative sampling"
    )
    parser.add_argument("--url", required=True, help="API server URL")
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to JSONL training data file"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (auto-detect if not specified)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Number of samples to test (default: all)"
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=4,
        help="Maximum concurrent requests"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print individual responses"
    )
    parser.add_argument(
        "--output_json",
        action="store_true",
        help="Output results as JSON"
    )
    args = parser.parse_args()

    # Auto-detect model name if not specified
    if args.model is None:
        try:
            resp = await asyncio.to_thread(
                lambda: __import__("requests").get(
                    f"{args.url}/v1/models",
                    headers=get_headers()
                ).json()
            )
            args.model = resp["data"][0]["id"]
            print(f"✓ Auto-detected model: {args.model}")
        except Exception as e:
            print(f"❌ Failed to auto-detect model: {e}")
            print("Please specify --model explicitly")
            return

    # Load training data
    print(f"Loading data from: {args.dataset_path}")
    data = load_training_data(args.dataset_path)
    print(f"✓ Loaded {len(data)} samples")

    # Limit batch size if specified
    if args.batch_size:
        data = data[:args.batch_size]
        print(f"✓ Using first {len(data)} samples")

    # Sampling parameters
    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_new_tokens
    }

    print(f"\n🚀 Starting benchmark...")
    print(f"   URL: {args.url}")
    print(f"   Model: {args.model}")
    print(f"   Concurrency: {args.max_concurrency}")
    print(f"   Sampling: temp={args.temperature}, top_p={args.top_p}, "
          f"max_tokens={args.max_new_tokens}")

    # Run benchmark
    result = await bench_batch(
        args.url,
        args.model,
        data,
        args.max_concurrency,
        verbose=args.verbose,
        **sampling_params
    )

    if not result:
        print("❌ No successful results")
        return

    # Print results
    if args.output_json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)


if __name__ == "__main__":
    asyncio.run(main())
