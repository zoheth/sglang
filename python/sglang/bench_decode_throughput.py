"""
Benchmark decode throughput at a target batch size with long-context prefill.

With --prefill-max-requests 1 on the server, requests are prefilled one by one.
The scheduler always prioritizes prefill over decode, so all requests will be
prefilled first, then they decode together at the target batch size.

Usage:
  # Step 1: Launch server (prefill one at a time)
  python -m sglang.launch_server \
      --model-path <model> \
      --prefill-max-requests 1 \
      --chunked-prefill-size 8192 \
      --enable-hisparse

  # Step 2: Run this benchmark
  python -m sglang.bench_decode_throughput \
      --num-requests 64 \
      --input-len 131072 \
      --output-len 256
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import requests


def send_one_request(
    url: str,
    input_ids: List[int],
    output_len: int,
    req_id: int,
) -> Dict:
    """Send one streaming request and record per-token timestamps."""
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "max_new_tokens": output_len,
            "temperature": 0,
        },
        "stream": True,
    }

    # Each SSE chunk corresponds to one decode step. With speculative decoding
    # (MTP), one step accepts multiple tokens but still emits a single chunk,
    # so we diff the cumulative meta_info.completion_tokens to get the real
    # per-step token count. Note: output_ids in default (non-incremental)
    # streaming mode is the full cumulative list, NOT a delta — do not count it.
    token_timestamps = []  # list of (timestamp, tokens_in_this_step)
    first_token_time = None
    prev_completion_tokens = 0
    final_meta_info = {}

    with requests.post(url, json=payload, stream=True) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[len("data: "):])
            if "text" not in data:
                continue
            t = time.perf_counter()
            meta = data.get("meta_info", {})
            cur_completion = meta.get("completion_tokens", prev_completion_tokens + 1)
            step_tokens = cur_completion - prev_completion_tokens
            prev_completion_tokens = cur_completion
            if step_tokens > 0:
                token_timestamps.append((t, step_tokens))
                if first_token_time is None:
                    first_token_time = t
            # The final chunk carries server-computed spec metrics.
            final_meta_info = meta

    return {
        "req_id": req_id,
        "first_token_time": first_token_time,
        "token_timestamps": token_timestamps,
        "num_output_tokens": prev_completion_tokens,
        "meta_info": final_meta_info,
    }


def run_benchmark(args):
    url = f"{args.base_url}/generate"
    input_ids = [1] * args.input_len

    print(f"=== Decode Throughput Benchmark ===")
    print(f"Server:        {args.base_url}")
    print(f"Num requests:  {args.num_requests}")
    print(f"Input length:  {args.input_len}")
    print(f"Output length: {args.output_len}")
    print()

    print(f"Sending {args.num_requests} requests concurrently...")
    t_start = time.perf_counter()

    results = []
    with ThreadPoolExecutor(max_workers=args.num_requests) as executor:
        futures = {
            executor.submit(send_one_request, url, input_ids, args.output_len, i): i
            for i in range(args.num_requests)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            rid = result["req_id"]
            if result["first_token_time"]:
                ttft = result["first_token_time"] - t_start
                print(f"  Request {rid} first token at {ttft:.2f}s, "
                      f"generated {result['num_output_tokens']} tokens")

    t_end = time.perf_counter()

    # --- Analysis ---
    valid_results = [r for r in results if r["first_token_time"] is not None]
    if not valid_results:
        print("ERROR: No requests produced output tokens.")
        return

    last_prefill_done = max(r["first_token_time"] for r in valid_results)
    first_prefill_done = min(r["first_token_time"] for r in valid_results)

    # Count tokens generated AFTER all prefills are done (pure decode phase).
    # Each entry is (timestamp, tokens_in_this_step) — with MTP one step can
    # emit multiple tokens in a single SSE chunk. decode_steps counts chunks
    # so we can compute the spec-decode accept rate (tokens / step).
    decode_tokens = 0
    decode_steps = 0
    decode_end_time = 0.0
    for r in valid_results:
        for ts, step_tokens in r["token_timestamps"]:
            if ts > last_prefill_done:
                decode_tokens += step_tokens
                decode_steps += 1
                decode_end_time = max(decode_end_time, ts)

    total_tokens = sum(r["num_output_tokens"] for r in valid_results)
    total_prefill_time = last_prefill_done - t_start
    decode_duration = decode_end_time - last_prefill_done if decode_tokens > 0 else 0

    print(f"\n=== Results ===")
    print(f"Total time:                  {t_end - t_start:.2f} s")
    print(f"Prefill phase:               {total_prefill_time:.2f} s  "
          f"(first done: {first_prefill_done - t_start:.2f}s, "
          f"last done: {total_prefill_time:.2f}s)")
    print(f"  Avg prefill per request:   {total_prefill_time / args.num_requests:.2f} s")
    print()
    print(f"Decode phase (after all prefills done):")
    print(f"  Duration:                  {decode_duration:.2f} s")
    print(f"  Tokens generated:          {decode_tokens}")
    print(f"  Decode steps (chunks):     {decode_steps}")
    if decode_steps > 0:
        accept_len = decode_tokens / decode_steps
        print(f"  Client-side tokens/step:   {accept_len:.3f}  "
              f"(1.0 means no speculation)")
    print(f"  Decode batch size:         {args.num_requests}")
    if decode_duration > 0:
        throughput = decode_tokens / decode_duration
        per_req = throughput / args.num_requests
        print(f"  Throughput (total):        {throughput:.2f} tokens/s")
        print(f"  Throughput (per request):  {per_req:.2f} tokens/s")

    # Server-side speculative-decoding metrics (authoritative when MTP is on).
    spec_accept_lengths = [
        r["meta_info"].get("spec_accept_length")
        for r in valid_results
        if r["meta_info"].get("spec_accept_length") is not None
    ]
    spec_accept_rates = [
        r["meta_info"].get("spec_accept_rate")
        for r in valid_results
        if r["meta_info"].get("spec_accept_rate") is not None
    ]
    if spec_accept_lengths:
        print()
        print(f"Server-side spec-decoding metrics (avg across requests):")
        print(f"  spec_accept_length:        "
              f"{sum(spec_accept_lengths) / len(spec_accept_lengths):.3f}")
        if spec_accept_rates:
            print(f"  spec_accept_rate:          "
                  f"{sum(spec_accept_rates) / len(spec_accept_rates):.3f}")
    print()
    print(f"Total output tokens:         {total_tokens}")
    print(f"Overall throughput:          {total_tokens / (t_end - t_start):.2f} tokens/s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark decode throughput at target batch size")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:30000",
                        help="SGLang server URL")
    parser.add_argument("--num-requests", type=int, default=64,
                        help="Number of concurrent requests (= decode batch size)")
    parser.add_argument("--input-len", type=int, default=131072,
                        help="Input token length per request")
    parser.add_argument("--output-len", type=int, default=256,
                        help="Max output tokens per request")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
