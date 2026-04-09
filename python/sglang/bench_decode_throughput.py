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

    token_timestamps = []
    first_token_time = None
    num_output_tokens = 0

    with requests.post(url, json=payload, stream=True) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[len("data: "):])
            if "text" in data:
                t = time.perf_counter()
                num_output_tokens += 1
                token_timestamps.append(t)
                if first_token_time is None:
                    first_token_time = t

    return {
        "req_id": req_id,
        "first_token_time": first_token_time,
        "token_timestamps": token_timestamps,
        "num_output_tokens": num_output_tokens,
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

    # Count tokens generated AFTER all prefills are done (pure decode phase)
    decode_tokens = 0
    decode_end_time = 0.0
    for r in valid_results:
        for ts in r["token_timestamps"]:
            if ts > last_prefill_done:
                decode_tokens += 1
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
    print(f"  Decode batch size:         {args.num_requests}")
    if decode_duration > 0:
        throughput = decode_tokens / decode_duration
        per_req = throughput / args.num_requests
        print(f"  Throughput (total):        {throughput:.2f} tokens/s")
        print(f"  Throughput (per request):  {per_req:.2f} tokens/s")
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
