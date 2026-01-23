#!/bin/bash
# Complete workflow for testing Qwen2.5-VL-7B with EAGLE3

set -e

echo "=========================================="
echo "Qwen2.5-VL-7B + EAGLE3 Testing Workflow"
echo "=========================================="

# Configuration - MODIFY THESE PATHS
EAGLE_MODEL_PATH="/path/to/your/eagle/model"  # Your EAGLE model directory
TRAINING_DATA="/nlp_group/chenjiapeng/ykx_keye8b_rl/SpecForge/cache/training_data.jsonl"  # Your JSONL file
PORT=30000

# Step 1: Start SGLang server with EAGLE3
echo ""
echo "Step 1: Starting SGLang server with EAGLE3..."
echo "----------------------------------------"

# Set environment variable to allow context length override
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# Launch server (this runs in background)
nohup python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path "${EAGLE_MODEL_PATH}" \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 8 \
  --speculative-num-draft-tokens 32 \
  --mem-fraction 0.7 \
  --cuda-graph-max-bs 8 \
  --dtype bfloat16 \
  --port ${PORT} \
  > sglang_server.log 2>&1 &

SERVER_PID=$!
echo "✓ Server started (PID: ${SERVER_PID})"
echo "  Log file: sglang_server.log"

# Step 2: Wait for server to be ready
echo ""
echo "Step 2: Waiting for server to be ready..."
echo "----------------------------------------"

max_wait=300  # 5 minutes
waited=0
while ! curl -s http://localhost:${PORT}/v1/models > /dev/null 2>&1; do
    if [ $waited -ge $max_wait ]; then
        echo "❌ Server failed to start within ${max_wait} seconds"
        echo "Check sglang_server.log for errors"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
    echo "  Waiting... (${waited}s / ${max_wait}s)"
    sleep 5
    waited=$((waited + 5))
done

echo "✓ Server is ready!"

# Step 3: Run benchmark tests
echo ""
echo "Step 3: Running benchmark tests..."
echo "----------------------------------------"

python test_qwen2_5_vl_eagle.py \
  --url http://localhost:${PORT} \
  --dataset_path "${TRAINING_DATA}" \
  --batch_size 7 \
  --max_concurrency 4 \
  --temperature 0.7 \
  --top_p 0.9 \
  --max_new_tokens 100 \
  --verbose

# Step 4: Cleanup
echo ""
echo "Step 4: Cleanup (press Ctrl+C to keep server running)..."
echo "----------------------------------------"
read -p "Press Enter to shutdown server, or Ctrl+C to keep it running..."

kill $SERVER_PID 2>/dev/null || true
echo "✓ Server shutdown complete"

echo ""
echo "=========================================="
echo "Testing Complete!"
echo "=========================================="
