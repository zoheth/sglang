import base64
import requests

BASE_URL = "http://localhost:30000"
PROFILE_OUTPUT_DIR = "/tmp/sglang_profile"

model_name = "Qwen/Qwen3-VL-2B-Instruct"
# model_name = "Qwen/Qwen2.5-VL-3B-Instruct"


# 本地视频路径
video_path = "/home/devuser/models/../downloads/jobs_presenting_ipod.mp4"
# 如果挂载了 downloads 目录，用容器内的路径；否则用 base64

def start_profile():
    resp = requests.post(
        f"{BASE_URL}/start_profile",
        json={"output_dir": PROFILE_OUTPUT_DIR, "activities": ["CPU", "GPU"]},
    )
    print(f"start_profile: {resp.status_code}")


def stop_profile():
    resp = requests.post(f"{BASE_URL}/stop_profile")
    print(f"stop_profile: {resp.status_code}")
    print(f"trace 文件: {PROFILE_OUTPUT_DIR}/*.json.gz -> https://ui.perfetto.dev/")


def test_video_base64(video_path: str):
    with open(video_path, "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode("utf-8")

    url = f"{BASE_URL}/v1/chat/completions"
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video."},
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:video/mp4;base64,{video_base64}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": 20,
    }

    response = requests.post(url, json=data)
    result = response.json()

    if "choices" in result:
        print("=== 模型回复 ===")
        print(result["choices"][0]["message"]["content"])
    else:
        print("=== 错误 ===")
        print(result)



if __name__ == "__main__":
    import sys

    # 默认视频路径（宿主机路径，运行时需要改成容器内路径）
    video = sys.argv[1] if len(sys.argv) > 1 else "/home/x/downloads/jobs_presenting_ipod.mp4"

    print(f"测试视频: {video}")
    print()

    # start_profile()
    test_video_base64(video)
    # stop_profile()

"""
export VIDEO_MAX_PIXELS=3000000
export SGLANG_VLM_CACHE_SIZE_MB=200
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-VL-2B-Instruct \
    --attention-backend flashinfer \
    --mem-fraction-static 0.6 \
    --max-total-tokens 8192 \
    --max-running-requests 1 \
    --chunked-prefill-size 2048 \
    --disable-cuda-graph
"""