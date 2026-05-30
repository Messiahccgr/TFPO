import json
import os
import urllib.error
import urllib.request

BASE_URL = os.getenv("LLM_BASE_URL", "http://172.22.2.242:3010/v1")

MODEL_NAME = os.getenv("LLM_MODEL", "qwen3.5-397b-a17b")

API_KEY = os.getenv("LLM_API_KEY", "sk-mHbHbIE1xWN7khlheTnL6E7eTRmHR1aQpMamEnasIk1S7jEx")


def chat(prompt: str) -> str:
    url = f"{BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"连接失败: {exc}") from exc


if __name__ == "__main__":
    answer = chat("你好，请做一个简短的自我介绍。告诉我你最新的知识库到什么时候")
    print(answer)
