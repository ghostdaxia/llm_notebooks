import os
import json
import requests
from typing import List, Dict, Optional, Callable, Tuple

# 1. 环境变量统一入口
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://192.168.56.1:11434")
LLMSTUDIO_HOST = os.environ.get("LLMSTUDIO_HOST", "http://192.168.56.1:11434/v1")
USE_LLMSTUDIO = os.environ.get("USE_LLMSTUDIO", "1").lower() in ("1", "true", "yes")

# 2. 内部工具：把 Ollama 风格参数 -> OpenAI messages
def _build_messages(prompt: str, system: Optional[str] = None) -> List[Dict[str, str]]:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return msgs

# 3. 核心：流式 generate -> LLM Studio 的 /v1/chat/completions
def generate(model_name: str,
             prompt: str,
             system: Optional[str] = None,
             template=None,
             context=None,
             options=None,
             callback: Optional[Callable] = None) -> Tuple[Optional[str], Optional[list]]:
    """
    与原版签名完全一致，但内部走 LLM Studio 的 OpenAI 接口。
    template/context/options 字段 LLM Studio 无法识别，本实现直接忽略。
    返回 (full_response, None)  # context 不再有意义
    """
    if not USE_LLMSTUDIO:
        # 降级到原生 Ollama（把旧代码 copy 过来即可，这里不重复）
        raise RuntimeError("原生 Ollama 降级逻辑未实现，请设置 USE_LLMSTUDIO=1")

    url = f"{LLMSTUDIO_HOST}/chat/completions"
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer llm-studio"}  # LLM Studio 不做校验，可随意
    payload = {
        "model": model_name,
        "messages": _build_messages(prompt, system),
        "stream": True,          # 必须流式，才能逐句回调
        "max_tokens": 4096
    }

    full_response = ""
    try:
        with requests.post(url, headers=headers, json=payload, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(delimiter=b'\n'):
                if not line:
                    continue
                # LLM Studio 返回 "data: {...}\n"
                line = line.decode().strip()
                if line.startswith("data:"):
                    chunk_str = line[5:].strip()
                    if chunk_str == "[DONE]":
                        break
                    chunk = json.loads(chunk_str)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    full_response += delta
                    if callback:
                        # 为了兼容旧回调格式，伪造一个 Ollama 风格 dict
                        callback({"response": delta, "done": False})
            if callback:
                callback({"response": "", "done": True})
            return full_response, None
    except Exception as e:
        print(f"[LLM Studio] 请求失败: {e}")
        return None, None

# 4. 其余接口 LLM Studio 不支持，占位防止调用崩溃
def create(*a, **k):
    print("[LLM Studio] /api/create 未实现，跳过")

def pull(*a, **k):
    print("[LLM Studio] /api/pull 未实现，跳过")

def push(*a, **k):
    print("[LLM Studio] /api/push 未实现，跳过")

def copy(*a, **k):
    print("[LLM Studio] /api/copy 未实现，跳过")

def delete(*a, **k):
    print("[LLM Studio] /api/delete 未实现，跳过")

def show(*a, **k):
    print("[LLM Studio] /api/show 未实现，跳过")
    return None

def heartbeat():
    try:
        r = requests.get(LLMSTUDIO_HOST.replace("/v1", "") + "/health")
        return "LLM Studio is running" if r.status_code == 200 else "LLM Studio health check failed"
    except Exception as e:
        return f"LLM Studio is not running: {e}"

# 5. list 接口：LLM Studio 没有 /api/tags，我们构造一个假列表
def list() -> list:
    """
    返回一个仅含当前已加载模型的假列表，方便旧代码遍历。
    """
    try:
        # 先查已加载模型
        r = requests.get(LLMSTUDIO_HOST.replace("/v1", "") + "/v1/models")
        r.raise_for_status()
        data = r.json()
        # 把 OpenAI 格式转成 Ollama 风格
        models = []
        for m in data.get("data", []):
            models.append({"name": m["id"], "size": 0, "digest": ""})
        return models
    except Exception as e:
        print(f"[LLM Studio] 无法枚举模型: {e}")
        return []

# 6. 保持旧模块名兼容，直接 import * 即可
__all__ = ["generate", "create", "pull", "push", "list", "copy", "delete", "show", "heartbeat"]