# 替换原来的 client.py 无感从 ollama 转换到 llm studio
import openai

_BASE_URL = "http://192.168.56.1:11434/v1"   # LLM Studio 默认地址
_cli = openai.OpenAI(api_key="sk-dummy", base_url=_BASE_URL)

def generate(*, model_name: str, system: str, prompt: str):
    """与原 ollama.client.generate 签名保持一致，返回 (str, None)"""
    stream = _cli.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        stream=True,
        max_tokens=2048
    )
    full = ""
    for chunk in stream:
        full += chunk.choices[0].delta.content or ""
    return full, None