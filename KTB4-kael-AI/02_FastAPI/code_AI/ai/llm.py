import httpx

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma3:1b"


def ask(prompt: str) -> str:
    response = httpx.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]
