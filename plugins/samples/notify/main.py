"""Sample notify plugin.
Demonstrates webhook delivery with simple retry and optional Go offload for signing or heavy work.
"""
import time
import requests
from typing import Dict, Any


class NotifyPlugin:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.max_retries = self.config.get("max_retries", 3)

    def send_webhook(self, url: str, payload: Dict[str, Any]):
        attempt = 0
        while attempt < self.max_retries:
            try:
                resp = requests.post(url, json=payload, timeout=5)
                if resp.status_code < 500:
                    return resp.status_code, resp.text
            except requests.RequestException as e:
                print(f"[notify] request error: {e}")
            attempt += 1
            backoff = 2 ** attempt
            time.sleep(backoff)
        return None, None


if __name__ == "__main__":
    n = NotifyPlugin({"max_retries": 2})
    code, body = n.send_webhook("https://httpbin.org/status/200", {"test": "ok"})
    print("result:", code, body)
