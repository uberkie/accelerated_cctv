"""Sample ONVIF plugin entrypoint.
This file demonstrates the plugin shape expected by the repo: a `Plugin` class
with lifecycle hooks. Heavy compute work (if any) should be offloaded to `go_worker`.
"""
import time
import json
from typing import Dict, Any


class Plugin:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.poll_interval = self.config.get("poll_interval", 30)
        self.running = False

    def start(self):
        self.running = True
        print("[onvif] plugin started")

    def stop(self):
        self.running = False
        print("[onvif] plugin stopped")

    def discover(self):
        # Demo implementation: return a static list of discovered cameras.
        # Real implementation should query ONVIF devices on the network.
        devices = [
            {"id": "cam01", "rtsp": "rtsp://example.local/stream1", "model": "demo-cam"},
        ]
        print("[onvif] discovered devices:", json.dumps(devices))
        return devices

    def run(self):
        self.start()
        try:
            while self.running:
                self.discover()
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            self.stop()


if __name__ == "__main__":
    p = Plugin({"poll_interval": 5})
    p.run()
