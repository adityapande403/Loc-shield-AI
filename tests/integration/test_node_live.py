"""Integration test verifying Node.js SDK against live running FastAPI server."""

import subprocess
import sys
import time
import requests
import pytest
import uvicorn
import threading
from localshield.api.app import create_app


def run_server(port=8000):
    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def test_node_integration_live():
    # Start FastAPI server in a daemon thread
    server_thread = threading.Thread(target=run_server, args=(8000,), daemon=True)
    server_thread.start()

    # Wait for server to come up
    ready = False
    for _ in range(30):
        try:
            r = requests.get("http://127.0.0.1:8000/health", timeout=1)
            if r.status_code == 200:
                ready = True
                break
        except Exception:
            time.sleep(0.2)

    assert ready, "Server failed to start within timeout"

    # Execute Node.js integration script
    res = subprocess.run(["node", "sdk_node/test_live_service.js"], capture_output=True, text=True)
    print("Node output:\n", res.stdout)
    if res.returncode != 0:
        print("Node error:\n", res.stderr)

    assert res.returncode == 0, f"Node.js integration failed: {res.stderr}"
