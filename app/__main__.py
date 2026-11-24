"""Start the app:  .venv/bin/python -m app [--port 8765] [--host 127.0.0.1]"""
import argparse
import os

import uvicorn

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    uvicorn.run("app.server:app", host=a.host, port=a.port, log_level="warning")
