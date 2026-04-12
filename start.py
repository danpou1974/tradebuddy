"""
start.py
========
Launches both Streamlit dashboard and FastAPI API server.
Streamlit runs on $PORT (required by Render).
FastAPI runs on $PORT + 1 for the mobile app API.

For Render: use two services or a single service with FastAPI as primary.
"""

import os
import subprocess
import sys

PORT = int(os.environ.get("PORT", 8000))

def main():
    # Run FastAPI as the main web process (Render needs one process on $PORT)
    os.execvp(
        sys.executable,
        [
            sys.executable, "-m", "uvicorn",
            "api_server:app",
            "--host", "0.0.0.0",
            "--port", str(PORT),
        ],
    )

if __name__ == "__main__":
    main()
