#!/usr/bin/env python3
"""
Entry point – start the Police Scanner web server.

Usage:
    python run.py

Open a browser to  http://localhost:8000
"""
import logging
import sys
from pathlib import Path

# Make sure the project root is on sys.path when run directly.
sys.path.insert(0, str(Path(__file__).parent))

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(name)-25s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn is not installed.  Run:  pip install -r requirements.txt")
        sys.exit(1)

    print()
    print("  Police Scanner")
    print(f"  http://localhost:{config.PORT}")
    print()
    print("  Press Ctrl+C to stop")
    print()

    uvicorn.run(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        log_level="info",
        # Use lifespan events for clean startup/shutdown.
        lifespan="on",
    )
