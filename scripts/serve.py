"""Start the live dashboard:  python scripts/serve.py  ->  http://localhost:8000"""
import os
import sys

import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
port = int(os.environ.get("PORT", "8000"))
print(f"PXP Blackboard dashboard -> http://127.0.0.1:{port}   (Ctrl+C to stop)")
uvicorn.run("pxp_blackboard.server.app:app", host="127.0.0.1", port=port)
