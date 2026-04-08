"""
OpenEnv Server entry point for T1D Environment
Imports the FastAPI app from the root server module and provides a main() for the script entry point.
"""

import sys
import os
import uvicorn

# Ensure the project root is on the path so server.py and t1d_env can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app  # noqa: E402


def main():
    """Entry point for `server` console script."""
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")


if __name__ == "__main__":
    main()
