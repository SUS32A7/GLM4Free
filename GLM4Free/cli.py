"""CLI entry points for GLM4Free."""

import sys
import argparse


def run_api():
    """Entry point for `glm4free-api` command."""
    parser = argparse.ArgumentParser(
        prog="glm4free-api",
        description="Start the GLM4Free REST API server",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("[!] uvicorn not found. Install it with:  pip install uvicorn")
        sys.exit(1)

    print(f"[*] Starting GLM4Free API on http://{args.host}:{args.port}")
    print(f"[*] Interactive docs →  http://localhost:{args.port}/docs\n")

    uvicorn.run(
        "glm4free.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def run_chat():
    """Entry point for `glm4free` command (original CLI chat)."""
    from glm4free.client import main
    main()


if __name__ == "__main__":
    run_api()
