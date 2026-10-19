"""Command line entry point for the import and query services."""

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a ContextForge service")
    parser.add_argument("service", choices=("import", "query"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    module = {
        "import": "app.import_process.api.file_import_service:app",
        "query": "app.query_process.api.query_service:app",
    }[args.service]
    port = args.port or (8000 if args.service == "import" else 8001)
    uvicorn.run(module, host=args.host, port=port)


if __name__ == "__main__":
    main()
