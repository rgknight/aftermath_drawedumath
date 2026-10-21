"""Serve the question explorer on localhost so student-work images show inline.

The published artifact can't load images from other sites; a local page can, straight from
the dataset's public S3 URLs, so nothing is downloaded ahead of time.

Usage (from the repo root, after `uv run python code/build_subset.py`):
    uv run python code/explorer/serve.py            # then open http://127.0.0.1:8765
"""

import argparse
import logging
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE_PATH = Path(__file__).resolve().parent / "question_explorer.html"
DATA_PATH = REPO_ROOT / "output" / "subsets" / "explorer_data.json"
# Per-problem model answers; only digits allowed so no path can escape the answers folder.
ANSWER_PATH = re.compile(r"/answers/\d+\.json")

# The artifact host wraps the page in this skeleton; do the same locally.
PAGE_HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>
</head><body>
"""
PAGE_TAIL = "\n</body></html>\n"


class ExplorerHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            body = (PAGE_HEAD + PAGE_PATH.read_text(encoding="utf-8") + PAGE_TAIL).encode("utf-8")
            self._send(body, "text/html; charset=utf-8")
        elif path == "/explorer_data.json":
            self._send(DATA_PATH.read_bytes(), "application/json")
        elif ANSWER_PATH.fullmatch(path) and (DATA_PATH.parent / path.lstrip("/")).exists():
            self._send((DATA_PATH.parent / path.lstrip("/")).read_bytes(), "application/json")
        else:
            self.send_error(404)

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        logger.debug(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found; run code/build_subset.py first")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), ExplorerHandler)
    logger.info("Question explorer at http://127.0.0.1:%s (Ctrl+C to stop)", args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
