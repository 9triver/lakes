"""React frontend and static asset routes."""

from pathlib import Path

from lake_workbench.utils import is_frontend_route


def handle_frontend_get(handler, path: str, static_dir: Path) -> bool:
    if is_frontend_route(path):
        handler._serve_file(static_dir / "dist" / "index.html")
    elif path.startswith("/static/"):
        handler._serve_file(static_dir / path.removeprefix("/static/"))
    else:
        return False
    return True
