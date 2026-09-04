"""Network and archive transfer helpers used by preparation scripts."""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from lake_workbench.utils import display_path


def download_file(
    url: str, path: Path, force: bool = False, timeout: int = 120, proxy: str = ""
) -> None:
    """Download a file atomically, preferring curl when available."""
    if path.exists() and path.stat().st_size > 0 and not force:
        print(f"exists {display_path(path)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_suffix(path.suffix + ".part")
    part_path.unlink(missing_ok=True)
    print(f"download {url}")
    if download_with_curl(url, path, proxy):
        print(f"wrote {display_path(path)}")
        return
    request = Request(url, headers={"User-Agent": "lakes-prepare-data/0.1"})
    try:
        total = 0
        with (
            urlopen(request, timeout=timeout) as response,
            part_path.open("wb") as handle,
        ):
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                total += len(chunk)
                if total % (25 * 1024 * 1024) < len(chunk):
                    print(f"downloaded {total / 1024 / 1024:.1f} MiB", flush=True)
    except (HTTPError, URLError) as exc:
        part_path.unlink(missing_ok=True)
        raise RuntimeError(f"failed to download {url}: {exc}") from exc
    part_path.replace(path)
    print(f"wrote {display_path(path)}")


def download_with_curl(url: str, path: Path, proxy: str = "") -> bool:
    """Try curl and return false so callers can fall back to urllib."""
    if shutil.which("curl") is None:
        return False
    part_path = path.with_suffix(path.suffix + ".part")
    command = [
        "curl",
        "-L",
        "--fail",
        "--connect-timeout",
        "30",
        "--retry",
        "3",
        "--retry-delay",
        "2",
        "-o",
        str(part_path),
        url,
    ]
    command[1:1] = ["--proxy", normalize_proxy(proxy)] if proxy else ["--noproxy", "*"]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        part_path.unlink(missing_ok=True)
        return False
    part_path.replace(path)
    return True


def normalize_proxy(proxy: str) -> str:
    proxy = proxy.strip()
    if proxy and "://" not in proxy:
        return f"http://{proxy}"
    return proxy


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    """Extract a trusted archive without allowing path traversal."""
    destination = out_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        members = []
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(
                    f"archive member escapes destination: {member.filename}"
                )
            members.append(member)
        archive.extractall(destination, members=members)
