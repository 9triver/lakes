import tempfile
import unittest
import zipfile
from pathlib import Path

from lake_workbench.data.transfer import extract_zip, normalize_proxy


class DataTransferTests(unittest.TestCase):
    def test_normalize_proxy_accepts_host_port_and_url(self) -> None:
        self.assertEqual(normalize_proxy("127.0.0.1:7897"), "http://127.0.0.1:7897")
        self.assertEqual(normalize_proxy("socks5://127.0.0.1:7897"), "socks5://127.0.0.1:7897")
        self.assertEqual(normalize_proxy(""), "")

    def test_extract_zip_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "archive.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../outside.txt", "unsafe")

            with self.assertRaises(ValueError):
                extract_zip(archive_path, root / "output")

