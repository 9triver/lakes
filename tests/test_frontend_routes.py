from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from lake_workbench.routes.frontend import handle_frontend_get


class FrontendRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.static_dir = Path("/application/static")
        self.handler = SimpleNamespace(served_path=None)
        self.handler._serve_file = lambda path: setattr(self.handler, "served_path", path)

    def test_react_routes_serve_root_index(self) -> None:
        self.assertTrue(handle_frontend_get(self.handler, "/regions/gansu/sites", self.static_dir))
        self.assertEqual(self.handler.served_path, self.static_dir / "index.html")

    def test_root_assets_are_served_from_static_assets(self) -> None:
        self.assertTrue(handle_frontend_get(self.handler, "/assets/index.js", self.static_dir))
        self.assertEqual(self.handler.served_path, self.static_dir / "assets" / "index.js")

    def test_service_worker_is_served_from_static_root(self) -> None:
        self.assertTrue(handle_frontend_get(self.handler, "/sw.js", self.static_dir))
        self.assertEqual(self.handler.served_path, self.static_dir / "sw.js")

    def test_old_static_dist_url_is_not_a_frontend_asset_route(self) -> None:
        self.assertFalse(handle_frontend_get(self.handler, "/static/dist/index.html", self.static_dir))


if __name__ == "__main__":
    unittest.main()
