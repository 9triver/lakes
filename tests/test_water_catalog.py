from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from shapely.geometry import box

from lake_workbench.water_catalog import WaterCatalogMixin


class WaterCatalogStub(WaterCatalogMixin):
    def __init__(self, root: Path) -> None:
        self.region = SimpleNamespace(
            esa_polygon_dir=root / "esa_polygons",
            jrc_polygon_dir=root / "jrc_polygons",
        )


def lake(**properties):
    return SimpleNamespace(
        object_id="lake_1",
        area_km2=300.0,
        geometry=box(100, 20, 101, 21),
        bbox=(100, 20, 101, 21),
        properties=properties,
    )


class WaterCatalogTests(unittest.TestCase):
    def test_osm_layer_requires_osm_source_and_polygon(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = WaterCatalogStub(Path(directory))
            self.assertIsNotNone(catalog._osm_layer(lake(source_primary="osm", has_osm_polygon=True)))
            self.assertIsNone(catalog._osm_layer(lake(source_primary="local", has_osm_polygon=True)))
            self.assertIsNone(catalog._osm_layer(lake(source_primary="osm", has_osm_polygon=False)))

    def test_large_lakes_require_precomputed_esa_and_jrc_layers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = WaterCatalogStub(Path(directory))
            target = lake(source_primary="osm")
            esa = catalog._esa_smoothed_layer(target)
            jrc = catalog._jrc_occurrence_layer(target, threshold=120)
            self.assertTrue(esa["properties"]["skipped"])
            self.assertTrue(jrc["properties"]["skipped"])
            self.assertEqual(jrc["properties"]["threshold"], 100)


if __name__ == "__main__":
    unittest.main()
