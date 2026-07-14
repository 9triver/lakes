from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from shapely.geometry import box

from lake_workbench.water.annotations import WaterAnnotationsMixin
from lake_workbench.water.providers import ANNOTATION_PROVIDERS


class Candidate(dict):
    def __init__(self, geometry, **values) -> None:
        super().__init__(values)
        self.geometry = geometry


class WaterAnnotationsStub(WaterAnnotationsMixin):
    def __init__(self, root: Path) -> None:
        self.region = SimpleNamespace(
            esa_polygon_dir=root / "esa_polygons",
            jrc_polygon_dir=root / "jrc_polygons",
        )
        self.candidates = {}
        self.local_annotations = {}

    def suggested_water_candidate(self, target, source):
        return self.candidates.get(source)

    def local_label_geojson(self, target, label_id):
        payload = self.local_annotations.get(label_id)
        if payload is None:
            raise FileNotFoundError(label_id)
        return payload


def site(**properties):
    return SimpleNamespace(
        site_id="site_1",
        area_km2=300.0,
        geometry=box(100, 20, 101, 21),
        bbox=(100, 20, 101, 21),
        properties=properties,
    )


class WaterAnnotationsTests(unittest.TestCase):
    def test_all_annotation_sources_share_one_result_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = WaterAnnotationsStub(Path(directory))
            catalog.candidates["osm"] = Candidate(
                box(100, 20, 101, 21),
                candidate_id="osm_1",
                site_id="site_1",
                properties_json='{"name": "Test water"}',
                intersection_area_km2=4.2,
                site_coverage_ratio=0.8,
            )

            result = catalog.annotation_for_site(site(), "osm")

            self.assertEqual(set(ANNOTATION_PROVIDERS), {"osm", "hydrolakes", "local", "esa", "jrc"})
            self.assertEqual(result["site_id"], "site_1")
            self.assertEqual(result["source"], "osm")
            self.assertEqual(result["status"], "available")
            self.assertEqual(result["parameters"], {})
            self.assertEqual(result["annotation"]["properties"]["name"], "Test water")

            missing = catalog.annotation_for_site(site(), "hydrolakes")
            self.assertEqual(missing["status"], "missing")
            self.assertIsNone(missing["annotation"])

    def test_local_annotation_uses_label_id_and_feature_collection_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = WaterAnnotationsStub(Path(directory))
            catalog.local_annotations["label_1"] = {
                "label": {"id": "label_1", "name": "water.shp"},
                "geojson": {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {"type": "Polygon", "coordinates": []},
                            "properties": {"class": "water"},
                        }
                    ],
                },
            }

            result = catalog.annotation_for_site(site(), "local", {"label_id": "label_1"})

            self.assertEqual(result["status"], "available")
            self.assertEqual(result["parameters"], {"label_id": "label_1"})
            self.assertEqual(result["annotation"]["type"], "FeatureCollection")
            self.assertEqual(result["annotation"]["properties"]["label"]["name"], "water.shp")

            missing = catalog.annotation_for_site(site(), "local", {"label_id": "missing"})
            self.assertEqual(missing["status"], "missing")

            catalog.local_annotations["empty"] = {
                "label": {"id": "empty", "name": "empty.shp"},
                "geojson": {"type": "FeatureCollection", "features": []},
            }
            empty = catalog.annotation_for_site(site(), "local", {"label_id": "empty"})
            self.assertEqual(empty["status"], "empty")

            with self.assertRaisesRegex(ValueError, "label_id is required"):
                catalog.annotation_for_site(site(), "local")

    def test_large_sites_require_precomputed_esa_and_jrc_layers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = WaterAnnotationsStub(Path(directory))
            target = site(source_primary="osm")
            esa = catalog.annotation_for_site(target, "esa")
            jrc = catalog.annotation_for_site(target, "jrc", {"threshold": 120})
            self.assertEqual(esa["status"], "skipped")
            self.assertEqual(jrc["status"], "skipped")
            self.assertTrue(esa["annotation"]["properties"]["skipped"])
            self.assertTrue(jrc["annotation"]["properties"]["skipped"])
            self.assertEqual(jrc["parameters"]["threshold"], 100)
            self.assertEqual(jrc["annotation"]["properties"]["threshold"], 100)

    def test_unknown_annotation_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = WaterAnnotationsStub(Path(directory))
            with self.assertRaisesRegex(ValueError, "unknown annotation source"):
                catalog.annotation_for_site(site(), "unknown")


if __name__ == "__main__":
    unittest.main()
