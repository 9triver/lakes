from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import requests
import rasterio
from affine import Affine
from PIL import Image
from rasterio.transform import from_bounds

from lake_workbench.automatic_labels.basemap import (
    DEFAULT_OSM_PROXY,
    DEFAULT_OSM_TILE_URL,
    _cached_tile,
    bounded_tile_range,
)
from lake_workbench.automatic_labels.service import (
    OSM_SPECTRAL_CONSENSUS,
    SPECTRAL_OSM_CONSENSUS,
    SPECTRAL_OSM_INTERSECTION,
    SPECTRAL_WATER,
    generate_derived_label,
)
from lake_workbench.automatic_labels.spectral import (
    ExternalQualityMasks,
    IGNORE_LABEL,
    WATER_LABEL,
    _normalized_difference,
    classify_spectral_array,
    normalize_reflectance,
    read_spectral_evidence,
    spectral_band_indexes,
    spectral_labels,
    spectral_processing_level,
)


class AutomaticLabelTests(unittest.TestCase):
    def test_osm_defaults_use_direct_osm_de_tiles(self) -> None:
        self.assertEqual(
            DEFAULT_OSM_TILE_URL,
            "https://tile.openstreetmap.de/{z}/{x}/{y}.png",
        )
        self.assertEqual(DEFAULT_OSM_PROXY, "")

    def test_cached_osm_tile_is_read_before_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_path = root / "basemap" / "osm_test" / "11" / "1610" / "815.png"
            cache_path.parent.mkdir(parents=True)
            image = Image.new("RGB", (256, 256), (170, 211, 223))
            image.save(cache_path, format="PNG")

            class UnexpectedNetwork:
                def get(self, *_args, **_kwargs):
                    raise AssertionError("a valid cached tile must avoid the network")

            self.assertEqual(
                _cached_tile(
                    UnexpectedNetwork(),
                    "https://tile.openstreetmap.de/{z}/{x}/{y}.png",
                    root,
                    11,
                    1610,
                    815,
                    "osm_test",
                    "png",
                ),
                cache_path.read_bytes(),
            )

    def test_cached_osm_tile_retries_connection_abort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB", (256, 256), (170, 211, 223))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            payload = buffer.getvalue()

            class Response:
                status_code = 200
                content = payload

                def raise_for_status(self):
                    return None

            class FlakyNetwork:
                attempts = 0

                def get(self, *_args, **_kwargs):
                    self.attempts += 1
                    if self.attempts < 3:
                        raise requests.ConnectionError("Connection aborted.")
                    return Response()

            network = FlakyNetwork()
            with patch("lake_workbench.automatic_labels.basemap.time.sleep"):
                result = _cached_tile(
                    network,
                    DEFAULT_OSM_TILE_URL,
                    root,
                    11,
                    1610,
                    815,
                    "osm_test",
                    "png",
                )
            self.assertEqual(result, payload)
            self.assertEqual(network.attempts, 3)

    def test_missing_osm_tile_is_treated_as_uncovered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            class MissingTile:
                status_code = 404
                content = b""

                def raise_for_status(self):
                    raise requests.HTTPError("404")

            class Network:
                def get(self, *_args, **_kwargs):
                    return MissingTile()

            self.assertIsNone(
                _cached_tile(
                    Network(),
                    DEFAULT_OSM_TILE_URL,
                    Path(directory),
                    11,
                    1610,
                    815,
                    "osm_test",
                    "png",
                )
            )

    def test_generated_sources_are_persisted_as_independent_layers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.tif"
            image_path.write_bytes(b"test")
            catalog = SimpleNamespace(
                region=SimpleNamespace(key="gansu", cache_dir=root / "cache"),
                selected_training_imagery=lambda _site, _view: [
                    {"tci_path": str(image_path), "product_name": "image.tif"}
                ],
            )
            site = SimpleNamespace(site_id="gansu_17407", bbox=(100, 20, 104, 24))
            spectral = SimpleNamespace(
                water_score=np.array(
                    [
                        [0.9, 0.9, 0.0, 0.0],
                        [0.9, 0.9, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    dtype="float32",
                ),
                labels=np.array(
                    [
                        [1, 1, 0, 0],
                        [1, 1, 0, 0],
                        [255, 255, 0, 0],
                        [255, 255, 0, 0],
                    ],
                    dtype="uint8",
                ),
                valid=np.ones((4, 4), dtype=bool),
                transform=Affine(1, 0, 100, 0, -1, 24),
                crs="EPSG:4326",
                shape=(4, 4),
                bands={"green": 2, "nir": 4},
                diagnostics={"ndwi_mean": 0.5},
            )
            with (
                patch(
                    "lake_workbench.automatic_labels.service.read_spectral_evidence",
                    return_value=spectral,
                ),
            ):
                spectral_result = generate_derived_label(
                    catalog,
                    site,
                    SPECTRAL_WATER,
                    {"extent": [0, 0, 200, 200]},
                    root / "labels",
                )

            self.assertEqual(spectral_result["source"], SPECTRAL_WATER)
            path = root / "labels" / SPECTRAL_WATER / f"{spectral_result['label_id']}.geojson"
            self.assertTrue(path.exists())
            self.assertEqual(
                {
                    feature["properties"]["label_value"]
                    for feature in spectral_result["label"]["features"]
                },
                {WATER_LABEL},
            )
            mask_path = path.with_suffix(".npz")
            self.assertTrue(mask_path.exists())
            with np.load(mask_path, allow_pickle=False) as mask:
                np.testing.assert_array_equal(mask["labels"], spectral.labels)
                np.testing.assert_array_equal(
                    mask["valid"], spectral.valid.astype("uint8")
                )
                np.testing.assert_array_equal(
                    mask["transform"], np.asarray(tuple(spectral.transform)[:6])
                )
                self.assertEqual(str(mask["crs"].item()), "EPSG:4326")
            self.assertEqual(
                spectral_result["label"]["properties"]["raster_label"]["path"],
                mask_path.name,
            )
            self.assertEqual(spectral_result["stats"]["water_pixels"], 4)
            self.assertEqual(spectral_result["stats"]["ignore_pixels"], 4)
            self.assertEqual(spectral_result["stats"]["ignore_polygon_count"], 0)
            self.assertEqual(spectral_result["label"]["properties"]["extent"], [100.0, 20.0, 104.0, 24.0])

    def test_xyz_range_reduces_zoom_to_respect_tile_limit(self) -> None:
        zoom, tiles = bounded_tile_range((100, 20, 110, 30), 15, max_tiles=16)
        self.assertLess(zoom, 15)
        self.assertLessEqual(len(tiles), 16)

    def test_spectral_osm_consensus_promotes_connected_spectral_water(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.tif"
            image_path.write_bytes(b"test")
            catalog = SimpleNamespace(
                region=SimpleNamespace(key="yunnan", cache_dir=root / "cache"),
                selected_training_imagery=lambda _site, _view: [
                    {"tci_path": str(image_path), "product_name": "image.tif"}
                ],
            )
            site = SimpleNamespace(site_id="yunnan_17567", bbox=(98, 28, 99, 29))
            spectral = SimpleNamespace(
                water_score=np.ones((4, 4), dtype="float32"),
                labels=np.array(
                    [
                        [1, 1, 0, 0],
                        [1, 1, 0, 0],
                        [1, 1, 0, 0],
                        [255, 255, 0, 1],
                    ],
                    dtype="uint8",
                ),
                valid=np.ones((4, 4), dtype=bool),
                transform=Affine(1, 0, 98, 0, -1, 29),
                crs="EPSG:4326",
                shape=(4, 4),
                bands={"blue": 1, "green": 2, "red": 3, "nir": 4, "swir1": 5},
                diagnostics={"ndwi_mean": 0.1},
            )
            osm_rgb = np.full((4, 4, 3), 245, dtype="uint8")
            osm_rgb[:2, :2] = (170, 211, 223)
            osm = SimpleNamespace(
                rgb=osm_rgb,
                valid=np.ones((4, 4), dtype=bool),
                provider="osm_standard",
                zoom=14,
                tile_count=4,
                tile_url="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            )
            with (
                patch(
                    "lake_workbench.automatic_labels.service.read_spectral_evidence",
                    return_value=spectral,
                ),
                patch(
                    "lake_workbench.automatic_labels.service.aligned_osm_rgb",
                    side_effect=[
                        osm,
                        osm,
                        SimpleNamespace(
                            rgb=np.concatenate(
                                [
                                    osm_rgb[:2],
                                    np.array(
                                        [[
                                            (245, 245, 245),
                                            (245, 245, 245),
                                            (170, 211, 223),
                                            (245, 245, 245),
                                        ]],
                                        dtype="uint8",
                                    ),
                                    osm_rgb[3:],
                                ],
                                axis=0,
                            ),
                            valid=np.ones((4, 4), dtype=bool),
                            provider="osm_standard",
                            zoom=14,
                            tile_count=4,
                            tile_url="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                        ),
                    ],
                ),
            ):
                intersection_result = generate_derived_label(
                    catalog,
                    site,
                    SPECTRAL_OSM_INTERSECTION,
                    {"extent": [98, 28, 99, 29]},
                    root / "labels",
                )
                result = generate_derived_label(
                    catalog,
                    site,
                    SPECTRAL_OSM_CONSENSUS,
                    {"extent": [98, 28, 99, 29]},
                    root / "labels",
                )
                reverse_result = generate_derived_label(
                    catalog,
                    site,
                    OSM_SPECTRAL_CONSENSUS,
                    {"extent": [98, 28, 99, 29]},
                    root / "labels",
                )

            self.assertEqual(intersection_result["source"], SPECTRAL_OSM_INTERSECTION)
            self.assertEqual(intersection_result["stats"]["water_pixels"], 4)
            self.assertEqual(intersection_result["stats"]["background_pixels"], 7)
            self.assertEqual(intersection_result["stats"]["ignore_pixels"], 5)
            self.assertEqual(
                intersection_result["details"]["processing_mode"],
                "spectral_osm_intersection",
            )
            self.assertEqual(result["source"], SPECTRAL_OSM_CONSENSUS)
            self.assertEqual(result["stats"]["water_pixels"], 6)
            self.assertEqual(result["stats"]["ignore_pixels"], 3)
            self.assertEqual(result["details"]["spectral_only_pixels"], 3)
            self.assertEqual(result["details"]["consensus_seed_pixels"], 4)
            self.assertEqual(result["details"]["spectral_only_promoted_pixels"], 2)
            self.assertEqual(result["details"]["spectral_only_ignored_pixels"], 1)
            self.assertTrue(
                all(
                    feature["properties"]["label_value"] == WATER_LABEL
                    for feature in result["label"]["features"]
                )
            )
            self.assertEqual(reverse_result["source"], OSM_SPECTRAL_CONSENSUS)
            self.assertEqual(reverse_result["stats"]["water_pixels"], 5)
            self.assertEqual(reverse_result["stats"]["ignore_pixels"], 5)
            self.assertEqual(reverse_result["details"]["osm_only_pixels"], 1)
            self.assertEqual(
                reverse_result["details"]["osm_only_promoted_pixels"], 1
            )
            self.assertEqual(
                reverse_result["details"]["processing_mode"],
                "osm_water_connected_to_spectral_seed",
            )

    def test_spectral_reader_uses_named_bands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.tif"
            data = np.zeros((5, 4, 4), dtype="float32")
            data[0] = 50
            data[1] = 100
            data[2] = 30
            data[3] = 20
            data[4] = 10
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=4,
                height=4,
                count=5,
                dtype="float32",
                crs="EPSG:4326",
                transform=from_bounds(100, 20, 104, 24, 4, 4),
            ) as dataset:
                dataset.write(data)
                for index, description in enumerate(
                    ("rhot_492", "rhot_560", "rhot_665", "rhot_833", "rhot_1614"),
                    start=1,
                ):
                    dataset.set_band_description(index, description)

            with rasterio.open(path) as dataset:
                self.assertEqual(
                    spectral_band_indexes(dataset),
                    {"blue": 1, "green": 2, "red": 3, "nir": 4, "swir1": 5},
                )
            evidence = read_spectral_evidence(path, (100, 20, 104, 24))
            self.assertEqual(evidence.shape, (4, 4))
            self.assertTrue(np.all(evidence.valid))
            self.assertGreater(float(evidence.water_score.mean()), 0.8)

    def test_processing_level_recognizes_rhot_and_rhos(self) -> None:
        l1c = SimpleNamespace(
            descriptions=("rhot_492", "rhot_560"),
            tags=lambda: {},
            name="scene.tif",
        )
        l2a = SimpleNamespace(
            descriptions=("rhos_492", "rhos_560"),
            tags=lambda: {},
            name="scene.tif",
        )
        self.assertEqual(spectral_processing_level(l1c), "L1C")
        self.assertEqual(spectral_processing_level(l2a), "L2A")

    def test_reflectance_normalization_keeps_valid_negative_values(self) -> None:
        image = np.array(
            [
                [[10000.0, -99.0], [5000.0, 0.0]],
                [[9000.0, -50.0], [4000.0, 100.0]],
            ],
            dtype="float32",
        )
        valid = np.ones((2, 2), dtype=bool)
        reflectance, divisor, metadata_applied = normalize_reflectance(
            image, valid
        )
        self.assertEqual(divisor, 10000.0)
        self.assertFalse(metadata_applied)
        self.assertAlmostEqual(float(reflectance[0, 0, 1]), -0.0099, places=6)

    def test_normalized_difference_is_bounded_with_negative_reflectance(self) -> None:
        first = np.array([[0.01, 0.01, -0.02]], dtype="float32")
        second = np.array([[-0.0099, 0.02, 0.0199]], dtype="float32")

        result = _normalized_difference(first, second)

        np.testing.assert_allclose(result, [[1.0, -1.0 / 3.0, 1.0]])

    def test_scene_adaptive_classification_is_reproducible(self) -> None:
        rng = np.random.default_rng(7)
        height, width = 80, 80
        water = np.zeros((height, width), dtype=bool)
        water[:, :24] = True
        values = {
            "blue": (800, 1200),
            "green": (900, 1400),
            "red": (600, 1600),
            "nir": (200, 3200),
            "swir1": (100, 2800),
        }
        image = np.empty((5, height, width), dtype="float32")
        for index, (water_value, land_value) in enumerate(values.values()):
            image[index] = np.where(water, water_value, land_value)
            image[index] += rng.normal(0, 25, size=(height, width))
        valid = np.ones((height, width), dtype=bool)
        bands = {name: index for index, name in enumerate(values, start=1)}

        first = classify_spectral_array(image, valid, bands, random_seed=19)
        second = classify_spectral_array(image, valid, bands, random_seed=19)

        np.testing.assert_array_equal(first.labels, second.labels)
        np.testing.assert_allclose(first.water_score, second.water_score)
        self.assertEqual(first.diagnostics["cluster_status"], "ready")
        self.assertGreater(float(np.mean(first.labels[water] == WATER_LABEL)), 0.95)
        self.assertLess(float(np.mean(first.labels[~water] == WATER_LABEL)), 0.01)

    def test_cloud_and_snow_are_ignored(self) -> None:
        height, width = 20, 20
        image = np.full((5, height, width), 1200.0, dtype="float32")
        image[2:] = 2500.0
        cloud = np.zeros((height, width), dtype=bool)
        cloud[:5, :] = True
        image[:, cloud] = 5000.0
        snow = np.zeros((height, width), dtype=bool)
        snow[5:10, :] = True
        image[0, snow] = 3000.0
        image[1, snow] = 6000.0
        image[2, snow] = 3000.0
        image[3, snow] = 3000.0
        image[4, snow] = 1000.0
        result = classify_spectral_array(
            image,
            np.ones((height, width), dtype=bool),
            {"blue": 1, "green": 2, "red": 3, "nir": 4, "swir1": 5},
        )
        self.assertTrue(np.all(result.labels[cloud] == IGNORE_LABEL))
        self.assertTrue(np.all(result.labels[snow] == IGNORE_LABEL))
        self.assertEqual(result.diagnostics["cloud_pixels"], int(cloud.sum()))
        self.assertEqual(result.diagnostics["snow_pixels"], int(snow.sum()))

    def test_external_quality_masks_ignore_cloud_snow_and_shadow(self) -> None:
        height, width = 8, 8
        image = np.full((5, height, width), 1200.0, dtype="float32")
        image[1] = 1800.0
        image[3] = 300.0
        image[4] = 150.0
        cloud = np.zeros((height, width), dtype=bool)
        cloud[:2, :] = True
        snow = np.zeros((height, width), dtype=bool)
        snow[2:4, :] = True
        shadow = np.zeros((height, width), dtype=bool)
        shadow[4:6, :] = True
        result = classify_spectral_array(
            image,
            np.ones((height, width), dtype=bool),
            {"blue": 1, "green": 2, "red": 3, "nir": 4, "swir1": 5},
            external_quality=ExternalQualityMasks(
                invalid=cloud | snow | shadow,
                cloud=cloud,
                snow=snow,
                shadow=shadow,
                sources=("SCL:SCL_20m.tif",),
            ),
        )
        self.assertTrue(np.all(result.labels[cloud | snow | shadow] == IGNORE_LABEL))
        self.assertEqual(result.diagnostics["shadow_pixels"], int(shadow.sum()))
        self.assertEqual(result.diagnostics["quality_mask_sources"], ["SCL:SCL_20m.tif"])

    def test_reader_uses_only_named_bands_and_scl_quality_layer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "source.tif"
            safe_dir = root / "scene.SAFE"
            safe_dir.mkdir()
            data = np.zeros((6, 4, 4), dtype="float32")
            data[0] = 50
            data[1] = 100
            data[2] = 30
            data[3] = 20
            data[4] = 10
            data[5] = -9999
            transform = from_bounds(100, 20, 104, 24, 4, 4)
            with rasterio.open(
                image_path,
                "w",
                driver="GTiff",
                width=4,
                height=4,
                count=6,
                dtype="float32",
                crs="EPSG:4326",
                transform=transform,
                nodata=-9999,
            ) as dataset:
                dataset.write(data)
                for index, description in enumerate(
                    ("rhot_492", "rhot_560", "rhot_665", "rhot_833", "rhot_1614", "auxiliary"),
                    start=1,
                ):
                    dataset.set_band_description(index, description)
            scl = np.full((4, 4), 6, dtype="uint8")
            scl[0, :] = 8
            scl[1, :] = 11
            scl[2, :] = 3
            scl_path = safe_dir / "SCL_20m.tif"
            with rasterio.open(
                scl_path,
                "w",
                driver="GTiff",
                width=4,
                height=4,
                count=1,
                dtype="uint8",
                crs="EPSG:4326",
                transform=transform,
            ) as dataset:
                dataset.write(scl, 1)

            evidence = read_spectral_evidence(
                image_path,
                (100, 20, 104, 24),
                quality_path=safe_dir,
            )
            self.assertTrue(np.all(evidence.valid))
            self.assertTrue(np.all(evidence.quality_valid[3, :]))
            self.assertFalse(np.any(evidence.quality_valid[:3, :]))
            self.assertEqual(evidence.diagnostics["cloud_pixels"], 4)
            self.assertEqual(evidence.diagnostics["snow_pixels"], 4)
            self.assertEqual(evidence.diagnostics["shadow_pixels"], 4)
            self.assertEqual(evidence.diagnostics["quality_mask_sources"], ["SCL:SCL_20m.tif"])

    def test_method_disagreement_is_ignored(self) -> None:
        water_score = np.array([[0.8, 0.1, 0.8]], dtype="float32")
        valid = np.ones((1, 3), dtype=bool)
        labels = spectral_labels(
            water_score,
            {
                "one": np.array([[True, False, True]]),
                "two": np.array([[False, False, True]]),
                "three": np.array([[False, False, False]]),
            },
            valid,
            valid,
        )
        np.testing.assert_array_equal(
            labels, np.array([[IGNORE_LABEL, 0, WATER_LABEL]], dtype="uint8")
        )

    def test_otsu_and_dswx_consensus_survives_missing_cluster_vote(self) -> None:
        water_score = np.array([[0.54, 0.54, 0.2]], dtype="float32")
        valid = np.ones((1, 3), dtype=bool)
        labels = spectral_labels(
            water_score,
            {
                "waterdetect_cluster": np.array([[False, False, False]]),
                "bounded_mndwi_otsu": np.array([[True, True, False]]),
                "dswx_five_band_subset": np.array([[True, False, False]]),
            },
            valid,
            valid,
        )

        np.testing.assert_array_equal(
            labels,
            np.array([[WATER_LABEL, IGNORE_LABEL, 0]], dtype="uint8"),
        )


if __name__ == "__main__":
    unittest.main()
