"""Lake-local Shapefile label discovery and GeoJSON conversion."""

import hashlib
import re
from pathlib import Path

import pyogrio
from shapely.geometry import mapping
from shapely.validation import make_valid

from lake_workbench.utils import display_path, jsonable, resolve_data_path


class LocalLabelCatalogMixin:
    def local_label_items(self, lake) -> dict:
        labels = []
        seen = set()
        for directory in self._local_imagery_dirs_for_lake(lake):
            for path in sorted(directory.glob("*.shp")):
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                labels.append(self._local_label_item(path))
        labels.sort(key=lambda item: (item.get("date") or "", item["name"]), reverse=True)
        return {"lake_id": lake.object_id, "items": labels}

    def local_label_geojson(self, lake, label_id: str) -> dict:
        labels = {item["id"]: item for item in self.local_label_items(lake)["items"]}
        item = labels.get(label_id)
        if item is None:
            raise FileNotFoundError(f"Local label not found: {label_id}")
        path = resolve_data_path(item["path"], self.region)
        if not path.exists():
            raise FileNotFoundError(f"Local label file not found: {display_path(path)}")
        data = pyogrio.read_dataframe(path)
        if data.empty:
            return {
                "lake_id": lake.object_id,
                "label": item,
                "geojson": {"type": "FeatureCollection", "features": []},
            }
        if data.crs is not None:
            data = data.to_crs("EPSG:4326")
        features = []
        for _, row in data.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            props = {key: jsonable(row.get(key)) for key in data.columns if key != "geometry"}
            props.update({"label_id": item["id"], "label_name": item["name"], "source": "local_label"})
            features.append({"type": "Feature", "geometry": mapping(make_valid(geom)), "properties": props})
        return {
            "lake_id": lake.object_id,
            "label": {**item, "feature_count": len(features)},
            "geojson": {"type": "FeatureCollection", "features": features},
        }

    def _local_imagery_dirs_for_lake(self, lake) -> list[Path]:
        directories = []
        seen = set()
        for rows in self.user_tci_rows.values():
            for row in rows:
                if row.get("source") != "local_img" or row.get("lake_id") != lake.object_id:
                    continue
                directory = row.get("safe_path")
                if not directory:
                    continue
                directory = Path(directory)
                if not directory.exists() or not directory.is_dir():
                    continue
                key = str(directory.resolve())
                if key in seen:
                    continue
                seen.add(key)
                directories.append(directory)
        return directories

    def _local_label_item(self, path: Path) -> dict:
        display = display_path(path)
        digest = hashlib.sha1(display.encode("utf-8")).hexdigest()[:12]
        date_match = re.search(r"(20\d{2}|19\d{2})[-_]?([01]\d)[-_]?([0-3]\d)", path.stem)
        label_date = "-".join(date_match.groups()) if date_match else ""
        return {
            "id": digest,
            "name": path.name,
            "stem": path.stem,
            "date": label_date,
            "path": display,
        }
