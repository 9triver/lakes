"""Adaptive statistical methods used by multispectral water classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ClusterResult:
    water_score: np.ndarray
    vote: np.ndarray
    status: str
    best_k: int
    score: float
    sample_size: int
    water_cluster_mndwi: float


def waterdetect_cluster(
    mndwi: np.ndarray,
    ndwi: np.ndarray,
    nir: np.ndarray,
    swir1: np.ndarray,
    valid: np.ndarray,
    *,
    random_seed: int,
    max_sample_size: int,
) -> ClusterResult:
    """WaterDetect-style sample clustering followed by GaussianNB expansion."""
    output = np.zeros(mndwi.shape, dtype=np.float32)
    valid_positions = np.flatnonzero(valid)
    if valid_positions.size < 64:
        return _fallback_cluster(mndwi, ndwi, nir, swir1, valid, "too_few_pixels")

    flat_features = np.column_stack(
        (
            mndwi.ravel()[valid_positions],
            ndwi.ravel()[valid_positions],
            nir.ravel()[valid_positions] * 4.0,
            swir1.ravel()[valid_positions] * 4.0,
        )
    ).astype(np.float32)
    rng = np.random.default_rng(random_seed)
    sample_indexes = _balanced_sample_indexes(
        np.maximum(flat_features[:, 0], flat_features[:, 1]),
        min(max(64, max_sample_size), flat_features.shape[0]),
        rng,
    )
    sample = flat_features[sample_indexes]
    if np.unique(np.round(sample, decimals=5), axis=0).shape[0] < 2:
        return _fallback_cluster(mndwi, ndwi, nir, swir1, valid, "constant_scene")

    scaler = StandardScaler().fit(sample)
    scaled_sample = scaler.transform(sample)
    best_labels: np.ndarray | None = None
    best_k = 0
    best_score = float("-inf")
    max_k = min(5, sample.shape[0] - 1)
    try:
        for cluster_count in range(2, max_k + 1):
            labels = AgglomerativeClustering(
                n_clusters=cluster_count, linkage="ward"
            ).fit_predict(scaled_sample)
            if np.unique(labels).size < 2:
                continue
            score = float(calinski_harabasz_score(scaled_sample, labels))
            if score > best_score:
                best_score = score
                best_k = cluster_count
                best_labels = labels
        if best_labels is None:
            raise ValueError("No valid clustering solution")

        cluster_mndwi = np.array(
            [
                float(sample[best_labels == label, 0].mean())
                for label in range(best_k)
            ]
        )
        cluster_swir1 = np.array(
            [
                float(sample[best_labels == label, 3].mean()) / 4.0
                for label in range(best_k)
            ]
        )
        water_cluster = int(
            np.argmax(cluster_mndwi - np.maximum(cluster_swir1 - 0.12, 0.0))
        )
        water_mndwi = float(cluster_mndwi[water_cluster])
        if water_mndwi < -0.08:
            return _fallback_cluster(
                mndwi, ndwi, nir, swir1, valid, "no_plausible_water_cluster"
            )

        classifier = GaussianNB(var_smoothing=1e-6).fit(
            scaled_sample, best_labels
        )
        classes = classifier.classes_
        if classes is None:
            raise ValueError("GaussianNB did not expose fitted classes")
        probabilities = classifier.predict_proba(
            scaler.transform(flat_features)
        )[:, list(classes).index(water_cluster)]
        plausible = (nir.ravel()[valid_positions] < 0.35) & (
            swir1.ravel()[valid_positions] < 0.25
        )
        probabilities = probabilities.astype(np.float32)
        probabilities[~plausible] *= 0.1
        output.ravel()[valid_positions] = probabilities
        return ClusterResult(
            water_score=output,
            vote=output >= 0.55,
            status="ready",
            best_k=best_k,
            score=best_score,
            sample_size=int(sample.shape[0]),
            water_cluster_mndwi=water_mndwi,
        )
    except (ValueError, FloatingPointError):
        return _fallback_cluster(mndwi, ndwi, nir, swir1, valid, "fallback")


def bounded_otsu_threshold(values: np.ndarray) -> float:
    selected = values[np.isfinite(values)]
    if selected.size < 32 or float(np.std(selected)) < 1e-4:
        return 0.10
    lower, upper = np.percentile(selected, (1, 99))
    if upper <= lower:
        return 0.10
    histogram, edges = np.histogram(selected, bins=256, range=(lower, upper))
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight_low = np.cumsum(histogram).astype(np.float64)
    weight_high = np.cumsum(histogram[::-1]).astype(np.float64)[::-1]
    mean_low = np.divide(
        np.cumsum(histogram * centers),
        weight_low,
        out=np.zeros_like(weight_low),
        where=weight_low > 0,
    )
    mean_high = np.divide(
        np.cumsum((histogram * centers)[::-1])[::-1],
        weight_high,
        out=np.zeros_like(weight_high),
        where=weight_high > 0,
    )
    variance = weight_low[:-1] * weight_high[1:] * (
        mean_low[:-1] - mean_high[1:]
    ) ** 2
    threshold = float(centers[int(np.argmax(variance))])
    return float(np.clip(threshold, -0.05, 0.25))


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


def _fallback_cluster(
    mndwi: np.ndarray,
    ndwi: np.ndarray,
    nir: np.ndarray,
    swir1: np.ndarray,
    valid: np.ndarray,
    status: str,
) -> ClusterResult:
    water_score = sigmoid((0.6 * mndwi + 0.4 * ndwi - 0.02) / 0.09)
    water_score *= (nir < 0.35) & (swir1 < 0.25)
    water_score = water_score.astype(np.float32)
    water_score[~valid] = 0.0
    return ClusterResult(
        water_score=water_score,
        vote=valid & (water_score >= 0.60),
        status=status,
        best_k=0,
        score=0.0,
        sample_size=int(np.count_nonzero(valid)),
        water_cluster_mndwi=0.0,
    )


def _balanced_sample_indexes(
    water_score: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    high = np.flatnonzero(water_score >= 0.10)
    low = np.flatnonzero(water_score < 0.10)
    target_high = min(high.size, sample_size // 2)
    target_low = min(low.size, sample_size // 2)
    selected = [
        rng.choice(high, size=target_high, replace=False) if target_high else high,
        rng.choice(low, size=target_low, replace=False) if target_low else low,
    ]
    current = target_high + target_low
    if current < sample_size:
        chosen = np.concatenate(selected) if current else np.empty(0, dtype=int)
        remaining = np.setdiff1d(
            np.arange(water_score.size), chosen, assume_unique=False
        )
        fill_size = min(sample_size - current, remaining.size)
        if fill_size:
            selected.append(rng.choice(remaining, size=fill_size, replace=False))
    return np.concatenate(selected).astype(int, copy=False)
