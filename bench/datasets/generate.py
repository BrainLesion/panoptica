"""Seeded synthetic dataset generator for parity and performance benchmarks.

Produces deterministic (fixed seed, no wall-clock) multi-instance label volumes
for testing correctness and performance. The dimension knobs are:

- **volume shape** — cubic *and* anisotropic (thin-slab, elongated, wide-2D),
- **voxel spacing** — uniform *and* anisotropic (`get_spacing`),
- **instance morphology** — ``ball``, ``ellipsoid``, ``shell`` (hollow), ``tube``
  (elongated) — these stress surface/EDT and clDSC very differently,
- **prediction displacement** — sub-voxel shift, dropout, over-segmentation
  (``split``) and under-segmentation (``merge``) to sweep the matcher / TP-FP-FN
  regime.

Generators are deterministic: same size key + seed -> identical arrays. The six
original cubic keys (``small`` … ``2d``) are byte-identical to before. Typical
usage:

    from bench.datasets.generate import generate_semantic_pair, get_spacing
    ref, pred = generate_semantic_pair(size="thin_slab")
    spacing = get_spacing("thin_slab")            # (1, 1, 4)
"""

from __future__ import annotations

import numpy as np
from typing import Literal

__all__ = [
    "generate_semantic_pair",
    "generate_unmatched_instance_pair",
    "generate_matched_instance_pair",
    "get_spacing",
    "SIZES",
]


# Each config: shape, n_instances, rmin, rmax, seed, and optional
# spacing / morphology / disp (prediction-displacement) overrides. The six
# original cubic keys keep ball morphology, unit spacing, and the default
# shift-1 / drop-10% displacement, so their output is unchanged.
_DEFAULT_DISP = {"shift": 1, "drop_frac": 0.1, "split_frac": 0.0, "merge_frac": 0.0}

_SIZE_CONFIGS: dict[str, dict] = {
    # --- original cubic keys (unchanged) ---
    "small": dict(shape=(128, 128, 128), n=20, rmin=3, rmax=6, seed=0),
    "medium": dict(shape=(256, 256, 256), n=200, rmin=3, rmax=8, seed=1),
    "large": dict(shape=(512, 512, 512), n=1000, rmin=3, rmax=10, seed=2),
    "pathological_many_tiny": dict(shape=(256, 256, 256), n=5000, rmin=1, rmax=2, seed=3),
    "pathological_few_huge": dict(shape=(512, 512, 512), n=5, rmin=8, rmax=15, seed=4),
    "2d": dict(shape=(1024, 1024), n=100, rmin=3, rmax=8, seed=5),

    # --- anisotropic volume shapes (non-cubic grids from real scans) ---
    "thin_slab": dict(shape=(512, 512, 64), n=300, rmin=3, rmax=8, seed=6,
                      spacing=(1.0, 1.0, 4.0)),          # CT stack: coarse Z
    "anisotropic_hd": dict(shape=(320, 320, 160), n=250, rmin=3, rmax=8, seed=7,
                           spacing=(0.8, 0.8, 2.0)),      # brain-MRI-ish
    "elongated": dict(shape=(384, 128, 128), n=150, rmin=3, rmax=6, seed=8),
    "2d_wide": dict(shape=(512, 2048), n=150, rmin=3, rmax=8, seed=9),

    # --- anisotropic voxel spacing on an isotropic grid ---
    "aniso_spacing": dict(shape=(256, 256, 256), n=200, rmin=3, rmax=8, seed=10,
                          spacing=(0.5, 0.5, 3.0)),

    # --- instance morphology (non-spherical) ---
    "ellipsoids": dict(shape=(256, 256, 256), n=200, rmin=3, rmax=8, seed=11,
                       morphology="ellipsoid"),
    "shells": dict(shape=(256, 256, 256), n=150, rmin=4, rmax=9, seed=12,
                   morphology="shell"),
    "tubes": dict(shape=(256, 256, 256), n=120, rmin=3, rmax=7, seed=13,
                  morphology="tube"),

    # --- displacement / overlap sweep (medium base, matcher-regime stress) ---
    "large_shift": dict(shape=(256, 256, 256), n=200, rmin=3, rmax=8, seed=14,
                        disp={"shift": 4}),
    "heavy_dropout": dict(shape=(256, 256, 256), n=200, rmin=3, rmax=8, seed=15,
                          disp={"drop_frac": 0.4}),
    "oversegmented": dict(shape=(256, 256, 256), n=200, rmin=3, rmax=8, seed=16,
                          disp={"split_frac": 0.3}),
    "undersegmented": dict(shape=(256, 256, 256), n=200, rmin=3, rmax=8, seed=17,
                           disp={"merge_frac": 0.3}),
}

SIZES = tuple(_SIZE_CONFIGS)

_SizeKey = Literal[
    "small", "medium", "large", "pathological_many_tiny", "pathological_few_huge", "2d",
    "thin_slab", "anisotropic_hd", "elongated", "2d_wide", "aniso_spacing",
    "ellipsoids", "shells", "tubes",
    "large_shift", "heavy_dropout", "oversegmented", "undersegmented",
]


def _cfg(size: str) -> dict:
    if size not in _SIZE_CONFIGS:
        raise ValueError(f"Unknown size {size!r}; choices: {list(_SIZE_CONFIGS)}")
    c = _SIZE_CONFIGS[size]
    ndim = len(c["shape"])
    return {
        "shape": c["shape"],
        "n": c["n"],
        "rmin": c["rmin"],
        "rmax": c["rmax"],
        "seed": c["seed"],
        "spacing": c.get("spacing", (1.0,) * ndim),
        "morphology": c.get("morphology", "ball"),
        "disp": {**_DEFAULT_DISP, **c.get("disp", {})},
    }


def get_spacing(size: _SizeKey = "medium") -> tuple[float, ...]:
    """Voxel spacing for a size key (unit spacing unless the config overrides it)."""
    return _cfg(size)["spacing"]


def _ball(radius: int, ndim: int) -> np.ndarray:
    """Boolean (2*radius+1)**ndim ball mask."""
    line = np.arange(-radius, radius + 1)
    grids = np.meshgrid(*([line] * ndim), indexing="ij")
    dist2 = sum(g.astype(np.int64) ** 2 for g in grids)
    return dist2 <= radius * radius


def _ellipsoid(radii: tuple[int, ...]) -> np.ndarray:
    """Boolean ellipsoid mask with per-axis half-extents ``radii``."""
    grids = np.meshgrid(*[np.arange(-h, h + 1) for h in radii], indexing="ij")
    dist2 = sum((g / max(1, h)) ** 2 for g, h in zip(grids, radii))
    return dist2 <= 1.0


def _shape_mask(
    kind: str, r: int, ndim: int, rng: np.random.Generator
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return (boolean mask, per-axis half-extents) for one instance.

    ``ball`` consumes no RNG here (mask is deterministic in r), so the original
    cubic keys keep their exact draw order and byte-identical output.
    """
    if kind == "ball":
        return _ball(r, ndim), (r,) * ndim
    if kind == "ellipsoid":
        radii = tuple(max(1, int(round(r * f))) for f in rng.uniform(0.5, 1.5, ndim))
        return _ellipsoid(radii), radii
    if kind == "shell":  # hollow ball: outer minus a centered inner ball
        outer = _ball(r, ndim)
        ir = max(1, r - 2)
        inner = np.zeros_like(outer)
        off = r - ir
        inner[tuple(slice(off, off + 2 * ir + 1) for _ in range(ndim))] = _ball(ir, ndim)
        return outer & ~inner, (r,) * ndim
    if kind == "tube":  # elongated along axis 0
        radii = tuple(min(3 * r, r * 4) if ax == 0 else max(1, r // 2) for ax in range(ndim))
        return _ellipsoid(radii), radii
    raise ValueError(f"Unknown morphology {kind!r}")


def _make_reference(
    shape: tuple[int, ...],
    n_instances: int,
    rng: np.random.Generator,
    rmin: int = 3,
    rmax: int = 7,
    morphology: str = "ball",
) -> tuple[np.ndarray, int]:
    """Place up to ``n_instances`` non-overlapping instances with distinct labels."""
    arr = np.zeros(shape, dtype=np.uint32)
    ndim = len(shape)
    label = 0
    attempts = 0
    while label < n_instances and attempts < n_instances * 25:
        attempts += 1
        r = int(rng.integers(rmin, rmax + 1))
        mask, half = _shape_mask(morphology, r, ndim, rng)
        if any(2 * h + 1 >= s for h, s in zip(half, shape)):
            continue  # instance too large for this (possibly thin) axis
        center = [int(rng.integers(h + 1, s - h - 1)) for h, s in zip(half, shape)]
        slices = tuple(slice(c - h, c + h + 1) for c, h in zip(center, half))
        region = arr[slices]
        if np.any(region[mask] != 0):
            continue  # keep instances separate
        label += 1
        region[mask] = label
    return arr, label


def _make_prediction(
    ref: np.ndarray,
    rng: np.random.Generator,
    shift: int = 1,
    drop_frac: float = 0.1,
    split_frac: float = 0.0,
    merge_frac: float = 0.0,
) -> np.ndarray:
    """Perturb the reference into a plausible prediction.

    shift/drop reproduce the original behaviour; ``split_frac`` over-segments
    (one instance -> two labels) and ``merge_frac`` under-segments (two -> one),
    exercising the matcher's FP/FN handling.
    """
    pred = ref.copy()
    for ax in range(ref.ndim):
        pred = np.roll(pred, int(rng.integers(-shift, shift + 1)), axis=ax)
    labels = [int(x) for x in np.unique(pred) if x > 0]
    n_drop = int(len(labels) * drop_frac)
    if n_drop:
        for label in rng.choice(labels, size=n_drop, replace=False):
            pred[pred == label] = 0

    if split_frac > 0.0:  # over-segmentation: bisect some instances along axis 0
        alive = [int(x) for x in np.unique(pred) if x > 0]
        next_id = (max(alive) if alive else 0) + 1
        for label in rng.choice(alive, size=int(len(alive) * split_frac), replace=False):
            idx = np.where(pred == label)
            if len(idx[0]) < 2:
                continue
            mid = int(np.median(idx[0]))
            pred[tuple(i[idx[0] > mid] for i in idx)] = next_id
            next_id += 1

    if merge_frac > 0.0:  # under-segmentation: relabel pairs to a shared id
        alive = [int(x) for x in np.unique(pred) if x > 0]
        rng.shuffle(alive)
        n_pairs = int(len(alive) * merge_frac) // 2
        for a, b in zip(alive[:n_pairs], alive[n_pairs : 2 * n_pairs]):
            pred[pred == b] = a
    return pred


def generate_semantic_pair(
    size: _SizeKey = "medium",
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic SEMANTIC pair (binary masks with instance labels).

    Args:
        size: Dataset size key (see ``SIZES``).
        seed: RNG seed. If None, uses the default seed for this size.

    Returns:
        (ref, pred): label arrays with shape from the size config.
    """
    c = _cfg(size)
    rng = np.random.default_rng(c["seed"] if seed is None else seed)
    ref, _ = _make_reference(c["shape"], c["n"], rng, c["rmin"], c["rmax"], c["morphology"])
    pred = _make_prediction(ref, rng, **c["disp"])
    return ref, pred


def generate_unmatched_instance_pair(
    size: _SizeKey = "medium",
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic UNMATCHED_INSTANCE pair (pre-approximated).

    Same as SEMANTIC; the distinction is conceptual (input type).
    """
    return generate_semantic_pair(size=size, seed=seed)


def generate_matched_instance_pair(
    size: _SizeKey = "medium",
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a deterministic MATCHED_INSTANCE pair with instance correspondence.

    Returns:
        (ref, pred, matched_ids): ref and pred are instance label arrays;
            matched_ids is (K, 2) array of [ref_id, pred_id] pairs.
    """
    c = _cfg(size)
    rng = np.random.default_rng(c["seed"] if seed is None else seed)
    ref, _ = _make_reference(c["shape"], c["n"], rng, c["rmin"], c["rmax"], c["morphology"])
    pred = _make_prediction(ref, rng, **c["disp"])

    # Match instances by spatial overlap (IoU-argmax), vectorized: the whole
    # intersection matrix comes from a single bincount over encoded (ref, pred)
    # voxel pairs (O(voxels)), not a nested per-pair full-volume scan.
    ref_flat = ref.reshape(-1).astype(np.int64)
    pred_flat = pred.reshape(-1).astype(np.int64)
    n_ref_max = int(ref_flat.max()) + 1
    n_pred_max = int(pred_flat.max()) + 1
    ref_area = np.bincount(ref_flat, minlength=n_ref_max).astype(np.float64)
    pred_area = np.bincount(pred_flat, minlength=n_pred_max).astype(np.float64)
    inter = np.bincount(
        ref_flat * n_pred_max + pred_flat, minlength=n_ref_max * n_pred_max
    ).reshape(n_ref_max, n_pred_max)
    union = ref_area[:, None] + pred_area[None, :] - inter
    iou = np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)

    matched_pairs = []
    used_pred_ids = set()
    for ref_id in range(1, n_ref_max):
        if ref_area[ref_id] == 0:
            continue
        best_iou, best_pred_id = -1.0, None
        for pred_id in range(1, n_pred_max):
            if pred_id in used_pred_ids or pred_area[pred_id] == 0:
                continue
            if iou[ref_id, pred_id] > best_iou:
                best_iou = iou[ref_id, pred_id]
                best_pred_id = pred_id
        if best_pred_id is not None and best_iou > 0:
            matched_pairs.append([ref_id, best_pred_id])
            used_pred_ids.add(best_pred_id)

    matched_ids = np.array(matched_pairs, dtype=np.uint32)
    return ref, pred, matched_ids
