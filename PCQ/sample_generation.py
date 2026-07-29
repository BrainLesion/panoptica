"""Generate a binary mask of non-touching disks for PCQ experiments.

Uses variable-radius Poisson-disk sampling (Bridson 2007, adapted) so that
disk positions form a blue-noise distribution. A position-dependent radius
map — peaking at ``r_max`` in the centre, falling off to ``r_min`` toward
the borders — gives a few large disks in the middle and many small ones
outside, while a border cap prevents any disk from crossing the image edge.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from skimage.draw import disk
from skimage.io import imsave

app = typer.Typer(add_completion=False)


def _radius_map_factory(
    size: int, r_min: float, r_max: float, sigma_frac: float
):
    """Return a callable ``(y, x) -> radius`` implementing the radius map.

    Peaks at ``r_max`` at the image centre, falls off with a Gaussian of
    standard deviation ``sigma_frac * size`` toward ``r_min``, then is
    capped by the distance to the nearest border (minus 1 px to keep the
    disk strictly inside).
    """
    center = (size - 1) / 2.0
    sigma = sigma_frac * size
    two_sigma_sq = 2.0 * sigma * sigma

    def radius_at(y: float, x: float) -> float:
        d2 = (y - center) ** 2 + (x - center) ** 2
        r_gauss = r_min + (r_max - r_min) * math.exp(-d2 / two_sigma_sq)
        r_border = min(y, x, size - 1 - y, size - 1 - x) - 1.0
        return min(r_gauss, r_border)

    return radius_at


def poisson_disk_sample(
    size: int,
    r_min: float,
    r_max: float,
    sigma_frac: float,
    gap: float,
    k: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Variable-radius Poisson-disk sampling.

    Two points ``p_a``, ``p_b`` with disk radii ``r_a``, ``r_b`` are
    compatible iff ``dist(p_a, p_b) >= r_a + r_b + gap``. Candidates whose
    local radius (after border capping) drops below ``r_min`` are rejected,
    so every returned disk has radius ``>= r_min`` and lies fully inside
    the image.
    """
    radius_at = _radius_map_factory(size, r_min, r_max, sigma_frac)
    cell = float(r_min)
    n_cells = int(math.ceil(size / cell))
    grid: list[list[int]] = [[] for _ in range(n_cells * n_cells)]

    def cell_index(y: float, x: float) -> int:
        cy = min(int(y / cell), n_cells - 1)
        cx = min(int(x / cell), n_cells - 1)
        return cy * n_cells + cx

    points: list[tuple[float, float]] = []
    radii: list[float] = []

    def try_add(y: float, x: float) -> bool:
        if not (0.0 <= y < size and 0.0 <= x < size):
            return False
        r_c = radius_at(y, x)
        if r_c < r_min:
            return False
        search = r_c + r_max + gap
        span = int(math.ceil(search / cell))
        cy = min(int(y / cell), n_cells - 1)
        cx = min(int(x / cell), n_cells - 1)
        for gy in range(max(0, cy - span), min(n_cells, cy + span + 1)):
            for gx in range(max(0, cx - span), min(n_cells, cx + span + 1)):
                for j in grid[gy * n_cells + gx]:
                    dy = y - points[j][0]
                    dx = x - points[j][1]
                    if dy * dy + dx * dx < (r_c + radii[j] + gap) ** 2:
                        return False
        points.append((y, x))
        radii.append(r_c)
        grid[cell_index(y, x)].append(len(points) - 1)
        return True

    # Seed the process. Try uniformly until a valid position is found.
    for _ in range(10_000):
        y0 = float(rng.uniform(0, size))
        x0 = float(rng.uniform(0, size))
        if try_add(y0, x0):
            break
    else:
        raise RuntimeError(
            "Could not place an initial point — r_min may be too large "
            "for the image size."
        )

    active = [0]
    while active:
        idx_pos = int(rng.integers(len(active)))
        i = active[idx_pos]
        py, px = points[i]
        r_p = radii[i]
        placed = False
        # Annulus [d_min, 2*d_min] with d_min = r_p + r_min + gap (the
        # smallest allowed centre-to-centre distance from p to any new
        # point). This mirrors Bridson's [r, 2r] annulus.
        d_min = r_p + r_min + gap
        for _ in range(k):
            theta = float(rng.uniform(0.0, 2.0 * math.pi))
            d = float(rng.uniform(d_min, 2.0 * d_min))
            cy_ = py + d * math.sin(theta)
            cx_ = px + d * math.cos(theta)
            if try_add(cy_, cx_):
                active.append(len(points) - 1)
                placed = True
                break
        if not placed:
            active.pop(idx_pos)

    return np.asarray(points, dtype=float), np.asarray(radii, dtype=float)


def render_mask(
    points: np.ndarray, radii: np.ndarray, size: int
) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    for (y, x), r in zip(points, radii):
        if r <= 0:
            continue
        rr, cc = disk((y, x), r, shape=mask.shape)
        mask[rr, cc] = True
    return mask


def shrink_by_one(mask: np.ndarray) -> np.ndarray:
    """Erode every foreground component by one pixel on each side."""
    from scipy.ndimage import binary_erosion

    return binary_erosion(mask, iterations=1)


@app.command()
def main(
    output: Annotated[
        Path, typer.Option(help="Where to write the PNG.")
    ] = Path("mask.png"),
    size: Annotated[
        int, typer.Option(help="Side length of the square image.")
    ] = 1024,
    r_min: Annotated[
        float,
        typer.Option(help="Minimum disk radius (in pixels). Enforced everywhere."),
    ] = 8.0,
    r_max: Annotated[
        float,
        typer.Option(
            help="Maximum disk radius (in pixels), reached at the image centre."
        ),
    ] = 60.0,
    sigma_frac: Annotated[
        float,
        typer.Option(
            help=(
                "Std-dev of the Gaussian radius map, as a fraction of `size`. "
                "Smaller = tighter high-radius blob in the centre; larger = "
                "nearly uniform radius."
            )
        ),
    ] = 0.35,
    gap: Annotated[
        float, typer.Option(help="Minimum gap between disks, in pixels.")
    ] = 1.0,
    k: Annotated[
        int, typer.Option(help="Bridson candidate attempts per active seed.")
    ] = 30,
    shrink: Annotated[
        bool,
        typer.Option(help="Erode every disk by one pixel per side after sampling."),
    ] = False,
    seed: Annotated[int, typer.Option(help="RNG seed for reproducibility.")] = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points, radii = poisson_disk_sample(
        size=size,
        r_min=r_min,
        r_max=r_max,
        sigma_frac=sigma_frac,
        gap=gap,
        k=k,
        rng=rng,
    )
    mask = render_mask(points, radii, size)
    if shrink:
        mask = shrink_by_one(mask)
    imsave(output, (mask.astype(np.uint8) * 255))
    typer.echo(f"Wrote {output} ({mask.sum()} foreground px, {len(points)} disks).")
    return mask


if __name__ == "__main__":
    app()
