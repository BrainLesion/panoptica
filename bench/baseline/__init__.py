"""Reference-standard panoptica for performance baselining.

`panoptica` (this repo) is a batched rewrite; to claim it is faster we time it
against **the published reference standard** — upstream BrainLesion/panoptica,
pinned to :data:`BASELINE_COMMIT`. That package owns the import name `panoptica`
too, so it can't share our interpreter: it's installed once into a dedicated
venv and driven through a subprocess (:mod:`bench.baseline._runner`).

    from bench.baseline import run_baseline, baseline_available, ensure_baseline

`ensure_baseline()` builds `.venv-baseline` on first use (needs network).
`run_baseline` returns `{group: {"scalars": {...}, "lists": {metric: [...]}}}`,
or — with `time_n` — `{"time_ms": float, "tp": int}` (median-of-N in the venv).

This is a PERFORMANCE reference only; it is never imported for correctness
(that's device-parity + golden files, §5).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

BASELINE_COMMIT = "7abda75"
BASELINE_SPEC = f"git+https://github.com/BrainLesion/panoptica@{BASELINE_COMMIT}"

_ROOT = Path(__file__).resolve().parents[2]
_VENV = _ROOT / ".venv-baseline"
_PY = _VENV / "bin" / "python"
_RUNNER = Path(__file__).resolve().parent / "_runner.py"


def baseline_available() -> bool:
    """True once the reference-standard venv exists."""
    return _PY.exists()


def ensure_baseline() -> None:
    """Create `.venv-baseline` with the reference panoptica (idempotent)."""
    if baseline_available():
        return
    subprocess.run([sys.executable, "-m", "venv", str(_VENV)], check=True)
    pip = str(_VENV / "bin" / "pip")
    subprocess.run([pip, "install", "-q", "--upgrade", "pip"], check=True)
    # reference evaluates a `sitk.Image` annotation at import time, so these must
    # be present even though they're nominally optional.
    subprocess.run(
        [pip, "install", "-q", BASELINE_SPEC, "SimpleITK>=2.2.2", "nibabel", "pynrrd"],
        check=True,
    )


def run_baseline(
    a,
    b,
    expected_input,
    *,
    matcher="naive",
    threshold=0.5,
    instance_metrics,
    global_metrics=None,
    groups=None,
    spacing=None,
    time_n=None,
):
    """Evaluate `(a, b)` with the reference standard in its venv.

    `a`/`b` are passed to the reference `evaluate` in the same positional order
    the caller uses, so any differential comparison stays order-matched.
    `groups` is `None` or `{name: [labels, ...]}`.

    Default returns `{group: {"scalars": ..., "lists": ...}}`. With `time_n`
    set, the venv runs 1 warmup + median-of-`time_n` timing (excluding
    interpreter startup) and returns `{"time_ms": float, "tp": int}`.
    """
    if not baseline_available():
        raise RuntimeError(
            "baseline venv missing; call ensure_baseline() (needs network) first"
        )
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        np.savez(td / "arrays.npz", a=np.asarray(a), b=np.asarray(b))
        params = {
            "npz": str(td / "arrays.npz"),
            "expected_input": expected_input,
            "matcher": matcher,
            "threshold": threshold,
            "instance_metrics": list(instance_metrics),
            "global_metrics": list(global_metrics or []),
            "groups": groups,
            "spacing": list(spacing) if spacing else None,
            "time_n": time_n,
        }
        (td / "params.json").write_text(json.dumps(params))
        out = td / "out.json"
        # cwd=td so the repo-root `panoptica/` dir can't shadow the venv's copy.
        proc = subprocess.run(
            [str(_PY), str(_RUNNER), str(td / "params.json"), str(out)],
            cwd=td,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(
                f"baseline runner failed (exit {proc.returncode}):\n{proc.stderr}"
            )
        return json.loads(out.read_text())
