# v2 (panoptica v2 — local dev)

Batched, backend-swappable (numpy/CuPy) rewrite of panoptica. See `../V2_SPEC.md`.
Local development only — no CI, no publishing. Package/import name `v2` is a working title.

```sh
pip install -e '.[cuda]'   # CuPy 14 / CUDA 12 on this host
pytest tests/unit
```
