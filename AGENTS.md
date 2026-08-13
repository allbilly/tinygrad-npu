# Notes

- Always speak English only
- comments and docstrings are not counted in sz.py, no need delete them for lines cleanup

- Run tests with `-n12` for speed (e.g. `python -m pytest test/null/test_dtype.py -x -q -n12`)
- Run `python -m mypy tinygrad/` to typecheck
- Run `python -m ruff check .` to lint
- Read `./tinygrad/viz/README.md` for profiling and debugging rewrite rules
- Run Rockchip hardware census tests with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`
- When Rockchip NPU health is in doubt, run `.venv/bin/python ~/rk3588/examples/elementwise.py` before concluding that a reboot is needed
