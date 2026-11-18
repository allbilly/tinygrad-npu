#!/usr/bin/env python3
"""Run inmodel_program_cost.py with the four-way kernel split installed."""
import pathlib, runpy, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thread_split  # noqa: E402
argv = sys.argv[1:]
if argv[:1] == ["--"]: argv = argv[1:]
if "--no-split" in argv:
    argv = [a for a in argv if a != "--no-split"]
    thread_split.THREAD_SPLIT.clear()
    print("thread_split: DISABLED for this run (parity control)", flush=True)
thread_split.install()
print(f"thread_split: {sorted(thread_split.THREAD_SPLIT)}", flush=True)
sys.argv = [str(HERE / "inmodel_program_cost.py"), *argv]
runpy.run_path(str(HERE / "inmodel_program_cost.py"), run_name="__main__")
