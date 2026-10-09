"""Run the whole pipeline in order. Stops at the first failing step.

Usage:
    python run_all.py                  # all steps except the ~15 min replay
    python run_all.py --with-replay    # also Step 16 (replay.py)
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("Step 1: load + sanity checks", ["src/load.py"]),
    ("Steps 3-4: parse addresses", ["src/textparse.py"]),
    ("Step 5: text prior", ["src/prior.py"]),
    ("Steps 6-7: visit evidence + integrity", ["src/evidence.py"]),
    ("Step 8: remark corrections", ["src/remarks.py"]),
    ("Step 10: neighbour street keys", ["src/neighbours.py"]),
    ("Step 9: fusion (2 passes)", ["src/fuse.py"]),
    ("Step 12: R90 + action", ["src/calibrate.py"]),
    ("Steps 14-15: directions + consumer files", ["src/outputs.py"]),
]
REPLAY = ("Step 16: learning-over-time replay", ["src/replay.py"])
FINAL = [
    ("Final evaluation table", ["src/final_eval.py"]),
    ("Scorer", ["evaluate.py", "outputs/geocodes.csv", "run_all"]),
]


def main():
    steps = STEPS + ([REPLAY] if "--with-replay" in sys.argv else []) + FINAL
    t0 = time.time()
    for name, args in steps:
        print(f"\n===== {name}: python {' '.join(args)}", flush=True)
        t = time.time()
        r = subprocess.run([sys.executable, *args], cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"FAILED: {name} (exit {r.returncode})")
        print(f"----- {name}: {time.time() - t:.0f} s", flush=True)
    print(f"\nall {len(steps)} steps done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
