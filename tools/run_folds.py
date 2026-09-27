"""Run the remaining training folds through Kaggle's 2-concurrent-GPU-session cap.

Kaggle rejects a push with "Maximum batch GPU session count of 2 reached" rather
than queueing it, and the CLI still exits 0, so folds must be submitted as slots
free up. This waits for a slot, pushes the next fold, and repeats.

  python tools/run_folds.py --folds 2 3 4
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER = "tejuuss"
MAX_CONCURRENT = 2


def kag(args):
    env = dict(os.environ)
    env["KAGGLE_CONFIG_DIR"] = HERE
    p = subprocess.run(["kaggle"] + args, capture_output=True, text=True,
                       shell=True, env=env)
    return ((p.stdout or "") + (p.stderr or "")).strip()


def status(slug):
    out = kag(["kernels", "status", "{0}/{1}".format(USER, slug)])
    if "404" in out or "not found" in out.lower():
        return "ABSENT"
    for s in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"):
        if s in out:
            return s
    return "UNKNOWN"


def push_fold(fold):
    out = subprocess.run(
        [sys.executable, os.path.join(HERE, "tools", "push_train_kernel.py"),
         "--fold", str(fold)],
        capture_output=True, text=True, shell=True, cwd=HERE)
    txt = (out.stdout or "") + (out.stderr or "")
    return "successfully pushed" in txt and "Maximum batch GPU" not in txt, txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", required=True)
    ap.add_argument("--watch", type=int, nargs="+", default=[0, 1],
                    help="folds already submitted, counted against the cap")
    a = ap.parse_args()

    pending = list(a.folds)
    live = ["rsna-knee-arma-f{0}".format(f) for f in a.watch]
    submitted = list(a.watch)

    deadline = time.time() + 60 * 60 * 14
    while (pending or any(status(s) == "RUNNING" for s in live)) and time.time() < deadline:
        states = {s: status(s) for s in live}
        running = sum(1 for v in states.values() if v in ("RUNNING", "QUEUED"))
        print("{0}  running={1}  {2}  pending={3}".format(
            time.strftime("%H:%M:%S"), running,
            {k[-2:]: v for k, v in states.items()}, pending), flush=True)

        while pending and running < MAX_CONCURRENT:
            f = pending[0]
            ok, txt = push_fold(f)
            if ok:
                print("  pushed fold {0}".format(f), flush=True)
                pending.pop(0)
                slug = "rsna-knee-arma-f{0}".format(f)
                live.append(slug)
                submitted.append(f)
                running += 1
                time.sleep(20)
            else:
                if "Maximum batch GPU" in txt:
                    print("  slot still busy; waiting", flush=True)
                else:
                    print("  push failed for fold {0}:\n{1}".format(f, txt[-600:]),
                          flush=True)
                break
        if not pending and all(
                states.get(s) in ("COMPLETE", "ERROR", "CANCEL") for s in live):
            break
        time.sleep(60)

    print("final:", {s: status(s) for s in live}, flush=True)


if __name__ == "__main__":
    main()
