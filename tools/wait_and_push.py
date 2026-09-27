"""Wait for a Kaggle GPU slot, push the combined multi-fold kernel, confirm it is
RUNNING, then exit.

Kaggle caps concurrent batch GPU sessions at 2 and REJECTS a third push rather
than queueing it, so the remaining folds have to be submitted the moment a slot
frees. Once the combined kernel reports RUNNING it is entirely server-side and the
local machine can be shut down.

Progress is written to work/handoff.log so it can be read fresh at any time.

  python tools/wait_and_push.py --folds 2 3 4 --watch 0 1
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER = "tejuuss"
LOGP = os.path.join(HERE, "work", "handoff.log")


def say(msg):
    line = "{0}  {1}".format(time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOGP), exist_ok=True)
    with open(LOGP, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", required=True)
    ap.add_argument("--watch", type=int, nargs="+", default=[0, 1])
    a = ap.parse_args()

    watch = ["rsna-knee-arma-f{0}".format(f) for f in a.watch]
    tag = "".join(str(f) for f in a.folds)
    target = "rsna-knee-arma-f{0}".format(tag)

    say("waiting for a GPU slot to push {0} (folds {1})".format(target, a.folds))
    deadline = time.time() + 60 * 60 * 6
    while time.time() < deadline:
        st = {s: status(s) for s in watch}
        busy = sum(1 for v in st.values() if v in ("RUNNING", "QUEUED"))
        say("watch={0} busy={1}".format({k[-2:]: v for k, v in st.items()}, busy))

        if busy < 2:
            out = subprocess.run(
                [sys.executable, os.path.join(HERE, "tools", "push_train_kernel.py"),
                 "--folds"] + [str(f) for f in a.folds],
                capture_output=True, text=True, shell=True, cwd=HERE)
            txt = (out.stdout or "") + (out.stderr or "")
            if "successfully pushed" in txt and "Maximum batch GPU" not in txt:
                say("PUSHED {0}".format(target))
                # confirm it really started before declaring handoff complete
                for _ in range(20):
                    time.sleep(30)
                    s = status(target)
                    say("{0} -> {1}".format(target, s))
                    if s in ("RUNNING", "QUEUED"):
                        say("HANDOFF COMPLETE -- training is now fully server-side.")
                        say("Safe to shut down the laptop. Outputs save privately "
                            "to kernel {0}/{1}.".format(USER, target))
                        return 0
                    if s in ("COMPLETE", "ERROR", "CANCEL"):
                        say("unexpected early terminal state: {0}".format(s))
                        return 1
                say("pushed but never observed RUNNING -- check manually")
                return 1
            if "Maximum batch GPU" in txt:
                say("slot reported free but push still rejected; retrying")
            else:
                say("push failed:\n{0}".format(txt[-800:]))
        time.sleep(60)

    say("timed out waiting for a slot")
    return 1


if __name__ == "__main__":
    sys.exit(main())
