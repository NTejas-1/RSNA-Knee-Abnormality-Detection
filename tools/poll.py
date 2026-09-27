"""Poll Kaggle kernel(s) until COMPLETE/ERROR. Writes fresh status to stdout.

  python tools/poll.py rsna-knee-cache-smoke
  python tools/poll.py rsna-knee-cache-s0 rsna-knee-cache-s1 ...
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER = "tejuuss"


def status(slug):
    env = dict(os.environ)
    env["KAGGLE_CONFIG_DIR"] = HERE
    p = subprocess.run(
        ["kaggle", "kernels", "status", "{0}/{1}".format(USER, slug)],
        capture_output=True, text=True, shell=True, env=env,
    )
    out = ((p.stdout or "") + (p.stderr or "")).strip().replace("\n", " ")
    for s in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"):
        if s in out:
            return s, out
    return "UNKNOWN", out


def main():
    slugs = sys.argv[1:]
    if not slugs:
        sys.exit("usage: poll.py <slug> [slug...]")
    timeout = time.time() + 60 * 60 * 12
    last = {}
    while time.time() < timeout:
        done = 0
        for s in slugs:
            st, raw = status(s)
            if last.get(s) != st:
                print("{0}  {1:10s} {2}".format(
                    time.strftime("%H:%M:%S"), st, s), flush=True)
                last[s] = st
            if st in ("COMPLETE", "ERROR", "CANCEL"):
                done += 1
        if done == len(slugs):
            break
        time.sleep(30)
    print("final:", {s: last.get(s) for s in slugs}, flush=True)


if __name__ == "__main__":
    main()
