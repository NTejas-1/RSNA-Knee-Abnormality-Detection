"""Generate and push Kaggle script kernels for cache building.

Kaggle script kernels cannot take runtime parameters, so shard configuration is
baked into a generated copy of build_cache.py before each push.

  python tools/push_cache_kernel.py --smoke          # ~9 studies, validates logic
  python tools/push_cache_kernel.py --shard 0 --n 6  # one real shard
  python tools/push_cache_kernel.py --all --n 6      # push all 6 shards
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "cache", "build_cache.py")
BUILD = os.path.join(HERE, "cache", "_build")
USER = "tejuuss"
COMP = "rsna-knee-abnormality-detection"


def render(shard, n_shards, threads=8):
    code = open(SRC, encoding="utf-8").read()
    code = code.replace(
        'SHARD = int(os.environ.get("RSNA_SHARD", "0"))',
        "SHARD = {0}".format(shard))
    code = code.replace(
        'N_SHARDS = int(os.environ.get("RSNA_N_SHARDS", "6"))',
        "N_SHARDS = {0}".format(n_shards))
    code = code.replace(
        'THREADS = int(os.environ.get("RSNA_THREADS", "8"))',
        "THREADS = {0}".format(threads))
    return code


def push(shard, n_shards, slug, title, enable_gpu=False):
    d = os.path.join(BUILD, slug)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

    with open(os.path.join(d, "build_cache.py"), "w", encoding="utf-8") as fh:
        fh.write(render(shard, n_shards))

    meta = {
        "id": "{0}/{1}".format(USER, slug),
        "title": title,
        "code_file": "build_cache.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [COMP],
        "kernel_sources": [],
    }
    with open(os.path.join(d, "kernel-metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    env = dict(os.environ)
    env["KAGGLE_CONFIG_DIR"] = HERE
    p = subprocess.run(["kaggle", "kernels", "push", "-p", d],
                       capture_output=True, text=True, shell=True, env=env)
    out = (p.stdout or "") + (p.stderr or "")
    print(out.strip()[-400:])
    return p.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--shard", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()

    if a.smoke:
        # 500 shards -> shard 0 holds ~9 studies: enough to validate ordering,
        # cropping, laterality and to measure seconds-per-study
        ok = push(0, 500, "rsna-knee-cache-smoke", "RSNA Knee Cache Smoke")
        print("smoke pushed" if ok else "smoke FAILED")
        return

    targets = range(a.n) if a.all else [a.shard]
    for s in targets:
        if s is None:
            sys.exit("need --shard or --all")
        ok = push(s, a.n,
                  "rsna-knee-cache-s{0}".format(s),
                  "RSNA Knee Cache Shard {0}".format(s))
        print("shard {0}: {1}".format(s, "pushed" if ok else "FAILED"))


if __name__ == "__main__":
    main()
