"""Push a one-off probe/test kernel.

  # CPU metadata probe, competition data only
  python tools/push_probe_kernel.py submit/probe_test_meta.py rsna-knee-meta-probe

  # GPU test of the blend path, with our checkpoints attached
  python tools/push_probe_kernel.py submit/test_blend.py rsna-knee-blend-test \\
      --gpu --dataset tejuuss/rsna-knee-arm-ckpts

The submission kernel is offline, so this is offline too -- if it needs the network
to load a checkpoint, that is a bug worth finding here rather than in a submission.
"""
import argparse
import json
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(HERE, "train", "_build")
USER = "tejuuss"
COMP = "rsna-knee-abnormality-detection"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("slug")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--internet", action="store_true")
    ap.add_argument("--dataset", action="append", default=[])
    a = ap.parse_args()

    src = a.script if os.path.isabs(a.script) else os.path.join(HERE, a.script)
    if not os.path.exists(src):
        raise SystemExit("no such script: {0}".format(src))

    d = os.path.join(BUILD, a.slug)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
    fname = os.path.basename(src)
    shutil.copy(src, os.path.join(d, fname))

    meta = {
        "id": "{0}/{1}".format(USER, a.slug),
        # Kaggle derives the slug from the TITLE, not from this id, so the title
        # must resolve to the slug or status/output lookups hit a 404
        "title": a.slug,
        "code_file": fname,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": bool(a.gpu),
        "enable_tpu": False,
        "enable_internet": bool(a.internet),
        "dataset_sources": list(a.dataset),
        "competition_sources": [COMP],
        "kernel_sources": [],
    }
    with open(os.path.join(d, "kernel-metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    env = dict(os.environ)
    env["KAGGLE_CONFIG_DIR"] = HERE
    p = subprocess.run(["kaggle", "kernels", "push", "-p", d],
                       capture_output=True, text=True, shell=True, env=env)
    out = ((p.stdout or "") + (p.stderr or "")).strip()
    print(out[-400:])
    # the CLI exits 0 even when a push is rejected, so trust the text not the code
    print("{0}: {1}".format(
        a.slug, "pushed" if "successfully pushed" in out else "FAILED"))


if __name__ == "__main__":
    main()
