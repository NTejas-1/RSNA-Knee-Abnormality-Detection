"""Push the arm-B training kernel (decorrelated arm).

Internet is ENABLED so timm can fetch ImageNet pretrained weights. That is safe
here: only the training kernel needs it, and the offline submission kernel will
load the checkpoint we save from a dataset.

  python tools/push_armb_kernel.py --folds 0 1 2 3 4
  python tools/push_armb_kernel.py --folds 0 --epochs 1 --smoke
"""
import argparse
import json
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "train", "train_armB.py")
BUILD = os.path.join(HERE, "train", "_build")
USER = "tejuuss"

CACHE_KERNELS = ["{0}/rsna-knee-cache-shard-{1}".format(USER, i) for i in range(4)]
DATASETS = [
    "{0}/rsna-knee-opus-labels-v1".format(USER),
    "marwanmath/resnet-50-radimagenet-marwan",
]


def render(folds, epochs, smoke):
    code = open(SRC, encoding="utf-8").read()
    code = code.replace("FOLDS = [0]", "FOLDS = {0}".format(list(folds)), 1)
    if epochs is not None:
        code = code.replace("EPOCHS = 8", "EPOCHS = {0}".format(epochs), 1)
    if smoke:
        code = code.replace(
            '    log("fold {0}: train {1}, val {2}".format(FOLD, len(tr_i), len(va_i)))',
            "    tr_i, va_i = tr_i[:48], va_i[:24]\n"
            "    log('SMOKE: train %d val %d' % (len(tr_i), len(va_i)))",
            1)
    return code


def push(folds, epochs, smoke):
    folds = list(folds)
    if smoke:
        slug, title = "rsna-knee-armB-smoke", "RSNA Knee ArmB Smoke"
    else:
        tag = "".join(str(f) for f in folds)
        slug = "rsna-knee-armB-f{0}".format(tag)
        title = "RSNA Knee ArmB F{0}".format(tag)
    d = os.path.join(BUILD, slug)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

    with open(os.path.join(d, "train_armB.py"), "w", encoding="utf-8") as fh:
        fh.write(render(folds, epochs, smoke))

    meta = {
        "id": "{0}/{1}".format(USER, slug),
        "title": title,
        "code_file": "train_armB.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        # needed for timm pretrained download; the submission kernel stays offline
        "enable_internet": True,
        "dataset_sources": DATASETS,
        "competition_sources": [],
        "kernel_sources": CACHE_KERNELS,
    }
    with open(os.path.join(d, "kernel-metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    env = dict(os.environ)
    env["KAGGLE_CONFIG_DIR"] = HERE
    p = subprocess.run(["kaggle", "kernels", "push", "-p", d],
                       capture_output=True, text=True, shell=True, env=env)
    out = ((p.stdout or "") + (p.stderr or "")).strip()
    print(out[-500:])
    ok = "successfully pushed" in out and "error" not in out.lower()
    return ok, slug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    ok, slug = push(a.folds, a.epochs, a.smoke)
    print("folds {0} -> {1}: {2}".format(a.folds, slug, "pushed" if ok else "FAILED"))


if __name__ == "__main__":
    main()
