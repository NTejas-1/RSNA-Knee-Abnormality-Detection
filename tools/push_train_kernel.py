"""Push the arm-A training kernel, one fold per kernel so folds run concurrently.

Fold index is baked into a generated copy of train_armA.py because Kaggle script
kernels take no runtime parameters.

  python tools/push_train_kernel.py --fold 0
  python tools/push_train_kernel.py --folds 0 1 2 3 4
  python tools/push_train_kernel.py --fold 0 --epochs 1 --smoke
"""
import argparse
import json
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "train", "train_armA.py")
BUILD = os.path.join(HERE, "train", "_build")
USER = "tejuuss"

CACHE_KERNELS = ["{0}/rsna-knee-cache-shard-{1}".format(USER, i) for i in range(4)]
DATASETS = [
    "{0}/rsna-knee-opus-labels-v1".format(USER),
    "marwanmath/resnet-50-radimagenet-marwan",
]


def render(folds, epochs, smoke, repaired=False, armb=False):
    code = open(SRC, encoding="utf-8").read()
    code = code.replace("FOLDS = [0]", "FOLDS = {0}".format(list(folds)), 1)
    if armb:
        # arm B == arm A with ImageNet weights and a different init seed. See the
        # ENCODER comment in train_armA.py for why arm B is rendered from this file
        # instead of a separate script.
        code = code.replace('ENCODER = "rad"', 'ENCODER = "imagenet"', 1)
        code = code.replace("MODEL_SEED = None", "MODEL_SEED = 777", 1)
        code = code.replace('RUN_PREFIX = "armA"', 'RUN_PREFIX = "armB"', 1)
        for need in ('ENCODER = "imagenet"', "MODEL_SEED = 777", 'RUN_PREFIX = "armB"'):
            if need not in code:
                raise SystemExit("arm B render failed to apply: {0}".format(need))
    if repaired:
        code = code.replace('LABELS_FILE = "labels_v1_opus45.csv"',
                            'LABELS_FILE = "labels_v2_repaired.csv"', 1)
        if "labels_v2_repaired.csv" not in code:
            raise SystemExit("failed to swap in the repaired labels file")
    if epochs is not None:
        code = code.replace("EPOCHS = 8", "EPOCHS = {0}".format(epochs), 1)
    if smoke:
        # tiny run purely to prove the pipeline end to end on real data
        code = code.replace(
            '    log("fold {0}: train {1}, val {2}".format(FOLD, len(tr_i), len(va_i)))',
            "    tr_i, va_i = tr_i[:48], va_i[:24]\n"
            "    log('SMOKE: truncated to train %d val %d' % (len(tr_i), len(va_i)))",
            1)
    return code


def push(folds, epochs, smoke, repaired=False, armb=False):
    folds = list(folds)
    if smoke:
        slug, title = "rsna-knee-armA-smoke", "RSNA Knee ArmA Smoke"
    elif len(folds) == 1:
        slug = "rsna-knee-armA-f{0}".format(folds[0])
        title = "RSNA Knee ArmA F{0}".format(folds[0])
    else:
        tag = "".join(str(f) for f in folds)
        slug = "rsna-knee-armA-f{0}".format(tag)
        title = "RSNA Knee ArmA F{0}".format(tag)
    if repaired:
        # distinct slug so the repaired run never overwrites the v1 checkpoints,
        # which are still the baseline we have to beat
        slug = slug + "-rep"
        title = title + " Repaired"
    if armb:
        slug = slug.replace("armA", "armB")
        title = title.replace("ArmA", "ArmB")
    d = os.path.join(BUILD, slug)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

    with open(os.path.join(d, "train_armA.py"), "w", encoding="utf-8") as fh:
        fh.write(render(folds, epochs, smoke, repaired, armb))

    meta = {
        "id": "{0}/{1}".format(USER, slug),
        "title": title,
        "code_file": "train_armA.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        # arm B downloads torchvision ImageNet weights, so it needs internet. The
        # RadImageNet arm does not (weights come from an attached dataset), and the
        # submission kernel must stay offline regardless.
        "enable_internet": bool(armb),
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
    # the CLI exits 0 even when the push is rejected (e.g. "Maximum batch GPU
    # session count of 2 reached"), so trust the message, not the return code
    ok = ("successfully pushed" in out) and ("error" not in out.lower())
    return ok, slug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int)
    ap.add_argument("--folds", type=int, nargs="+",
                    help="train these folds SEQUENTIALLY in ONE kernel")
    ap.add_argument("--separate", action="store_true",
                    help="with --folds, push one kernel per fold instead")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--repaired", action="store_true",
                    help="train on labels_v2_repaired.csv instead of v1")
    ap.add_argument("--armb", action="store_true",
                    help="ImageNet weights + seed 777 (the decorrelated arm)")
    a = ap.parse_args()

    if a.folds and not a.separate:
        ok, slug = push(a.folds, a.epochs, a.smoke, a.repaired, a.armb)
        print("folds {0} -> {1}: {2}".format(
            a.folds, slug, "pushed" if ok else "FAILED"))
        return

    folds = a.folds if a.folds else [a.fold if a.fold is not None else 0]
    for f in folds:
        ok, slug = push([f], a.epochs, a.smoke, a.repaired, a.armb)
        print("fold {0} -> {1}: {2}".format(f, slug, "pushed" if ok else "FAILED"))


if __name__ == "__main__":
    main()
