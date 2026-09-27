"""
Push the blended notebook as a competition submission kernel.

The attachment list is copied verbatim from push/kernel-metadata.json -- the exact
metadata that produced the 0.940 score -- with our checkpoint dataset added and
nothing removed. Every one of those 13 datasets, 2 kernel sources and the DINOv2
model source is loaded by some arm of the parent pipeline; dropping one does not
degrade the score gracefully, it raises on the first cell that needs it.

enable_internet stays False. This is a code competition and the submission kernel
must run offline -- our checkpoints come from the attached dataset, which is why
they were uploaded rather than downloaded.

  python submit/push_submission.py --variant full
  python submit/push_submission.py --variant halved
"""
import argparse
import json
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(HERE, "submit", "_push")
USER = "tejuuss"

# verbatim from the 0.940 run
DATASETS = [
    "dreaddevelopment/raptor-knee-maxspan",
    "dreaddevelopment/raptor-knee-native384",
    "dreaddevelopment/raptor-knee-native384dense",
    "mattiaangeli/knee-mri-fold-weights",
    "mattiaangeli/opencv-python-headless-4120088-x86",
    "marwanmath/resnet-50-radimagenet-marwan",
    "mattiaangeli/rsna-knee-coat-resgated-ep10-top3",
    "mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2",
    "antoinegg1/rsna-knee-e11-diverse-heads-v20",
    "antoinegg1/rsna-knee-e9-radimagenet-heads-v15",
    "pilkwang/rsna-knee-llm-labels",
    "prvsiyan/rsna-knee-v52-radimagenet-heads-20260812",
    "pilkwang/rsna-knee-weights",
    # ours
    "tejuuss/rsna-knee-arm-ckpts",
]
KERNEL_SOURCES = ["sofiaanjenje/rsna-knee-e11-train",
                  "sofiaanjenje/rsna-knee-e13-train"]
MODEL_SOURCES = ["metaresearch/dinov2/PyTorch/small/1"]
COMP = "rsna-knee-abnormality-detection"

VARIANTS = {
    "full":   ("out/notebook_blend_full.ipynb",
               "rsna-knee-blend-full",
               "RSNA Knee Blend Full"),
    "halved": ("out/notebook_blend_halved.ipynb",
               "rsna-knee-blend-halved",
               "RSNA Knee Blend Halved"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    a = ap.parse_args()
    nb_rel, slug, title = VARIANTS[a.variant]

    nb = os.path.join(HERE, nb_rel)
    if not os.path.exists(nb):
        raise SystemExit("missing {0} -- run make_blend_cell.py --insert".format(nb))

    # sanity: our cell must be present and the gates must still be there
    doc = json.load(open(nb, encoding="utf-8"))
    cells = doc["cells"]
    if len(cells) != 25:
        raise SystemExit("expected 25 cells, found {0}".format(len(cells)))
    ours = "".join(cells[23].get("source", []))
    if "APPENDED ARM BLEND" not in ours:
        raise SystemExit("cell 23 is not our blend cell")
    publish = "".join(cells[24].get("source", []))
    for gate in ("_DINOV2_MATCHED_MEMBERS", "V18_CALIBRATOR_APPLIED"):
        if gate not in publish:
            raise SystemExit("publish cell lost its {0} gate".format(gate))
    print("verified: 25 cells, blend at 23, publish gates intact at 24")

    d = os.path.join(BUILD, slug)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
    code_file = slug + ".ipynb"
    shutil.copy(nb, os.path.join(d, code_file))

    meta = {
        "id": "{0}/{1}".format(USER, slug),
        # Kaggle derives the slug from the TITLE; keep them consistent or status
        # and output lookups 404
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "health"],
        "dataset_sources": DATASETS,
        "kernel_sources": KERNEL_SOURCES,
        "competition_sources": [COMP],
        "model_sources": MODEL_SOURCES,
        "machine_shape": "NvidiaTeslaT4",
    }
    with open(os.path.join(d, "kernel-metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    env = dict(os.environ)
    env["KAGGLE_CONFIG_DIR"] = HERE
    p = subprocess.run(["kaggle", "kernels", "push", "-p", d],
                       capture_output=True, text=True, shell=True, env=env)
    out = ((p.stdout or "") + (p.stderr or "")).strip()
    print(out[-500:])
    # the CLI exits 0 even on rejection, so trust the text
    print("{0}: {1}".format(slug, "pushed" if "successfully pushed" in out else "FAILED"))


if __name__ == "__main__":
    main()
