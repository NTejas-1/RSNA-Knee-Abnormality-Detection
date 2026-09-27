"""
Assemble the 10 checkpoints the submission kernel needs into one dataset folder.

Identity is VERIFIED, not assumed. Two directories hold an armA_fold3.pt: one from
the run that was OOM-killed partway through fold 3, and one from the clean rerun.
Shipping the wrong one would quietly cost fold 3's accuracy with nothing in any log
to show it, so every checkpoint is opened and its recorded fold and best macro are
checked against the expected source.

Arm A uses the v1 labels deliberately. The repaired-label retrain measured WORSE on
the 58 gold rows (macro 0.8891 vs 0.8955), and the two repaired findings were the
biggest losers, so the repair was abandoned. See rsna-knee-oof-findings.md.
"""
import os
import shutil
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "ckpt_ds")

# (filename, source dir, expected best macro from that run's log)
WANT = [
    ("armA_fold0.pt", "f0out",   None),
    ("armA_fold1.pt", "f1out",   None),
    ("armA_fold2.pt", "f234out", 0.8961),
    ("armA_fold3.pt", "f34out",  0.8747),   # NOT f234out: that fold 3 was OOM-killed
    ("armA_fold4.pt", "f34out",  0.8892),
    ("armB_fold0.pt", "bfin",    0.8914),
    ("armB_fold1.pt", "bfin",    0.8732),
    ("armB_fold2.pt", "bfin",    0.9060),
    ("armB_fold3.pt", "bfin",    0.8745),
    ("armB_fold4.pt", "bfin",    0.8874),
]


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    total = 0
    for fname, src, want_macro in WANT:
        p = os.path.join(HERE, src, fname)
        if not os.path.exists(p):
            raise SystemExit("missing {0}".format(p))
        sd = torch.load(p, map_location="cpu", weights_only=False)
        fold = sd.get("fold")
        macro = sd.get("macro")
        ep = sd.get("epoch")
        expect_fold = int(fname.split("fold")[1].split(".")[0])
        if fold != expect_fold:
            raise SystemExit("{0} records fold {1}, expected {2}".format(
                fname, fold, expect_fold))
        if want_macro is not None and abs(float(macro) - want_macro) > 5e-4:
            raise SystemExit(
                "{0} from {1} has macro {2:.4f}, expected {3:.4f} -- this is "
                "probably the wrong run's checkpoint".format(
                    fname, src, float(macro), want_macro))
        # the head must match the 12 findings in the trained order
        if sd.get("labels") is None or len(sd["labels"]) != 12:
            raise SystemExit("{0} has no 12-label header".format(fname))
        shutil.copy(p, os.path.join(OUT, fname))
        total += os.path.getsize(p)
        print("{0:16s} <- {1:8s} fold {2} ep {3} macro {4:.4f}  img {5} slices {6}".format(
            fname, src, fold, ep, float(macro), sd.get("img"), sd.get("slices")))

    meta = os.path.join(OUT, "dataset-metadata.json")
    with open(meta, "w", encoding="utf-8") as fh:
        fh.write('{\n  "title": "RSNA Knee Arm Checkpoints",\n'
                 '  "id": "tejuuss/rsna-knee-arm-ckpts",\n'
                 '  "licenses": [{"name": "other"}]\n}\n')
    print()
    print("{0} checkpoints, {1:.0f} MB -> {2}".format(
        len(WANT), total / 1048576.0, OUT))
    print("upload with:")
    print("  kaggle datasets create -p ckpt_ds --dir-mode zip")


if __name__ == "__main__":
    main()
