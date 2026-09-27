"""
Out-of-fold label repair, applied ONLY where the evidence supports it.

Method (following Archit Konde, who reached 0.950 single-fold with it):
  * strictly out of fold -- in-fold predictions just parrot the labels back
  * MIX, do not replace -- repaired = (1-w) * llm_prob + w * oof_pred
  * keep fractions; rounding to 0/1 measurably hurt
  * push heavier toward the model than feels comfortable

What differs here, and why: we do NOT repair every cell. The calibration-gap
analysis on the 58 gold studies (tools/repair_targets.py) showed repair is
justified for some findings and actively harmful for others:

  Synovitis  86.3% silent corpus-wide, 2.5% explicitly negative, gold silent cells
             34% positive against a mean LLM probability of 0.24 -> 14 of 41 gold
             silent cells are recoverable false negatives. PRIMARY target, and the
             one with the largest silent-cell sample (n=41).
  Effusion   NOW INCLUDED. The silence gap is +0.35 on only 6 silent gold cells,
             which alone is too small to act on, but the 5-fold OOF evaluation
             gives independent confirmation from a much larger sample: image AUC
             0.957 vs the label's 0.820, the biggest margin of any finding, over
             35 gold positives. Both diagnostics agree, so the earlier "magnitude
             threshold, not silence" reading does not hold -- silent effusion
             cells really are 67% positive in gold.
  PF OA      EXCLUDED, reversing the earlier decision. The silence gap (+0.19, 5
             recoverable) says repair, but the 5-fold OOF says the image model is
             far WORSE than this label: 0.810 vs 0.945. Mixing predictions into a
             label that is already much better than the model would corrupt it.
             Repair needs BOTH conditions: silence hides positives AND the model
             can see what the report missed. PF OA fails the second.
  Fracture   gap +0.05 with only 2 recoverable cells. Real but negligible; not
             worth the variance. EXCLUDED.
  Lateral OA gold silent cells are only 5% positive -- silence there genuinely
             means absent. Repairing would inject noise. EXCLUDED.
  Baker's    3% positive on 30 silent gold cells. EXCLUDED.
  Medial OA  0% positive on 14 silent gold cells. EXCLUDED.

Note that Baker's, Medial OA and Contusion are all findings where the image model
BEATS the label on gold, yet they are still excluded: being a better ranker does
not mean the silent cells hide positives. Both tests have to pass.

Cells the report states explicitly (p or n) are never touched: those are the
labels the radiologist actually wrote.
"""
import argparse
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
          "Medial OA", "Lateral OA", "PF OA", "Effusion",
          "Synovitis", "Baker's", "Contusion", "Fracture"]

# weight on the image model, per finding, applied to SILENT cells only.
# Absent from this dict means "never repair".
REPAIR_W = {
    "Synovitis": 0.60,
    "Effusion": 0.50,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join(
        HERE, "labels_ds", "labels_v1_opus45.csv"))
    ap.add_argument("--oof", nargs="+", required=True,
                    help="armA_oof_fold*.csv from every fold (must cover all studies)")
    ap.add_argument("--out", default=os.path.join(
        HERE, "labels_ds", "labels_v2_repaired.csv"))
    a = ap.parse_args()

    lab = pd.read_csv(a.labels)
    lab["STUDY_UID"] = lab["STUDY_UID"].astype(str)

    oof = pd.concat([pd.read_csv(p) for p in a.oof], ignore_index=True)
    oof["STUDY_UID"] = oof["STUDY_UID"].astype(str)
    dup = oof["STUDY_UID"].duplicated().sum()
    if dup:
        raise SystemExit("{0} studies appear in more than one OOF file -- folds "
                         "overlap, so these are not out-of-fold".format(dup))
    print("labels:", len(lab), "| oof coverage:", len(oof))

    cov = oof["STUDY_UID"].isin(lab["STUDY_UID"]).sum()
    print("oof studies matched to labels:", cov)
    missing = len(lab) - lab["STUDY_UID"].isin(oof["STUDY_UID"]).sum()
    if missing:
        print("WARNING: {0} studies have no OOF prediction; they keep original "
              "labels".format(missing))

    m = lab.merge(oof, on="STUDY_UID", how="left", suffixes=("", "_oof"))
    out = lab.copy()

    print()
    print("{0:18s} {1:>8s} {2:>7s} {3:>9s} {4:>9s}".format(
        "finding", "silent", "weight", "mean_before", "mean_after"))
    total = 0
    for f in LABELS:
        w = REPAIR_W.get(f)
        st = m[f + "_state"]
        silent = (st == "u").to_numpy()
        pred = m[f + "_oof"].to_numpy(float)
        have = silent & np.isfinite(pred)
        if w is None:
            print("{0:18s} {1:8d} {2:>7s} {3:9.3f} {4:>9s}".format(
                f, int(silent.sum()), "--", float(lab[f].mean()), "unchanged"))
            continue
        before = float(lab[f].mean())
        v = out[f].to_numpy(float).copy()
        v[have] = (1.0 - w) * v[have] + w * pred[have]
        out[f] = v
        total += int(have.sum())
        print("{0:18s} {1:8d} {2:7.2f} {3:9.3f} {4:9.3f}".format(
            f, int(silent.sum()), w, before, float(out[f].mean())))

    out.to_csv(a.out, index=False)
    print()
    print("repaired {0} cells across {1} findings".format(total, len(REPAIR_W)))
    print("wrote", a.out)
    print()
    print("Do NOT validate the retrained model against these repaired labels --")
    print("CV improves by construction. Judge it on the 58 gold rows and the LB.")


if __name__ == "__main__":
    main()
