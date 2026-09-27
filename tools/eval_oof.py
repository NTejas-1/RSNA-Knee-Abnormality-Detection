"""Evaluate pooled out-of-fold predictions from arm A.

This is the honest gate before spending more compute:
  1. Pooled OOF against the 58 gold studies -- the only real ground truth.
  2. Per-label comparison against the extracted-label AUC (0.8919 macro) so we can
     see where the IMAGE model beats the REPORT, which is exactly where label
     repair can help.
  3. Correlation of our arm against the reference stack, if reference predictions
     are supplied -- Charles Savas measured +0.004 from a decorrelated arm at
     correlation 0.83, so a highly-correlated arm is not worth blending.

Caveats applied deliberately:
  * OOF must cover every study exactly once; overlap means it is not out of fold.
  * The 58 gold rows are ENRICHED (ACL 41% vs 13% corpus-wide), so absolute gold
    AUC is not comparable to the leaderboard. Use it for ranking, not calibration.
  * Differences under 0.003 are noise (Ziad Ahmed's measured floor).
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
          "Medial OA", "Lateral OA", "PF OA", "Effusion",
          "Synovitis", "Baker's", "Contusion", "Fracture"]
GOLD_COL = {
    "ACL": "GOLD_ACL", "MCL": "GOLD_MCL",
    "Medial Meniscus": "GOLD_MEDIAL_MENISCUS",
    "Lateral Meniscus": "GOLD_LATERAL_MENISCUS",
    "Medial OA": "GOLD_MEDIAL_OA", "Lateral OA": "GOLD_LATERAL_OA",
    "PF OA": "GOLD_PF_OA", "Effusion": "GOLD_EFFUSION",
    "Synovitis": "GOLD_SYNOVITIS", "Baker's": "GOLD_BAKERS",
    "Contusion": "GOLD_CONTUSION", "Fracture": "GOLD_FRACTURE",
}
# label-extraction AUC measured on the same 58 gold studies
LLM_AUC = {
    "ACL": 0.996, "MCL": 0.992, "Medial Meniscus": 0.955,
    "Lateral Meniscus": 0.875, "Medial OA": 0.905, "Lateral OA": 0.847,
    "PF OA": 0.945, "Effusion": 0.820, "Synovitis": 0.783,
    "Baker's": 0.831, "Contusion": 0.865, "Fracture": 0.890,
}


def auc(y, s):
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    p, n = int((y == 1).sum()), int((y == 0).sum())
    if p == 0 or n == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-glob", default=os.path.join(HERE, "oof", "armA_oof_fold*.csv"))
    ap.add_argument("--labels", default=os.path.join(
        HERE, "labels_ds", "labels_v1_opus45.csv"))
    ap.add_argument("--reference", help="optional reference-stack submission csv")
    a = ap.parse_args()

    files = sorted(glob.glob(a.oof_glob))
    if not files:
        raise SystemExit("no OOF files matched " + a.oof_glob)
    print("OOF files:", [os.path.basename(f) for f in files])
    oof = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    oof["STUDY_UID"] = oof["STUDY_UID"].astype(str)

    dup = int(oof["STUDY_UID"].duplicated().sum())
    if dup:
        raise SystemExit("{0} studies appear in multiple folds -- not out of fold".format(dup))
    print("pooled OOF studies:", len(oof))

    lab = pd.read_csv(a.labels)
    lab["STUDY_UID"] = lab["STUDY_UID"].astype(str)
    m = lab.merge(oof, on="STUDY_UID", how="inner", suffixes=("", "_p"))
    print("matched to labels:", len(m), "of", len(lab))

    gold = m[m["IS_GOLD"].fillna(False).astype(bool)]
    print("gold studies in OOF:", len(gold))

    print()
    print("== per-label AUC on the 58 gold studies ==")
    print("{0:18s} {1:>7s} {2:>7s} {3:>8s} {4:>5s}".format(
        "finding", "ARM_A", "LLM", "delta", "pos"))
    print("-" * 52)
    arm, llm = [], []
    for f in LABELS:
        y = gold[GOLD_COL[f]].to_numpy()
        pcol = f + "_p" if (f + "_p") in gold.columns else f
        s = gold[pcol].to_numpy()
        aa = auc(y, s)
        ll = LLM_AUC[f]
        arm.append(aa)
        llm.append(ll)
        flag = ""
        if np.isfinite(aa):
            if aa - ll > 0.02:
                flag = "  IMAGE WINS"
            elif ll - aa > 0.02:
                flag = "  report wins"
        print("{0:18s} {1:7.3f} {2:7.3f} {3:+8.3f} {4:5d}{5}".format(
            f, aa, ll, aa - ll, int(np.nansum(y)), flag))
    print()
    print("ARM A macro: {0:.4f}   LLM labels macro: {1:.4f}   delta {2:+.4f}".format(
        np.nanmean(arm), np.nanmean(llm), np.nanmean(arm) - np.nanmean(llm)))
    print()
    print("NOTE gold is enriched (ACL 41% vs 13% corpus) -- rank findings with this,")
    print("     do not read it as a leaderboard estimate. Noise floor is 0.003.")

    print()
    print("== where the image model beats the report ==")
    print("(these are the labels worth repairing / worth weighting our arm on)")
    for f, aa, ll in zip(LABELS, arm, llm):
        if np.isfinite(aa) and aa - ll > 0.02:
            print("  {0:18s} image {1:.3f} vs report {2:.3f} ({3:+.3f})".format(
                f, aa, ll, aa - ll))

    if a.reference and os.path.exists(a.reference):
        ref = pd.read_csv(a.reference)
        idcol = "StudyInstanceUID" if "StudyInstanceUID" in ref.columns else ref.columns[0]
        ref[idcol] = ref[idcol].astype(str)
        j = m.merge(ref, left_on="STUDY_UID", right_on=idcol,
                    how="inner", suffixes=("", "_ref"))
        print()
        print("== correlation with the reference stack ==")
        print("(Charles Savas: +0.004 from an arm at correlation 0.83;")
        print(" above ~0.95 a blend adds nothing)")
        cors = []
        for f in LABELS:
            pc = f + "_p" if (f + "_p") in j.columns else f
            rc = f + "_ref" if (f + "_ref") in j.columns else None
            if rc is None or pc not in j.columns:
                continue
            c = float(pd.Series(j[pc]).rank().corr(pd.Series(j[rc]).rank()))
            cors.append(c)
            print("  {0:18s} {1:.3f}".format(f, c))
        if cors:
            print("  mean rank correlation: {0:.3f}".format(float(np.mean(cors))))
            if np.mean(cors) > 0.95:
                print("  -> TOO CORRELATED, blending will not help")
            else:
                print("  -> decorrelated enough to be worth blending")


if __name__ == "__main__":
    main()
