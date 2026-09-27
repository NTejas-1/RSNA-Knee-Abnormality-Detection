"""
Correlation gate between two arms, plus what a blend would actually buy.

Charles Savas measured +0.004 from a second arm at rank correlation 0.83 with the
first. Above roughly 0.95 a blend is dead weight -- the arms are saying the same
thing and you have paid GPU hours for nothing. This decides whether arm B earns a
full 5-fold run BEFORE those hours are spent.

Correlation is computed on RANKS, per label, because the metric is column-wise ROC
AUC and AUC only sees ordering. Two arms can disagree wildly on calibration and
still be perfectly correlated for scoring purposes.

  python tools/arm_corr.py --a oof/armA_oof_fold0.csv --b bp3/armB_oof_fold0.csv
"""
import argparse
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
          "Medial OA", "Lateral OA", "PF OA", "Effusion",
          "Synovitis", "Baker's", "Contusion", "Fracture"]

GOLD_COLS = ["GOLD_ACL", "GOLD_MCL", "GOLD_MEDIAL_MENISCUS",
             "GOLD_LATERAL_MENISCUS", "GOLD_MEDIAL_OA", "GOLD_LATERAL_OA",
             "GOLD_PF_OA", "GOLD_EFFUSION", "GOLD_SYNOVITIS", "GOLD_BAKERS",
             "GOLD_CONTUSION", "GOLD_FRACTURE"]


def auc(y, s):
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    p, n = (y == 1).sum(), (y == 0).sum()
    if p == 0 or n == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--labels", default=os.path.join(
        HERE, "labels_ds", "labels_v1_opus45.csv"))
    args = ap.parse_args()

    A = pd.concat([pd.read_csv(p) for p in args.a], ignore_index=True)
    B = pd.concat([pd.read_csv(p) for p in args.b], ignore_index=True)
    for d in (A, B):
        d["STUDY_UID"] = d["STUDY_UID"].astype(str)

    m = A.merge(B, on="STUDY_UID", suffixes=("_a", "_b"))
    print("arm A rows {0}, arm B rows {1}, overlap {2}".format(len(A), len(B), len(m)))
    if len(m) == 0:
        raise SystemExit("no shared studies -- the two arms used different fold splits")

    lab = pd.read_csv(args.labels)
    lab["STUDY_UID"] = lab["STUDY_UID"].astype(str)
    g = m.merge(lab[["STUDY_UID", "IS_GOLD"] + GOLD_COLS], on="STUDY_UID", how="left")
    gm = g["IS_GOLD"].fillna(False).to_numpy(bool)
    print("gold studies in the overlap: {0}".format(int(gm.sum())))

    print()
    print("{0:18s} {1:>7s} {2:>7s} {3:>7s} {4:>7s} {5:>5s}".format(
        "finding", "rankcor", "A_gold", "B_gold", "blend", "pos"))
    print("-" * 62)
    cors, rows = [], []
    for j, f in enumerate(LABELS):
        ra = m[f + "_a"].rank().to_numpy()
        rb = m[f + "_b"].rank().to_numpy()
        c = float(np.corrcoef(ra, rb)[0, 1])
        cors.append(c)

        aa = bb = bl = np.nan
        pos = 0
        if gm.sum() >= 8:
            yv = g[GOLD_COLS[j]].to_numpy(float)[gm]
            pos = int(np.nansum(yv == 1))
            # rank-percentile average: lossless for AUC and needs no calibration
            pa = pd.Series(m[f + "_a"]).rank(pct=True).to_numpy()[gm]
            pb = pd.Series(m[f + "_b"]).rank(pct=True).to_numpy()[gm]
            aa, bb = auc(yv, pa), auc(yv, pb)
            bl = auc(yv, 0.5 * pa + 0.5 * pb)
        rows.append((f, c, aa, bb, bl, pos))
        print("{0:18s} {1:7.3f} {2:7.3f} {3:7.3f} {4:7.3f} {5:5d}".format(
            f, c, aa, bb, bl, pos))

    mc = float(np.mean(cors))
    va = [r[2] for r in rows if np.isfinite(r[2])]
    vb = [r[3] for r in rows if np.isfinite(r[3])]
    vl = [r[4] for r in rows if np.isfinite(r[4])]
    print()
    print("mean rank correlation: {0:.3f}".format(mc))
    if va:
        print("gold macro  A {0:.4f}   B {1:.4f}   50/50 blend {2:.4f}   "
              "blend-minus-best {3:+.4f}".format(
                  np.mean(va), np.mean(vb), np.mean(vl),
                  np.mean(vl) - max(np.mean(va), np.mean(vb))))

    print()
    if mc > 0.95:
        print("VERDICT: TOO CORRELATED ({0:.3f}) -- a blend adds nothing. "
              "Do not spend GPU hours on this arm.".format(mc))
    elif mc > 0.85:
        print("VERDICT: MARGINAL ({0:.3f}) -- above the 0.85 gate. Expect well "
              "under the +0.004 the community reported at 0.83.".format(mc))
    else:
        print("VERDICT: DECORRELATED ({0:.3f}) -- below the 0.85 gate, so the arm "
              "is worth completing.".format(mc))

    print()
    print("CAVEAT: if arm B here is a short probe rather than a full run, it is")
    print("undertrained and therefore noisier, which DEPRESSES correlation. Treat")
    print("this as an optimistic screen -- a real gate needs both arms fully trained.")


if __name__ == "__main__":
    main()
