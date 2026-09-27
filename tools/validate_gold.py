"""Validate extracted labels against the 58 gold studies.

Two questions:
  1. Is macro AUC in the 0.89-0.92 band the community reports? (sanity)
  2. Is Synovitis disproportionately "not_mentioned"? (tests the core hypothesis
     behind the 3-state design and the Phase 3 repair step)

Also checks calibration, which Archit Konde flagged as more informative than AUC:
labels near p=0.3 should be positive roughly 30% of the time.
"""
import io
import os
import subprocess
import pandas as pd
import numpy as np

CONN = "oruwqcb-kaggle"
RUN_TAG = "v1_opus45"

GOLD_COL = {
    "ACL": "GOLD_ACL", "MCL": "GOLD_MCL",
    "Medial Meniscus": "GOLD_MEDIAL_MENISCUS",
    "Lateral Meniscus": "GOLD_LATERAL_MENISCUS",
    "Medial OA": "GOLD_MEDIAL_OA", "Lateral OA": "GOLD_LATERAL_OA",
    "PF OA": "GOLD_PF_OA", "Effusion": "GOLD_EFFUSION",
    "Synovitis": "GOLD_SYNOVITIS", "Baker's": "GOLD_BAKERS",
    "Contusion": "GOLD_CONTUSION", "Fracture": "GOLD_FRACTURE",
}
ORDER = list(GOLD_COL.keys())


def fetch(sql):
    p = subprocess.run(
        ["snow", "sql", "-c", CONN, "-q", sql, "--format", "csv"],
        capture_output=True, text=True, shell=True,
    )
    out = p.stdout or ""
    start = out.find("STUDY_UID")
    if start < 0:
        raise SystemExit("query failed:\n" + out[-3000:])
    return pd.read_csv(io.StringIO(out[start:]))


def auc(y, s):
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    # rank-based AUC with tie handling
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    sql = (
        "USE DATABASE RSNA_KNEE; USE SCHEMA LABELS; "
        "SELECT e.STUDY_UID, e.FINDING, e.STATE, e.PROB, "
        + ", ".join("r.{0}".format(c) for c in GOLD_COL.values()) +
        " FROM EXTRACTED_LABELS e JOIN REPORTS r ON r.STUDY_UID = e.STUDY_UID "
        "WHERE e.RUN_TAG = '{0}' AND r.IS_GOLD".format(RUN_TAG)
    )
    df = fetch(sql)
    print("rows:", len(df), "studies:", df.STUDY_UID.nunique())

    print()
    print("== state distribution per finding (pct) ==")
    print("{0:18s} {1:>7s} {2:>7s} {3:>7s}".format("finding", "pos", "expl_neg", "SILENT"))
    st = df.pivot_table(index="FINDING", columns="STATE", values="PROB",
                        aggfunc="count", fill_value=0)
    for f in ORDER:
        if f not in st.index:
            continue
        row = st.loc[f]
        tot = row.sum()
        print("{0:18s} {1:6.0f}% {2:7.0f}% {3:6.0f}%".format(
            f, 100 * row.get("p", 0) / tot, 100 * row.get("n", 0) / tot,
            100 * row.get("u", 0) / tot))

    print()
    print("== per-label AUC vs gold (58 studies) ==")
    print("{0:18s} {1:>6s} {2:>5s} {3:>6s}".format("finding", "AUC", "pos", "silent"))
    aucs = []
    for f in ORDER:
        sub = df[df.FINDING == f]
        if not len(sub):
            continue
        y = sub[GOLD_COL[f]].to_numpy()
        a = auc(y, sub["PROB"].to_numpy())
        aucs.append(a)
        sil = (sub["STATE"] == "u").mean()
        print("{0:18s} {1:6.3f} {2:5d} {3:5.0f}%".format(
            f, a, int(np.nansum(y)), 100 * sil))
    print()
    print("MACRO AUC: {0:.4f}   (community single-prompt band: 0.89-0.92)".format(
        np.nanmean(aucs)))

    print()
    print("== calibration: observed positive rate by predicted bin ==")
    df["gold"] = [row[GOLD_COL[row["FINDING"]]] for _, row in df.iterrows()]
    bins = [0, .1, .2, .3, .5, .7, .9, 1.01]
    df["bin"] = pd.cut(df["PROB"], bins, right=False)
    cal = df.groupby("bin", observed=True)["gold"].agg(["mean", "count"])
    print(cal.to_string())

    print()
    print("== silent cells: are they actually positive? ==")
    sil = df[df.STATE == "u"]
    print("silent cells:", len(sil), "| observed gold positive rate:",
          round(float(sil["gold"].mean()), 3))
    nn = df[df.STATE == "n"]
    print("explicit-neg cells:", len(nn), "| observed gold positive rate:",
          round(float(nn["gold"].mean()), 3))
    print()
    print("If silent >> explicit-neg, the 3-state split is carrying real signal")
    print("and Phase 3 should repair only the silent cells.")


if __name__ == "__main__":
    main()
