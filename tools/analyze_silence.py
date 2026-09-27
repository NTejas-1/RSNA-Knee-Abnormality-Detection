"""Per-finding test of the silence hypothesis.

The aggregate silent-vs-explicit-negative comparison is confounded by base rate:
findings differ hugely in prevalence, so pooling them hides the effect. This
computes, within each finding, the gold positive rate among cells the report was
SILENT on versus cells it EXPLICITLY called negative.

A finding where silent cells are much more often positive than explicit-negative
cells is one where Phase 3 repair should fire. A finding where they are similar is
one where the report is simply wrong about magnitude (a threshold problem), and
repair will not help.
"""
import io
import subprocess
import numpy as np
import pandas as pd

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
    i = out.find("STUDY_UID")
    if i < 0:
        raise SystemExit("query failed:\n" + out[-3000:])
    return pd.read_csv(io.StringIO(out[i:]))


def main():
    sql = (
        "USE DATABASE RSNA_KNEE; USE SCHEMA LABELS; "
        "SELECT e.STUDY_UID, e.FINDING, e.STATE, e.PROB, "
        + ", ".join("r." + c for c in GOLD_COL.values()) +
        " FROM EXTRACTED_LABELS e JOIN REPORTS r ON r.STUDY_UID = e.STUDY_UID "
        "WHERE e.RUN_TAG = '{0}' AND r.IS_GOLD".format(RUN_TAG)
    )
    df = fetch(sql)
    df["gold"] = [row[GOLD_COL[row["FINDING"]]] for _, row in df.iterrows()]

    print("Within-finding gold positive rate by report state")
    print("(n = cell count; rate = fraction actually positive per radiologists)")
    print()
    print("{0:18s} {1:>12s} {2:>14s} {3:>10s} {4:>8s}".format(
        "finding", "SILENT", "EXPLICIT-NEG", "base", "repair?"))
    print("-" * 68)

    rows = []
    for f in ORDER:
        sub = df[df.FINDING == f]
        sil = sub[sub.STATE == "u"]["gold"]
        neg = sub[sub.STATE == "n"]["gold"]
        base = sub["gold"].mean()
        s_rate = sil.mean() if len(sil) else np.nan
        n_rate = neg.mean() if len(neg) else np.nan
        # repair is worthwhile when silent cells are materially more positive
        # than explicit negatives AND there are enough of them to matter
        gap = (s_rate - n_rate) if (len(sil) and len(neg)) else np.nan
        fire = "YES" if (len(sil) >= 8 and not np.isnan(gap) and gap >= 0.15) else "no"
        rows.append((f, len(sil), s_rate, len(neg), n_rate, base, gap, fire))
        print("{0:18s} {1:3d} {2:>7s} {3:4d} {4:>8s} {5:9.2f} {6:>8s}".format(
            f, len(sil), ("%.2f" % s_rate) if len(sil) else "-",
            len(neg), ("%.2f" % n_rate) if len(neg) else "-",
            base, fire))

    print()
    print("== findings where repair should fire ==")
    for r in rows:
        if r[7] == "YES":
            print("  {0:18s} silent n={1}, silent_pos={2:.2f} vs expl_neg={3:.2f} "
                  "(gap {4:+.2f})".format(r[0], r[1], r[2], r[4], r[6]))

    print()
    print("== findings that look like THRESHOLD problems, not silence ==")
    print("(low AUC but report does mention them -- repair will not help these;")
    print(" they need better image features or magnitude calibration)")
    for r in rows:
        if r[7] == "no" and r[1] < 12:
            print("  {0:18s} silent only {1} cells".format(r[0], r[1]))


if __name__ == "__main__":
    main()
