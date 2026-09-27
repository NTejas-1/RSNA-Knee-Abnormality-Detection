"""Correct test of where label repair should fire.

My first test (silent vs explicit-negative gold rate) was wrong: it needs a decent
number of explicit negatives, and Synovitis has only 3. It also ignores what the
LLM actually predicted.

The right diagnostic is a calibration gap, per finding, per state:

    gap = P(gold positive | state) - mean(LLM prob | state)

A large POSITIVE gap means the LLM systematically under-predicts those cells, so
they are recoverable false negatives -> repair with out-of-fold image predictions.
A gap near zero means the LLM is already calibrated there and repair only adds
noise. A large NEGATIVE gap means over-prediction.

n_pos_recoverable quantifies the prize: how many study-label cells are actually
positive but sit below 0.5 in the extracted labels.
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
STATE_NAME = {"p": "positive", "n": "expl_neg", "u": "SILENT"}


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

    print("== calibration gap on SILENT cells (the repair target) ==")
    print("gap = true positive rate  minus  mean LLM probability")
    print()
    print("{0:18s} {1:>4s} {2:>8s} {3:>8s} {4:>7s} {5:>6s} {6:>8s}".format(
        "finding", "n", "true_pos", "llm_prob", "gap", "recov", "verdict"))
    print("-" * 70)

    total_recov = 0
    fire = []
    for f in ORDER:
        sub = df[(df.FINDING == f) & (df.STATE == "u")]
        if len(sub) < 5:
            print("{0:18s} {1:4d} {2:>8s}".format(f, len(sub), "too few"))
            continue
        true_pos = sub["gold"].mean()
        llm = sub["PROB"].mean()
        gap = true_pos - llm
        recov = int(((sub["gold"] == 1) & (sub["PROB"] < 0.5)).sum())
        total_recov += recov
        if gap >= 0.12 and recov >= 4:
            verdict = "REPAIR"
            fire.append((f, len(sub), true_pos, llm, gap, recov))
        elif gap <= 0.05:
            verdict = "calibrated"
        else:
            verdict = "marginal"
        print("{0:18s} {1:4d} {2:8.2f} {3:8.2f} {4:+7.2f} {5:6d} {6:>8s}".format(
            f, len(sub), true_pos, llm, gap, recov, verdict))

    print()
    print("total recoverable false negatives on silent cells:", total_recov,
          "of", len(df), "cells ({0:.1f}%)".format(100 * total_recov / len(df)))

    print()
    print("== REPAIR THESE ==")
    for f, n, tp, llm, gap, recov in sorted(fire, key=lambda r: -r[4]):
        print("  {0:18s} n={1:3d}  true={2:.2f} vs llm={3:.2f}  gap={4:+.2f}  "
              "{5} recoverable".format(f, n, tp, llm, gap, recov))

    print()
    print("== DO NOT REPAIR (silence genuinely means absent) ==")
    for f in ORDER:
        sub = df[(df.FINDING == f) & (df.STATE == "u")]
        if len(sub) >= 5:
            gap = sub["gold"].mean() - sub["PROB"].mean()
            if gap <= 0.05:
                print("  {0:18s} n={1:3d}  true_pos={2:.2f} -- repair would add "
                      "noise".format(f, len(sub), sub["gold"].mean()))


if __name__ == "__main__":
    main()
