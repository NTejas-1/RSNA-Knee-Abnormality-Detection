"""Resumable Cortex label extraction.

Idempotent by (STUDY_UID, RUN_TAG): each batch only picks up studies not already
extracted for this run tag, so it can be re-run after any failure without paying
for inference twice.

Usage:
    python tools/extract_labels.py --gold-only
    python tools/extract_labels.py --all --batch 400
"""
import argparse
import os
import re
import subprocess
import sys
import time

CONN = "oruwqcb-kaggle"
MODEL = "claude-opus-4-5"
RUN_TAG = "v1_opus45"

PRE = "USE DATABASE RSNA_KNEE; USE SCHEMA LABELS; "


def sf(sql, timeout=3600):
    p = subprocess.run(
        ["snow", "sql", "-c", CONN, "-q", sql],
        capture_output=True, text=True, shell=True, timeout=timeout,
    )
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def scalar(sql, col):
    rc, out = sf(sql)
    m = re.findall(r"\|\s*(-?[\d.]+)\s*\|", out)
    return float(m[-1]) if m else None


def count_remaining(gold_only):
    where = "WHERE IS_GOLD " if gold_only else "WHERE 1=1 "
    q = (PRE + "SELECT COUNT(*) AS n FROM PROMPTS_V1 p " + where +
         "AND NOT EXISTS (SELECT 1 FROM EXTRACTED_RAW r "
         "WHERE r.STUDY_UID = p.STUDY_UID AND r.RUN_TAG = '{0}')".format(RUN_TAG))
    return scalar(q, "N")


def run_batch(n, gold_only):
    where = "WHERE IS_GOLD " if gold_only else "WHERE 1=1 "
    sql = (
        PRE +
        "INSERT INTO EXTRACTED_RAW (STUDY_UID, RUN_TAG, MODEL, RAW_JSON) "
        "SELECT p.STUDY_UID, '{tag}', '{model}', "
        "AI_COMPLETE('{model}', p.PROMPT, "
        "{{'temperature':0, 'max_tokens':900}}) "
        "FROM PROMPTS_V1 p " + where +
        "AND NOT EXISTS (SELECT 1 FROM EXTRACTED_RAW r "
        "WHERE r.STUDY_UID = p.STUDY_UID AND r.RUN_TAG = '{tag}') "
        "LIMIT {n}"
    ).format(tag=RUN_TAG, model=MODEL, n=n)
    t0 = time.time()
    rc, out = sf(sql)
    return rc, time.time() - t0, out


def parse_into_long():
    findings = [
        ("ACL", "ACL"), ("MCL", "MCL"),
        ("MedialMeniscus", "Medial Meniscus"), ("LateralMeniscus", "Lateral Meniscus"),
        ("MedialOA", "Medial OA"), ("LateralOA", "Lateral OA"), ("PFOA", "PF OA"),
        ("Effusion", "Effusion"), ("Synovitis", "Synovitis"), ("Bakers", "Baker's"),
        ("Contusion", "Contusion"), ("Fracture", "Fracture"),
    ]
    sel = " UNION ALL ".join(
        "SELECT STUDY_UID, RUN_TAG, '{lab}' AS FINDING, "
        "j:\"{key}\":\"s\"::VARCHAR AS STATE, "
        "j:\"{key}\":\"p\"::FLOAT AS PROB FROM parsed".format(
            key=k, lab=l.replace("'", "''"))
        for k, l in findings
    )
    sql = (
        PRE +
        "CREATE OR REPLACE TEMPORARY VIEW parsed AS "
        "SELECT STUDY_UID, RUN_TAG, TRY_PARSE_JSON(RAW_JSON) AS j "
        "FROM EXTRACTED_RAW WHERE RUN_TAG = '{tag}'; "
        "DELETE FROM EXTRACTED_LABELS WHERE RUN_TAG = '{tag}'; "
        "INSERT INTO EXTRACTED_LABELS (STUDY_UID, RUN_TAG, FINDING, STATE, PROB) "
        "{sel};"
    ).format(tag=RUN_TAG, sel=sel)
    return sf(sql)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-only", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--max-batches", type=int, default=100)
    a = ap.parse_args()
    gold_only = a.gold_only and not a.all

    rem = count_remaining(gold_only)
    print("remaining to extract:", rem, "| gold_only:", gold_only, flush=True)
    if not rem:
        print("nothing to do")
    for i in range(a.max_batches):
        rem = count_remaining(gold_only)
        if not rem:
            print("all extracted")
            break
        n = int(min(a.batch, rem))
        rc, dt, out = run_batch(n, gold_only)
        if rc != 0:
            print("batch failed:", out[-1500:])
            sys.exit(1)
        print("batch {0}: {1} rows in {2:.0f}s ({3:.2f}s/report)".format(
            i + 1, n, dt, dt / max(n, 1)), flush=True)

    print("parsing to long form...", flush=True)
    rc, out = parse_into_long()
    print(out[-1200:])


if __name__ == "__main__":
    main()
