"""Load the 4407 train reports + 58 gold label rows into RSNA_KNEE.LABELS.REPORTS.

Uses parquet via stage because report text contains embedded newlines, commas and
quotes that make CSV round-tripping fragile.
"""
import os
import subprocess
import sys
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = os.path.join(HERE, "meta")
WORK = os.path.join(HERE, "work")
CONN = "oruwqcb-kaggle"

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

GOLD_COL = {
    "ACL": "GOLD_ACL",
    "MCL": "GOLD_MCL",
    "Medial Meniscus": "GOLD_MEDIAL_MENISCUS",
    "Lateral Meniscus": "GOLD_LATERAL_MENISCUS",
    "Medial OA": "GOLD_MEDIAL_OA",
    "Lateral OA": "GOLD_LATERAL_OA",
    "PF OA": "GOLD_PF_OA",
    "Effusion": "GOLD_EFFUSION",
    "Synovitis": "GOLD_SYNOVITIS",
    "Baker's": "GOLD_BAKERS",
    "Contusion": "GOLD_CONTUSION",
    "Fracture": "GOLD_FRACTURE",
}


def sf(sql, expect_ok=True):
    p = subprocess.run(
        ["snow", "sql", "-c", CONN, "-q", sql],
        capture_output=True, text=True, shell=True,
    )
    out = (p.stdout or "") + (p.stderr or "")
    if expect_ok and p.returncode != 0:
        print("SQL FAILED:", sql[:200])
        print(out[-2000:])
        sys.exit(1)
    return out


def main():
    os.makedirs(WORK, exist_ok=True)
    tr = pd.read_csv(os.path.join(META, "train.csv"))

    out = pd.DataFrame({
        "STUDY_UID": tr["StudyInstanceUID"].astype(str),
        "REPORT": tr["Report"].astype(str),
    })
    out["REPORT_CHARS"] = out["REPORT"].str.len()
    for lab, col in GOLD_COL.items():
        out[col] = tr[lab]
    out["IS_GOLD"] = tr[LABELS].notna().all(axis=1)

    pq = os.path.join(WORK, "reports.parquet")
    out.to_parquet(pq, index=False)
    print("wrote", pq, os.path.getsize(pq), "bytes;", len(out), "rows;",
          int(out["IS_GOLD"].sum()), "gold")

    stage_path = pq.replace("\\", "/")
    print(sf("USE DATABASE RSNA_KNEE; USE SCHEMA LABELS; "
             "PUT 'file://{0}' @RSNA_STAGE/reports/ OVERWRITE=TRUE "
             "AUTO_COMPRESS=FALSE".format(stage_path))[-800:])

    cols = ["STUDY_UID", "REPORT", "REPORT_CHARS"] + list(GOLD_COL.values()) + ["IS_GOLD"]
    select = ", ".join(
        "$1:{0}::{1}".format(c, "VARCHAR" if c in ("STUDY_UID", "REPORT")
                             else ("BOOLEAN" if c == "IS_GOLD" else "NUMBER"))
        for c in cols
    )
    copy_sql = (
        "USE DATABASE RSNA_KNEE; USE SCHEMA LABELS; "
        "TRUNCATE TABLE REPORTS; "
        "COPY INTO REPORTS ({0}) FROM (SELECT {1} FROM @RSNA_STAGE/reports/) "
        "FILE_FORMAT = (TYPE = PARQUET) ON_ERROR = ABORT_STATEMENT;"
    ).format(", ".join(cols), select)
    print(sf(copy_sql)[-1500:])

    print(sf("SELECT COUNT(*) AS rows, SUM(IFF(IS_GOLD,1,0)) AS gold, "
             "SUM(REPORT_CHARS) AS chars FROM RSNA_KNEE.LABELS.REPORTS")[-800:])


if __name__ == "__main__":
    main()
