"""Export extracted labels to a wide CSV for the Kaggle training kernel.

Two matrices per study:
  <finding>       fractional probability (the training target)
  <finding>_state p / n / u, so the trainer knows which cells are SILENT and
                  therefore eligible for out-of-fold repair later.

Keeping state alongside the probability is what makes targeted repair possible;
a plain label matrix throws that information away.
"""
import io
import os
import subprocess
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "labels_ds")
CONN = "oruwqcb-kaggle"
RUN_TAG = "v1_opus45"

ORDER = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
         "Medial OA", "Lateral OA", "PF OA", "Effusion",
         "Synovitis", "Baker's", "Contusion", "Fracture"]


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
    os.makedirs(OUT, exist_ok=True)
    df = fetch(
        "SELECT STUDY_UID, FINDING, STATE, PROB "
        "FROM RSNA_KNEE.LABELS.EXTRACTED_LABELS WHERE RUN_TAG = '{0}'".format(RUN_TAG)
    )
    print("long rows:", len(df))

    prob = df.pivot(index="STUDY_UID", columns="FINDING", values="PROB")
    state = df.pivot(index="STUDY_UID", columns="FINDING", values="STATE")
    prob = prob[ORDER]
    state = state[ORDER]
    state.columns = [c + "_state" for c in ORDER]

    gold = fetch(
        "SELECT STUDY_UID, IS_GOLD, GOLD_ACL, GOLD_MCL, GOLD_MEDIAL_MENISCUS, "
        "GOLD_LATERAL_MENISCUS, GOLD_MEDIAL_OA, GOLD_LATERAL_OA, GOLD_PF_OA, "
        "GOLD_EFFUSION, GOLD_SYNOVITIS, GOLD_BAKERS, GOLD_CONTUSION, GOLD_FRACTURE "
        "FROM RSNA_KNEE.LABELS.REPORTS"
    ).set_index("STUDY_UID")

    wide = prob.join(state).join(gold)
    wide = wide.reset_index()
    path = os.path.join(OUT, "labels_v1_opus45.csv")
    wide.to_csv(path, index=False)
    print("wrote", path, os.path.getsize(path), "bytes;", len(wide), "studies")
    print("columns:", len(wide.columns))
    print(wide[ORDER].describe().loc[["mean", "min", "max"]].round(3).to_string())

    meta = {
        "title": "RSNA Knee Opus Labels v1",
        "id": "tejuuss/rsna-knee-opus-labels-v1",
        "licenses": [{"name": "other"}],
    }
    import json
    with open(os.path.join(OUT, "dataset-metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print("dataset metadata written; push with:")
    print("  kaggle datasets create -p labels_ds --dir-mode zip")


if __name__ == "__main__":
    main()
