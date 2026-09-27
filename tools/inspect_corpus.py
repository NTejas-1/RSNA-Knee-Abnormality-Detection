"""Characterise the RSNA knee training corpus: labels, reports, series geometry."""
import os
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = os.path.join(HERE, "meta")

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

tr = pd.read_csv(os.path.join(META, "train.csv"))
ts = pd.read_csv(os.path.join(META, "train_series.csv"))
te = pd.read_csv(os.path.join(META, "test.csv"))

print("== corpus size ==")
print("train studies :", len(tr))
print("train series  :", len(ts))
print("series/study  :", round(len(ts) / len(tr), 2))
print("test studies  :", len(te))
print("train cols    :", list(tr.columns))

lab = tr[LABELS]
print()
print("== label coverage ==")
print("rows with ANY label :", int(lab.notna().any(axis=1).sum()))
print("rows fully labelled :", int(lab.notna().all(axis=1).sum()))

print()
print("== positives among labelled (the 'gold' set) ==")
for c in LABELS:
    s = lab[c].dropna()
    if len(s):
        print("  {0:18s} n={1:4d} pos={2:3d} rate={3:.3f}".format(
            c, len(s), int(s.sum()), s.mean()))
    else:
        print("  {0:18s} none".format(c))

print()
print("== reports ==")
ln = tr["Report"].astype(str).str.len()
print("null reports:", int(tr["Report"].isna().sum()))
print("chars  min/med/mean/max:", ln.min(), int(ln.median()), int(ln.mean()), ln.max())
print("approx total chars:", int(ln.sum()))
print("approx tokens (chars/3.5):", int(ln.sum() / 3.5))

print()
print("== series geometry ==")
print(ts["Anatomical_Plane"].value_counts().to_string())
print("Fluid_Sensitive == Fat_Suppression everywhere:",
      bool((ts["Fluid_Sensitive"] == ts["Fat_Suppression"]).all()))
print()
print("plane x fluid slot populations (our 6 slots):")
g = ts.groupby(["Anatomical_Plane", "Fluid_Sensitive"]).size()
print(g.to_string())

print()
print("series per study distribution:")
print(ts.groupby("StudyInstanceUID").size().describe().to_string())

print()
print("studies missing a given plane entirely:")
have = ts.groupby(["StudyInstanceUID", "Anatomical_Plane"]).size().unstack(fill_value=0)
for plane in ["Sagittal", "Coronal", "Axial"]:
    if plane in have.columns:
        print("  no {0:9s}: {1}".format(plane, int((have[plane] == 0).sum())))
