"""CPU probe: what columns actually exist on train_series.csv vs test_series.csv.

Reason this exists: the 6-slot cache was built keyed on Fluid_Sensitive, but the
reference notebook treats Fat_Suppression as required on test_series.csv and
Fluid_Sensitive as possibly absent (cell 4 guards `if 'Fluid_Sensitive' in _meta`,
cell 17 slots on Fat_Suppression). If Fluid_Sensitive is missing at test time then
our slots cannot be rebuilt the way they were trained, which is silent train/test
skew -- the model would get systematically different inputs and nobody would see
an error. Verify before writing any inference code.

Also reports how well the two flags agree on train, since Fat_Suppression is the
only available fallback if Fluid_Sensitive really is test-absent.
"""
import os
import pandas as pd


def find_comp():
    for c in ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
              "/kaggle/input/rsna-knee-abnormality-detection"]:
        if os.path.isdir(c):
            return c
    for dirpath, _d, files in os.walk("/kaggle/input"):
        if "test_series.csv" in files:
            return dirpath
    raise SystemExit("competition data not found")


COMP = find_comp()
print("competition root:", COMP)
print("top-level files:", sorted(os.listdir(COMP))[:25])
print()

for name in ("train_series.csv", "test_series.csv"):
    p = os.path.join(COMP, name)
    if not os.path.exists(p):
        print("{0}: ABSENT".format(name))
        continue
    d = pd.read_csv(p)
    print("=" * 60)
    print("{0}: {1} rows".format(name, len(d)))
    print("columns:", list(d.columns))
    for col in ("Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"):
        if col in d.columns:
            vc = d[col].value_counts(dropna=False).to_dict()
            print("  {0}: {1}  nulls={2}".format(col, vc, int(d[col].isna().sum())))
        else:
            print("  {0}: COLUMN ABSENT".format(col))

    if "Fluid_Sensitive" in d.columns and "Fat_Suppression" in d.columns:
        sub = d.dropna(subset=["Fluid_Sensitive", "Fat_Suppression"])
        agree = (sub["Fluid_Sensitive"].astype(int) ==
                 sub["Fat_Suppression"].astype(int)).mean()
        print("  Fluid_Sensitive == Fat_Suppression on {0} rows: {1:.4f}".format(
            len(sub), agree))
        print("  cross-tab:")
        print(pd.crosstab(sub["Fluid_Sensitive"], sub["Fat_Suppression"]))
    print()

# how many of our 6 slots would actually be fillable on test
p = os.path.join(COMP, "test_series.csv")
if os.path.exists(p):
    d = pd.read_csv(p)
    key = "Fluid_Sensitive" if "Fluid_Sensitive" in d.columns else "Fat_Suppression"
    print("slot key usable at test time:", key)
    if "Anatomical_Plane" in d.columns and key in d.columns:
        n_st = d["StudyInstanceUID"].nunique()
        filled = 0
        for plane in ("Sagittal", "Coronal", "Axial"):
            for v in (1, 0):
                m = (d["Anatomical_Plane"] == plane) & (d[key] == v)
                c = d.loc[m, "StudyInstanceUID"].nunique()
                filled += c
                print("  slot ({0},{1}): {2} of {3} studies".format(
                    plane, v, c, n_st))
        print("  overall slot fill: {0:.2%}  (training corpus was 80.68%)".format(
            filled / float(max(n_st * 6, 1))))
