"""Verify the 4 cache shards in place on Kaggle, without downloading 28GB.

Resolves inputs by searching rather than assuming a mount layout, and prints the
tree so the real layout is recorded.
"""
import os
import numpy as np
import pandas as pd

ROOT = "/kaggle/input"
SLOTS = ["SAG_fluid", "SAG_struct", "COR_fluid", "COR_struct", "AX_fluid", "AX_struct"]


def print_tree(max_depth=3):
    print("== /kaggle/input tree (depth {0}) ==".format(max_depth))
    for dirpath, dirnames, filenames in os.walk(ROOT):
        depth = dirpath[len(ROOT):].count(os.sep)
        if depth >= max_depth:
            dirnames[:] = []
        pad = "  " * depth
        print("{0}{1}/".format(pad, os.path.basename(dirpath) or ROOT))
        for f in sorted(filenames)[:8]:
            print("{0}  {1}".format(pad, f))


def find_file(filename):
    for dirpath, _d, filenames in os.walk(ROOT):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None


print_tree()

total, all_present, uids, shapes = 0, [], [], []
for i in range(4):
    print()
    print("== shard", i, "==")
    cp = find_file("cache_shard{0}.npy".format(i))
    if cp is None:
        print("  MISSING cache_shard{0}.npy".format(i))
        continue
    d = os.path.dirname(cp)
    print("  found at:", d)
    c = np.load(cp, mmap_mode="r")
    p = np.load(os.path.join(d, "present_shard{0}.npy".format(i)))
    s = pd.read_csv(os.path.join(d, "studies_shard{0}.csv".format(i)))
    shapes.append(tuple(c.shape[1:]))
    print("  shape {0} dtype {1} {2:.2f} GB".format(
        c.shape, c.dtype, float(np.prod(c.shape)) / 1e9))
    print("  studies {0}  slot fill {1:.2%}  zero-slot studies {2}".format(
        len(s), p.mean(), int((p.sum(axis=1) == 0).sum())))
    for j, nm in enumerate(SLOTS):
        print("     {0:12s} {1:6.1%}".format(nm, p[:, j].mean()))
    idx = np.linspace(0, len(s) - 1, min(5, len(s))).astype(int)
    v = np.asarray(c[idx], np.float32)
    print("  sample mean {0:.1f} std {1:.1f} blank_slice% {2:.2%}".format(
        v.mean(), v.std(), float((v.std(axis=(3, 4)) < 2.0).mean())))
    total += len(s)
    all_present.append(p)
    uids += [str(u) for u in s["StudyInstanceUID"]]

print()
print("== corpus totals ==")
print("studies across shards:", total, "| unique:", len(set(uids)),
      "| duplicates:", total - len(set(uids)))
if all_present:
    P = np.concatenate(all_present)
    print("overall slot fill: {0:.2%}".format(P.mean()))
    print("studies with zero slots:", int((P.sum(axis=1) == 0).sum()))
print("per-study shapes consistent:", len(set(shapes)) == 1, shapes[:1])

lp = find_file("labels_v1_opus45.csv")
print()
print("labels csv:", lp)
if lp:
    lab = pd.read_csv(lp)
    lab["STUDY_UID"] = lab["STUDY_UID"].astype(str)
    inter = len(set(uids) & set(lab["STUDY_UID"]))
    print("label rows:", len(lab), "| matched to cache:", inter,
          "| MATCH OK" if inter == len(set(uids)) else "| MISMATCH")

rp = find_file("ResNet50.pt")
print("RadImageNet:", rp, os.path.getsize(rp) if rp else "")

print()
if total == 4407 and all_present and int(np.concatenate(all_present).sum(axis=1).min()) > 0:
    print("VERDICT: cache complete and usable")
else:
    print("VERDICT: incomplete -- {0} of 4407 studies".format(total))
