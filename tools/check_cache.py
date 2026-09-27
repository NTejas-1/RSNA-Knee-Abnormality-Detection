"""Validate cache content: are the volumes real images, correctly cropped, ordered
and laterality-normalised? A silently blank or scrambled cache would waste days of
training, so check before scaling up.
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "smoke_out")
SHARD = sys.argv[2] if len(sys.argv) > 2 else "0"

SLOTS = ["SAG_fluid", "SAG_struct", "COR_fluid", "COR_struct", "AX_fluid", "AX_struct"]

cache = np.load(os.path.join(D, "cache_shard{0}.npy".format(SHARD)), mmap_mode="r")
present = np.load(os.path.join(D, "present_shard{0}.npy".format(SHARD)))
studies = pd.read_csv(os.path.join(D, "studies_shard{0}.csv".format(SHARD)))

print("cache shape:", cache.shape, "dtype:", cache.dtype)
print("studies:", len(studies), "| present shape:", present.shape)
print("overall slot fill: {0:.1%}".format(present.mean()))
print()

print("== slot fill by slot ==")
for i, nm in enumerate(SLOTS):
    print("  {0:12s} {1:5.1%}".format(nm, present[:, i].mean()))

print()
print("== content sanity per slot (over present slots only) ==")
print("{0:12s} {1:>8s} {2:>8s} {3:>8s} {4:>9s} {5:>8s}".format(
    "slot", "mean", "std", "blank%", "dup_slice%", "p99"))
for i, nm in enumerate(SLOTS):
    idx = np.where(present[:, i] == 1)[0]
    if not len(idx):
        print("  {0:12s} absent".format(nm))
        continue
    v = np.asarray(cache[idx, i], dtype=np.float32)
    # blank = a slice with almost no dynamic range
    per_slice_std = v.std(axis=(2, 3))
    blank = float((per_slice_std < 2.0).mean())
    # duplicate slices indicate the sampler collapsed (too few source slices)
    dup = 0.0
    for k in range(v.shape[0]):
        s = v[k]
        d = sum(1 for a in range(1, s.shape[0])
                if np.array_equal(s[a], s[a - 1]))
        dup += d / max(s.shape[0] - 1, 1)
    dup /= v.shape[0]
    print("{0:12s} {1:8.1f} {2:8.1f} {3:7.1%} {4:8.1%} {5:8.0f}".format(
        nm, v.mean(), v.std(), blank, dup, np.percentile(v, 99)))

print()
print("== per-study check ==")
empty = int((present.sum(axis=1) == 0).sum())
print("studies with zero usable slots:", empty)
allstd = []
for k in range(min(len(studies), 12)):
    filled = present[k].sum()
    v = np.asarray(cache[k][present[k] == 1], dtype=np.float32)
    allstd.append(v.std())
    print("  study {0}: {1}/6 slots, mean {2:6.1f}, std {3:5.1f}".format(
        k, filled, v.mean(), v.std()))

print()
if np.mean(allstd) < 5:
    print("FAIL: volumes have almost no contrast -- decode or scaling is broken")
elif empty:
    print("WARN: some studies have no slots at all")
else:
    print("PASS: cache contains real, varied image data")

# dump a montage so the geometry can be eyeballed
try:
    from PIL import Image
    k = int(np.argmax(present.sum(axis=1)))
    tiles = []
    for i in range(6):
        if present[k, i]:
            row = np.concatenate([np.asarray(cache[k, i, j]) for j in (0, 5, 10, 15)], axis=1)
        else:
            row = np.zeros((cache.shape[3], cache.shape[4] * 4), np.uint8)
        tiles.append(row)
    img = np.concatenate(tiles, axis=0)
    p = os.path.join(D, "montage_shard{0}.png".format(SHARD))
    Image.fromarray(img).save(p)
    print("montage written:", p, "(rows = 6 slots, cols = slices 0/5/10/15)")
except Exception as exc:
    print("montage skipped:", exc)
