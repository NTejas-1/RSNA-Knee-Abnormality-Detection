"""
RSNA Knee -- preprocessing cache builder (Kaggle script kernel, CPU only).

Produces a compact uint8 corpus so that model training never needs the 570GB of
DICOMs. One shard per kernel run; shards are published as private Kaggle datasets
and attached to the training kernels.

Geometry choices, and why they differ from the public 0.940 stack:

  crop 140mm  -- the public stack uses 130mm. Parag (16th) reports the sweet spot
                 is 140-160mm, Dread Development uses 140mm, Tom Aindow 150mm.
                 130mm is below everyone's optimum and clips useful anatomy.
  256px       -- stored slightly above the 224 training resolution so random
                 cropping is available as augmentation. Resolution above 288 is
                 reported to HURT (Tucker, tennogh; Less scored 0.937 at 384),
                 and staying low is what makes Kaggle-only compute viable.
  band 6-94%  -- Dread Development measured 44 slices over 2-98% at 0.917 against
                 44 over 6-94% at 0.926, so at fixed slice count the NARROWER band
                 wins by 0.009: spreading the same slices thinner costs accuracy.
                 The smoke-test montage corroborated this directly -- at 2% the
                 sagittal and coronal slices fall outside the joint and carry no
                 anatomy. 6-94% keeps the same density over useful tissue only.
  6 slots     -- plane x fluid_sensitive, taken from the PROVIDED metadata columns
                 rather than re-derived from SeriesDescription regexes as the
                 reference notebook does. All 4407 studies populate all 3 planes.

Slice ordering uses ImagePositionPatient projected onto the dominant slice axis,
falling back to InstanceNumber then filename -- the same approach the reference
notebook uses, because raw filename order is NOT anatomical order.

Laterality is normalised so every knee presents as Left: right knees are flipped
horizontally for Coronal/Axial and along the slice axis for Sagittal.
"""
import os
import re
import sys
import time
import traceback
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- config

SHARD = int(os.environ.get("RSNA_SHARD", "0"))
N_SHARDS = int(os.environ.get("RSNA_N_SHARDS", "6"))

COMP_CANDIDATES = [
    "/kaggle/input/competitions/rsna-knee-abnormality-detection",
    "/kaggle/input/rsna-knee-abnormality-detection",
]
OUT = "/kaggle/working"

CROP_MM = 140.0
SIZE = 256
N_SLICE = 16
BAND = (0.06, 0.94)
SPLIT = "train_series"

# (plane, fluid_sensitive) -- slot order is fixed and must match at train time
SLOTS = [
    ("Sagittal", 1), ("Sagittal", 0),
    ("Coronal", 1), ("Coronal", 0),
    ("Axial", 1), ("Axial", 0),
]
N_SLOT = len(SLOTS)

THREADS = int(os.environ.get("RSNA_THREADS", "8"))


def log(msg):
    print("[{0:7.1f}s] {1}".format(time.time() - T0, msg), flush=True)


T0 = time.time()


def find_comp():
    """Locate the competition mount. Script and notebook kernels differ, so probe
    both known layouts and fail loudly with a listing rather than silently."""
    for c in COMP_CANDIDATES:
        if os.path.isfile(os.path.join(c, "train_series.csv")):
            return c
    listing = []
    for base in ("/kaggle/input", "/kaggle/input/competitions"):
        if os.path.isdir(base):
            try:
                listing.append("{0} -> {1}".format(base, sorted(os.listdir(base))[:40]))
            except Exception as exc:
                listing.append("{0} -> unreadable: {1}".format(base, exc))
    raise SystemExit("competition data not found.\n" + "\n".join(listing))

# ---------------------------------------------------------------- dicom io

import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from concurrent.futures import ThreadPoolExecutor

try:
    import cv2
    def resize(a, n):
        return cv2.resize(a, (n, n), interpolation=cv2.INTER_AREA)
except Exception:
    from PIL import Image
    def resize(a, n):
        return np.asarray(Image.fromarray(a).resize((n, n), Image.BILINEAR))


def read_header(path):
    """Header-only read; cheap enough to do for every slice."""
    try:
        return pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception:
        return None


def natural_key(name):
    return tuple(int(x) if x.isdigit() else x.lower()
                 for x in re.split(r"(\d+)", str(name)))


def order_slices(series_dir):
    """Return filenames in anatomical order, plus pixel spacing and laterality.

    Filename order is not anatomical. Prefer ImagePositionPatient projected onto
    the dominant varying axis; fall back to InstanceNumber, then natural sort.
    """
    names = [e.name for e in os.scandir(series_dir) if e.name.endswith(".dcm")]
    if not names:
        return [], None, None
    names.sort(key=natural_key)

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        hdrs = list(pool.map(
            lambda n: read_header(os.path.join(series_dir, n)), names))

    pos, inst, px, lat = [], [], None, None
    for h in hdrs:
        if h is None:
            pos.append(None)
            inst.append(None)
            continue
        ipp = getattr(h, "ImagePositionPatient", None)
        pos.append([float(v) for v in ipp] if ipp is not None and len(ipp) >= 3 else None)
        iv = getattr(h, "InstanceNumber", None)
        inst.append(float(iv) if iv is not None else None)
        if px is None:
            ps = getattr(h, "PixelSpacing", None)
            if ps is not None and len(ps) >= 1:
                try:
                    px = float(ps[0])
                except Exception:
                    px = None
        if lat is None:
            lv = getattr(h, "Laterality", None) or getattr(h, "ImageLaterality", None)
            if lv:
                s = str(lv).strip().upper()
                if s and s[0] in ("L", "R"):
                    lat = s[0]

    have = [p for p in pos if p is not None]
    idx = list(range(len(names)))
    if len(have) >= max(3, len(names) // 2):
        arr = np.array([p if p is not None else [np.nan] * 3 for p in pos], float)
        spread = np.nanmax(arr, axis=0) - np.nanmin(arr, axis=0)
        axis = int(np.nanargmax(spread))
        key = arr[:, axis]
        # unresolved slices sink to the end rather than corrupting the ordering
        key = np.where(np.isfinite(key), key, np.inf)
        idx = list(np.argsort(key, kind="stable"))
    elif any(v is not None for v in inst):
        key = np.array([v if v is not None else np.inf for v in inst], float)
        idx = list(np.argsort(key, kind="stable"))

    # geometric laterality fallback: median x of the image corner
    if lat is None and len(have) >= 3:
        mx = float(np.median([p[0] for p in have]))
        if abs(mx) >= 20.0:
            lat = "R" if mx < 0 else "L"

    return [names[i] for i in idx], px, lat


def sample_positions(n, k, band):
    """k evenly spaced indices across the band, deduplicated, clipped in range."""
    if n <= 0:
        return []
    lo = int(round(band[0] * (n - 1)))
    hi = int(round(band[1] * (n - 1)))
    if hi <= lo:
        return [min(max(lo, 0), n - 1)] * k
    raw = np.linspace(lo, hi, k)
    return [int(min(max(round(v), 0), n - 1)) for v in raw]


def load_slice(path, px, crop_mm, size):
    """Decode one slice, centre-crop to a physical FOV, resize, scale to uint8."""
    ds = pydicom.dcmread(path, force=True)
    a = ds.pixel_array
    try:
        a = apply_modality_lut(a, ds)
    except Exception:
        pass
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., 0] if a.shape[-1] <= 4 else a[a.shape[0] // 2]

    if px and px > 0:
        half = int(round(crop_mm / px / 2.0))
        cy, cx = a.shape[0] // 2, a.shape[1] // 2
        y0, y1 = max(0, cy - half), min(a.shape[0], cy + half)
        x0, x1 = max(0, cx - half), min(a.shape[1], cx + half)
        if y1 - y0 >= 8 and x1 - x0 >= 8:
            a = a[y0:y1, x0:x1]

    # per-slice robust scaling; MRI intensities are not comparable across series
    lo, hi = np.percentile(a, 1.0), np.percentile(a, 99.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros((size, size), np.uint8)
    a = np.clip((a - lo) / (hi - lo), 0, 1)
    return resize((a * 255.0).astype(np.uint8), size)


def build_slot(series_dir, plane):
    """One slot -> (N_SLICE, SIZE, SIZE) uint8, laterality-normalised."""
    names, px, lat = order_slices(series_dir)
    if not names:
        return None
    pos = sample_positions(len(names), N_SLICE, BAND)
    uniq = sorted(set(pos))

    def one(i):
        try:
            return i, load_slice(os.path.join(series_dir, names[i]), px,
                                 CROP_MM, SIZE)
        except Exception:
            return i, None

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        got = dict(pool.map(one, uniq))

    fill = next((v for v in got.values() if v is not None), None)
    if fill is None:
        return None
    vol = np.stack([got.get(i) if got.get(i) is not None else fill for i in pos])

    if lat == "R":
        # present every knee as Left
        vol = vol[:, :, ::-1] if plane in ("Coronal", "Axial") else vol[::-1]
    return np.ascontiguousarray(vol)


# ---------------------------------------------------------------- main

def main():
    COMP = find_comp()
    log("competition root: " + COMP)
    ser = pd.read_csv(os.path.join(COMP, "train_series.csv"))
    studies = sorted(ser["StudyInstanceUID"].unique())
    mine = [s for i, s in enumerate(studies) if i % N_SHARDS == SHARD]
    log("shard {0}/{1}: {2} of {3} studies".format(
        SHARD, N_SHARDS, len(mine), len(studies)))

    # deepest series wins each slot -- more slices means finer sampling
    ser = ser.sort_values("StudyInstanceUID")
    by_study = {}
    for st, g in ser.groupby("StudyInstanceUID"):
        by_study[st] = g

    n = len(mine)
    cache = np.lib.format.open_memmap(
        os.path.join(OUT, "cache_shard{0}.npy".format(SHARD)),
        mode="w+", dtype=np.uint8, shape=(n, N_SLOT, N_SLICE, SIZE, SIZE))
    present = np.zeros((n, N_SLOT), np.uint8)

    root = os.path.join(COMP, SPLIT)
    for k, st in enumerate(mine):
        g = by_study.get(st)
        if g is None:
            continue
        # count slices per series once so the deepest can be chosen per slot
        depth = {}
        for sid in g["SeriesInstanceUID"]:
            d = os.path.join(root, st, str(sid))
            try:
                depth[sid] = sum(1 for e in os.scandir(d) if e.name.endswith(".dcm"))
            except Exception:
                depth[sid] = 0

        for si, (plane, fluid) in enumerate(SLOTS):
            cand = g[(g["Anatomical_Plane"] == plane) &
                     (g["Fluid_Sensitive"] == fluid)]
            if not len(cand):
                continue
            sid = max(cand["SeriesInstanceUID"], key=lambda s: depth.get(s, 0))
            if depth.get(sid, 0) == 0:
                continue
            try:
                vol = build_slot(os.path.join(root, st, str(sid)), plane)
            except Exception:
                log("slot fail {0} {1}: {2}".format(st[-12:], plane,
                                                    traceback.format_exc(limit=1)))
                vol = None
            if vol is not None:
                cache[k, si] = vol
                present[k, si] = 1

        if (k + 1) % 25 == 0:
            rate = (time.time() - T0) / (k + 1)
            log("{0}/{1} studies  {2:.2f}s/study  eta {3:.0f}min  "
                "slot fill {4:.1%}".format(
                    k + 1, n, rate, rate * (n - k - 1) / 60,
                    present[:k + 1].mean()))

    cache.flush()
    np.save(os.path.join(OUT, "present_shard{0}.npy".format(SHARD)), present)
    pd.DataFrame({"StudyInstanceUID": mine}).to_csv(
        os.path.join(OUT, "studies_shard{0}.csv".format(SHARD)), index=False)

    log("DONE shard {0}: {1} studies, slot fill {2:.2%}, {3:.2f} GB".format(
        SHARD, n, present.mean(),
        n * N_SLOT * N_SLICE * SIZE * SIZE / 1e9))
    # a fully empty study would silently poison training -- surface it loudly
    empty = int((present.sum(axis=1) == 0).sum())
    if empty:
        log("WARNING: {0} studies have NO usable slots".format(empty))


if __name__ == "__main__":
    main()
