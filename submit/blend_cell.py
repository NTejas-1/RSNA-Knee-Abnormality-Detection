# ===================================================================
# APPENDED ARM BLEND -- RadImageNet arm A + ImageNet arm B, per-label.
#
# Reads /kaggle/working/_pipeline_stage.csv, blends our arms in, writes it back.
# The next cell publishes that file to submission.csv after its integrity gates.
# Nothing above this point is modified: the 0.940 baseline and its gates
# (_DINOV2_MATCHED_MEMBERS == 20, V18_CALIBRATOR_APPLIED) are untouched.
#
# Blending is on RANK PERCENTILES per column, mirroring _a5_blend_available:
# column-wise ROC AUC only sees ordering, so ranks are lossless and need no
# calibration between two models with different output scales. Our arm is ranked
# WITHIN the subset of studies we could actually predict, so studies we skip do
# not distort the ranks of those we did.
#
# If ANY part of this fails the cell restores the untouched staging file and
# re-raises. A broken blend must not be allowed to corrupt a 0.940 submission.
# ===================================================================
import os, shutil, traceback, re, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

_BL_STAGE = "/kaggle/working/_pipeline_stage.csv"
_BL_BACKUP = "/kaggle/working/_pipeline_stage_preblend.csv"
_BL_CKPT_DIR = None          # set below by discovery
_BL_LABELS = list(_RSNA_LABELS)

# per-label weight on our arms, in _RSNA_LABELS order. Principled from the 5-fold
# OOF evaluation, NOT fitted -- we have no out-of-fold predictions for the parent
# model, so these cannot be optimised, only sanity-checked on the leaderboard.
# Max 0.35 keeps the parent senior on every label.
_BL_W = np.asarray([0.1, 0.04, 0.125, 0.09, 0.16, 0.075, 0.04, 0.19, 0.125, 0.14, 0.16, 0.15], dtype=np.float64)
assert len(_BL_W) == 12

def _bl_log(m):
    print("[blend] {0}".format(m), flush=True)

# phase timing: the first end-to-end run came out at 143s per study, which would be
# ~52h over 1300 studies against a 9h cap. Attributing that to decode vs GPU vs
# checkpoint loading decides which optimisation is worth doing.
import time as _bl_time
_bl_clock = _bl_time.time
_BL_T = {"load_ckpts": 0.0, "decode": 0.0, "forward": 0.0, "find": 0.0}

# ---- resampler spliced verbatim from cache/build_cache.py ----
try:
    import cv2
    def resize(a, n):
        return cv2.resize(a, (n, n), interpolation=cv2.INTER_AREA)
except Exception:
    from PIL import Image
    def resize(a, n):
        return np.asarray(Image.fromarray(a).resize((n, n), Image.BILINEAR))
_bl_log("resize backend: " + ("cv2 INTER_AREA" if "cv2" in sys.modules else "PIL BILINEAR -- SKEW if the cache was built with cv2"))

# ---- geometry spliced verbatim from cache/build_cache.py ----
CROP_MM = 140.0

SIZE = 256

N_SLICE = 16

BAND = (0.06, 0.94)

SLOTS = [
    ("Sagittal", 1), ("Sagittal", 0),
    ("Coronal", 1), ("Coronal", 0),
    ("Axial", 1), ("Axial", 0),
]

THREADS = int(os.environ.get("RSNA_THREADS", "8"))

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

# ---- model spliced verbatim from train/train_armA.py ----
IMG = 224

TRAIN_SLICES = 8

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
          "Medial OA", "Lateral OA", "PF OA", "Effusion",
          "Synovitis", "Baker's", "Contusion", "Fracture"]

N_LAB = len(LABELS)

class RadEncoder(nn.Module):
    """RadImageNet ResNet50 trunk.

    The checkpoint was saved from a module whose only attribute is
    `backbone = nn.Sequential(*list(resnet50().children())[:-2])`, so its keys are
    numeric Sequential indices ("backbone.0.weight"), NOT torchvision's named
    modules ("conv1.weight"). Rebuilding the identical structure is what makes a
    strict load possible -- renaming or prefix-stripping matches nothing.
    """

    def __init__(self):
        super().__init__()
        import torchvision
        self.backbone = nn.Sequential(
            *list(torchvision.models.resnet50(weights=None).children())[:-2])

    def forward(self, x):
        return self.backbone(x).mean(dim=(2, 3))

class Attn(nn.Module):
    """Gated attention pooling (Ilse et al). +0.014 over masked mean in this comp."""

    def __init__(self, dim, hidden=256):
        super().__init__()
        self.v = nn.Linear(dim, hidden)
        self.u = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, x, mask=None):
        # x (B, N, D)
        a = self.w(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x)))  # (B, N, 1)
        if mask is not None:
            a = a.masked_fill(mask.unsqueeze(-1) < 0.5, float("-inf"))
        a = torch.softmax(a, dim=1)
        # a fully-masked row would softmax to NaN; the cache check found no such
        # study but keep this defensive so one bad row cannot poison a whole epoch
        a = torch.nan_to_num(a, nan=0.0)
        return (a * x).sum(1)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = load_encoder()
        d = 2048
        self.slice_pool = Attn(d)
        self.slot_pool = Attn(d)
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(0.2), nn.Linear(d, N_LAB))

    def forward(self, x, pres):
        # x (B, 6, S, H, W) already scaled to [-1, 1]
        B, SL, S, H, W = x.shape
        x = x.reshape(B * SL * S, 1, H, W).expand(-1, 3, -1, -1)
        f = self.enc(x)                             # (B*6*S, 2048)
        f = f.reshape(B, SL, S, -1)
        f = self.slice_pool(f.reshape(B * SL, S, -1)).reshape(B, SL, -1)
        f = self.slot_pool(f, mask=pres)
        return self.head(f)

def load_encoder():
    # inference-only: weights arrive from the checkpoint, not from disk or network
    return RadEncoder()


# ---------------------------------------------------------------- inference

N_SLOT = len(SLOTS)

def _bl_find_ckpts():
    # checkpoints ship as an attached dataset; the submission kernel is offline.
    #
    # PRUNE AGGRESSIVELY. A naive os.walk("/kaggle/input") also descends the mounted
    # competition tree -- 24k series directories of DICOMs -- and measured 251s of
    # the 276s total run time, dwarfing decode (0.9s/study) and GPU (1.6s/study).
    # The competition files are reached by explicit path elsewhere, never by walking.
    hits = {}
    skip = {"train_series", "test_series", "train_images", "test_images",
            "competitions", "train", "test"}
    for dirpath, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in skip]
        for f in files:
            if f.startswith(("armA_fold", "armB_fold")) and f.endswith(".pt"):
                hits[f] = os.path.join(dirpath, f)
        if len(hits) >= 10:
            break
    return hits

def _bl_norm(x, arm):
    # arm A is RadImageNet -> [-1,1]; arm B is ImageNet -> mean/std.
    # Getting this backwards silently degrades the features, so it is keyed off
    # the checkpoint name rather than a flag someone could forget to set.
    if arm == "armA":
        return x.div_(127.5).sub_(1.0)
    return x.div_(255.0).sub_(0.449).div_(0.226)

def _bl_build_study(root, st, rows):
    # same slot selection as training: deepest series wins each slot
    vol = np.zeros((N_SLOT, N_SLICE, SIZE, SIZE), np.uint8)
    pres = np.zeros(N_SLOT, np.float32)
    depth = {}
    for sid in rows["SeriesInstanceUID"]:
        d = os.path.join(root, st, str(sid))
        try:
            depth[sid] = sum(1 for e in os.scandir(d) if e.name.endswith(".dcm"))
        except Exception:
            depth[sid] = 0
    for si, (plane, fluid) in enumerate(SLOTS):
        cand = rows[(rows["Anatomical_Plane"] == plane) &
                    (rows["Fluid_Sensitive"] == fluid)]
        if not len(cand):
            continue
        sid = max(cand["SeriesInstanceUID"], key=lambda s: depth.get(s, 0))
        if depth.get(sid, 0) == 0:
            continue
        try:
            v = build_slot(os.path.join(root, st, str(sid)), plane)
        except Exception:
            v = None
        if v is not None:
            vol[si] = v
            pres[si] = 1.0
    return vol, pres

def _bl_to_batch(vol, arm):
    # deterministic centre crop and evenly spaced slices == the validation path
    sel = np.linspace(0, N_SLICE - 1, TRAIN_SLICES).round().astype(int)
    o = (SIZE - IMG) // 2
    v = vol[:, sel][:, :, o:o + IMG, o:o + IMG]
    x = torch.from_numpy(np.ascontiguousarray(v)).float()
    return _bl_norm(x, arm).unsqueeze(0)

def _bl_run():
    _t_find = _bl_clock()
    ckpts = _bl_find_ckpts()
    _BL_T["find"] = _bl_clock() - _t_find
    if not ckpts:
        raise RuntimeError("no armA_fold*.pt / armB_fold*.pt found under /kaggle/input")
    _bl_log("checkpoints: {0} found in {1:.1f}s".format(
        sorted(ckpts), _BL_T["find"]))

    comp = None
    for c in ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
              "/kaggle/input/rsna-knee-abnormality-detection"]:
        if os.path.isdir(c):
            comp = c
            break
    if comp is None:
        raise RuntimeError("competition root not found")

    ser = pd.read_csv(os.path.join(comp, "test_series.csv"),
                      dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    if "Fluid_Sensitive" not in ser.columns:
        # measured identical to Fat_Suppression on every train and test row
        ser["Fluid_Sensitive"] = ser["Fat_Suppression"]
    by_study = {st: g for st, g in ser.groupby("StudyInstanceUID")}
    root = os.path.join(comp, "test_series")

    stage = pd.read_csv(_BL_STAGE, dtype={"StudyInstanceUID": str})
    assert stage.columns.tolist()[1:] == _BL_LABELS, "staging schema drift"
    uids = stage["StudyInstanceUID"].astype(str).tolist()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _t_ck = _bl_clock()
    models = {}
    for f, p in sorted(ckpts.items()):
        arm = "armA" if f.startswith("armA") else "armB"
        sd = torch.load(p, map_location="cpu", weights_only=False)
        m = Net()
        m.load_state_dict(sd["model"], strict=True)
        models.setdefault(arm, []).append(m.to(dev).eval())
    _BL_T["load_ckpts"] = _bl_clock() - _t_ck
    _bl_log("loaded {0} in {1:.1f}s".format(
        {k: len(v) for k, v in models.items()}, _BL_T["load_ckpts"]))

    ours = {"armA": np.full((len(uids), 12), np.nan, np.float64),
            "armB": np.full((len(uids), 12), np.nan, np.float64)}
    ok = np.zeros(len(uids), bool)
    for i, st in enumerate(uids):
        g = by_study.get(st)
        if g is None:
            continue
        try:
            _t0 = _bl_clock()
            vol, pres = _bl_build_study(root, st, g)
            _BL_T["decode"] += _bl_clock() - _t0
            if pres.sum() == 0:
                continue
            _t1 = _bl_clock()
            pt = torch.from_numpy(pres).unsqueeze(0).to(dev)
            with torch.no_grad():
                for arm, ms in models.items():
                    x = _bl_to_batch(vol, arm).to(dev)
                    acc = np.zeros(12, np.float64)
                    for m in ms:
                        acc += torch.sigmoid(m(x, pt))[0].float().cpu().numpy()
                    # average WITHIN an arm only: folds of one arm share a
                    # calibration, so averaging their probabilities is sound
                    ours[arm][i] = acc / max(len(ms), 1)
            _BL_T["forward"] += _bl_clock() - _t1
            ok[i] = True
        except Exception:
            _bl_log("study {0} failed:\n{1}".format(st, traceback.format_exc()))
        if (i + 1) % 100 == 0:
            _bl_log("{0}/{1} studies, {2} predicted".format(i + 1, len(uids), int(ok.sum())))

    _bl_log("predicted {0} of {1} studies".format(int(ok.sum()), len(uids)))
    _bl_log("TIMING find {0:.1f}s  load_ckpts {1:.1f}s  decode {2:.1f}s  "
            "forward {3:.1f}s".format(_BL_T["find"], _BL_T["load_ckpts"],
                                      _BL_T["decode"], _BL_T["forward"]))
    if ok.sum():
        _bl_log("TIMING per study: decode {0:.1f}s  forward {1:.1f}s".format(
            _BL_T["decode"] / ok.sum(), _BL_T["forward"] / ok.sum()))
    if ok.sum() == 0:
        raise RuntimeError("no study could be predicted -- refusing to blend")

    base = stage[_BL_LABELS].to_numpy(np.float64)
    if not np.isfinite(base).all():
        raise RuntimeError("parent staging file is not finite")

    # ACROSS arms, rank first then average. Arm A is RadImageNet-pretrained and arm
    # B ImageNet-pretrained, so their probability scales differ; averaging raw
    # sigmoids would let the sharper arm dominate. Rank-averaging is what was
    # measured on the out-of-fold data (+0.0123 macro over the best single arm).
    mix = np.zeros((int(ok.sum()), 12), np.float64)
    n_arm = 0
    for arm in ("armA", "armB"):
        a = ours[arm][ok]
        if not np.isfinite(a).all():
            continue
        mix += pd.DataFrame(a).rank(method="average", pct=True).to_numpy()
        n_arm += 1
    if n_arm == 0:
        raise RuntimeError("no arm produced finite predictions")
    mix /= n_arm
    _bl_log("mixed {0} arm(s) by rank".format(n_arm))

    base_r = pd.DataFrame(base).rank(method="average", pct=True).to_numpy()
    out = base_r.copy()
    out[ok] = (1.0 - _BL_W) * base_r[ok] + _BL_W * mix
    if not np.isfinite(out).all():
        raise RuntimeError("blended output is not finite")

    stage[_BL_LABELS] = out
    stage.to_csv(_BL_STAGE, index=False)
    _bl_log("blended {0} studies; mean weight {1:.3f}".format(
        int(ok.sum()), float(_BL_W.mean())))

shutil.copy(_BL_STAGE, _BL_BACKUP)
try:
    _bl_run()
except Exception:
    # never let our arm break a working 0.940 submission
    shutil.copy(_BL_BACKUP, _BL_STAGE)
    _bl_log("FAILED, staging file restored unchanged:\n{0}".format(traceback.format_exc()))
    raise
