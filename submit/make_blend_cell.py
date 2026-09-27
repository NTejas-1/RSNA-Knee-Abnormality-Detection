"""
Generate the blend cell that gets appended to the reference submission notebook.

WHY SPLICE INSTEAD OF REWRITE
-----------------------------
Our arms were trained on a very specific geometry: 140mm physical-FOV centre crop,
256px, 16 slices sampled in the 0.06-0.94 band, 6 slots of (plane x Fluid_Sensitive),
every knee flipped to Left. If the inference cell rebuilt that by hand and any
constant drifted -- 130mm instead of 140, a different band, a missed flip -- the
model would silently receive out-of-distribution inputs. No exception, no failed
assert, just a quietly worse score.

So the geometry is EXTRACTED from cache/build_cache.py and the model from
train/train_armA.py using ast, and pasted verbatim. There is exactly one definition
of each, and it is the one that was trained against.

Verified prerequisites (tools -> rsna-knee-meta-probe):
  * test_series.csv exists and HAS Fluid_Sensitive, so the 6 slots are buildable
  * Fluid_Sensitive and Fat_Suppression are identical columns (1.0000 agreement
    over all 24371 train and 15 test rows), so keying on either is equivalent

  python submit/make_blend_cell.py --out submit/blend_cell.py
  python submit/make_blend_cell.py --insert out/notebook_with_blend.ipynb
"""
import argparse
import ast
import io
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_SRC = os.path.join(HERE, "cache", "build_cache.py")
TRAIN_SRC = os.path.join(HERE, "train", "train_armA.py")
NB_SRC = os.path.join(HERE, "ref_kernel",
                      "rsna-knee-speedy-raptors-coatnet-d4-sota.ipynb")

# geometry that defines the trained input distribution
CACHE_NAMES = ["CROP_MM", "SIZE", "N_SLICE", "BAND", "SLOTS", "THREADS",
               "read_header", "natural_key", "order_slices",
               "sample_positions", "load_slice", "build_slot"]
# the model itself. LABELS and N_LAB are pulled in too because Net's head is
# Linear(2048, N_LAB) -- omitting them yields a cell that compiles and then dies
# with NameError at runtime, inside the submission kernel, after hours of work.
TRAIN_NAMES = ["IMG", "TRAIN_SLICES", "LABELS", "N_LAB",
               "RadEncoder", "Attn", "Net"]

# Net.__init__ calls load_encoder(), which in training reads RadImageNet weights
# from a dataset or downloads ImageNet weights. At inference we need NEITHER: the
# checkpoint supplies every weight via load_state_dict(strict=True) immediately
# after construction. The submission kernel is also offline, so an ImageNet
# download would fail outright. This shim returns an empty trunk of the right
# shape and nothing else.
ENCODER_SHIM = '''

def load_encoder():
    # inference-only: weights arrive from the checkpoint, not from disk or network
    return RadEncoder()
'''


def extract(path, names):
    """Pull named top-level constants, functions and classes out verbatim."""
    src = io.open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if node.name in names:
                found[node.name] = ast.get_source_segment(src, node)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    found[t.id] = ast.get_source_segment(src, node)
    missing = [n for n in names if n not in found]
    if missing:
        raise SystemExit("could not extract {0} from {1} -- it was renamed or "
                         "moved, so the spliced geometry would be stale".format(
                             missing, os.path.basename(path)))
    # preserve the requested order so constants land before the code using them
    return "\n\n".join(found[n] for n in names)


HEADER = '''# ===================================================================
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
_BL_W = np.asarray(__WEIGHTS__, dtype=np.float64)
assert len(_BL_W) == 12

def _bl_log(m):
    print("[blend] {0}".format(m), flush=True)

# phase timing: the first end-to-end run came out at 143s per study, which would be
# ~52h over 1300 studies against a 9h cap. Attributing that to decode vs GPU vs
# checkpoint loading decides which optimisation is worth doing.
import time as _bl_time
_bl_clock = _bl_time.time
_BL_T = {"load_ckpts": 0.0, "decode": 0.0, "forward": 0.0, "find": 0.0}
'''

DRIVER = '''

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
            _bl_log("study {0} failed:\\n{1}".format(st, traceback.format_exc()))
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
    _bl_log("FAILED, staging file restored unchanged:\\n{0}".format(traceback.format_exc()))
    raise
'''


def extract_try_defining(path, name):
    """Extract the whole try/except block that defines `name`.

    build_cache.py picks its resampler at import time: cv2 INTER_AREA if cv2 is
    importable, else PIL BILINEAR. Those do NOT produce the same pixels. Copying
    only one branch, or reimplementing with a guessed interpolation, would feed the
    model subtly different images than it was trained on. So the whole block is
    lifted verbatim and the active branch is logged at run time.
    """
    src = io.open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Try):
            for sub in ast.walk(node):
                if isinstance(sub, ast.FunctionDef) and sub.name == name:
                    return ast.get_source_segment(src, node)
    raise SystemExit("no try/except defining {0} in {1}".format(name, path))


def check_names(cell):
    """Approximate undefined-name check.

    compile() only validates syntax, so a missing splice (load_encoder, N_LAB)
    sails through and fails at runtime inside the submission kernel. This walks the
    tree and reports Load-context names with no visible binding. It over-collects
    bindings deliberately, so it may miss something, but it will not cry wolf.
    """
    import builtins
    tree = ast.parse(cell)
    defined = set(dir(builtins))
    # names the host notebook already has in scope when our cell runs
    defined |= {"_RSNA_LABELS", "_RSNA_TEST_IDS", "_RSNA_AUDIT", "Path"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for al in node.names:
                defined.add((al.asname or al.name).split(".")[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
    return sorted(used - defined)


RESIZE_REPORT = (
    '\n_bl_log("resize backend: " + ("cv2 INTER_AREA" if "cv2" in sys.modules'
    ' else "PIL BILINEAR -- SKEW if the cache was built with cv2"))\n')


def build_cell(weights):
    geom = extract(CACHE_SRC, CACHE_NAMES)
    resizer = extract_try_defining(CACHE_SRC, "resize")
    model = extract(TRAIN_SRC, TRAIN_NAMES)
    head = HEADER.replace("__WEIGHTS__", repr(list(weights)))
    return (head
            + "\n# ---- resampler spliced verbatim from cache/build_cache.py ----\n"
            + resizer
            + RESIZE_REPORT
            + "\n# ---- geometry spliced verbatim from cache/build_cache.py ----\n"
            + geom
            + "\n\n# ---- model spliced verbatim from train/train_armA.py ----\n"
            + model
            + ENCODER_SHIM
            + DRIVER)


TEST_PREAMBLE = '''# Standalone harness: run our blend path alone, on the example test studies.
#
# The real cell runs deep inside the reference pipeline after hours of parent
# compute. Validating it there would cost a full submission run per bug. This fakes
# the two things the cell consumes from its host -- _RSNA_LABELS and a staging CSV --
# so the DICOM pass, the checkpoint loads and the blend arithmetic can all be
# exercised in a couple of minutes, and per-study time can be extrapolated to the
# real test set.
import time as _t_time
_T_START = _t_time.time()
import os as _t_os
import pandas as _t_pd

_RSNA_LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
                "Medial OA", "Lateral OA", "PF OA", "Effusion",
                "Synovitis", "Baker\\'s", "Contusion", "Fracture"]

_t_comp = None
for _c in ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
           "/kaggle/input/rsna-knee-abnormality-detection"]:
    if _t_os.path.isdir(_c):
        _t_comp = _c
        break
_t_sub = _t_pd.read_csv(_t_os.path.join(_t_comp, "sample_submission.csv"),
                        dtype={"StudyInstanceUID": str})
# neutral parent predictions: any constant works, we only check our arm mixes in
for _c in _RSNA_LABELS:
    _t_sub[_c] = 0.5
_t_sub = _t_sub[["StudyInstanceUID"] + _RSNA_LABELS]
_t_sub.to_csv("/kaggle/working/_pipeline_stage.csv", index=False)
print("[harness] staged {0} studies".format(len(_t_sub)), flush=True)

'''

TEST_EPILOGUE = '''

# ---------------------------------------------------------------- verdict
import numpy as _v_np
import pandas as _v_pd
_v = _v_pd.read_csv("/kaggle/working/_pipeline_stage.csv",
                    dtype={"StudyInstanceUID": str})
_vals = _v[_RSNA_LABELS].to_numpy(float)
_n = len(_v)
_el = _t_time.time() - _T_START
_marg = (_BL_T["decode"] + _BL_T["forward"]) / max(_n, 1)
_fixed = _el - (_BL_T["decode"] + _BL_T["forward"])
print("")
print("[harness] elapsed {0:.1f}s for {1} studies".format(_el, _n))
print("[harness] fixed cost {0:.1f}s (imports + checkpoint loads)".format(_fixed))
print("[harness] MARGINAL {0:.2f}s per study (decode + forward)".format(_marg))
# Extrapolate on the MARGINAL cost. Dividing total elapsed by 3 studies folds the
# one-off fixed cost into the per-study figure and overstates the real run by ~5x,
# which would push us into optimising something that is not actually a problem.
print("[harness] 1300 studies: {0:.2f}h marginal + {1:.1f}s fixed = {2:.2f}h".format(
    _marg * 1300 / 3600.0, _fixed, (_marg * 1300 + _fixed) / 3600.0))
print("[harness] naive (elapsed/n) extrapolation would claim {0:.2f}h -- misleading".format(
    _el / max(_n, 1) * 1300 / 3600.0))
print("[harness] finite: {0}  min {1:.4f}  max {2:.4f}".format(
    bool(_v_np.isfinite(_vals).all()), float(_vals.min()), float(_vals.max())))
# a blend that changed nothing means our arm silently contributed nothing
_moved = float((_v_np.abs(_vals - 0.5) > 1e-9).mean())
print("[harness] cells moved away from the 0.5 parent: {0:.1%}".format(_moved))
if _moved < 0.5:
    raise SystemExit("our arm changed almost nothing -- inference did not run")
print("[harness] PASS")
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "submit", "blend_cell.py"))
    ap.add_argument("--insert")
    ap.add_argument("--halved", action="store_true",
                    help="use the halved hedge weights")
    ap.add_argument("--test", help="write a standalone harness script instead")
    a = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bw", os.path.join(HERE, "submit", "blend_weights.py"))
    bw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bw)
    weights = bw.as_vector(bw.HALVED if a.halved else bw.BLEND_W)

    cell = build_cell(weights)
    compile(cell, "blend_cell", "exec")       # syntax only
    unresolved = check_names(cell)
    if unresolved:
        raise SystemExit("unresolved names in the generated cell -- it would raise "
                         "NameError at run time: {0}".format(unresolved))
    io.open(a.out, "w", encoding="utf-8").write(cell)
    print("wrote {0} ({1} chars, {2} lines) -- compiles, names resolve".format(
        a.out, len(cell), cell.count("\n") + 1))
    print("weights:", weights)

    if a.test:
        harness = TEST_PREAMBLE + cell + TEST_EPILOGUE
        compile(harness, "test_blend", "exec")
        os.makedirs(os.path.dirname(a.test) or ".", exist_ok=True)
        io.open(a.test, "w", encoding="utf-8").write(harness)
        print("wrote harness {0} ({1} lines)".format(
            a.test, harness.count("\n") + 1))

    if a.insert:
        nb = json.load(io.open(NB_SRC, encoding="utf-8"))
        n = len(nb["cells"])
        if n != 24:
            print("WARNING notebook has {0} cells, expected 24".format(n))
        # Jupyter's `source` is a list of lines that each KEEP their newline.
        # split("\n") strips them, which silently collapses the whole cell onto one
        # line -- it still looks like valid JSON and then fails to run.
        new = {"cell_type": "code", "execution_count": None,
               "metadata": {}, "outputs": [], "source": cell.splitlines(True)}
        # index 23 == after the CoAt arm, before the publish cell
        nb["cells"].insert(23, new)
        os.makedirs(os.path.dirname(a.insert), exist_ok=True)
        json.dump(nb, io.open(a.insert, "w", encoding="utf-8"))
        print("inserted at index 23 -> {0} ({1} cells)".format(
            a.insert, len(nb["cells"])))


if __name__ == "__main__":
    main()
