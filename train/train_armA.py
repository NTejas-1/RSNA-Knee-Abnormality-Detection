"""
RSNA Knee -- training arm A: RadImageNet ResNet50 + attention pooling.

Targets the two largest independently-measured levers in the discussions:
  * RadImageNet instead of ImageNet pretraining -- Ziad Ahmed isolated this as a
    single variable and measured +0.063, the biggest reported gain in the comp.
  * Attention pooling instead of masked mean pooling over slices -- +0.014.

Trains one fold per kernel run so folds can run concurrently in separate GPU
sessions and each stays well inside the session limit.

Targets are the FRACTIONAL Opus probabilities, not thresholded 0/1 labels:
Archit Konde reported that rounding them hurt. Loss is soft-target BCE.

IMPORTANT -- no horizontal flip augmentation. The cache already normalises every
knee to Left, and Medial OA / Lateral OA (and the two menisci) are distinguished
by side, so a horizontal flip would silently swap those four labels.
"""
import os
import sys
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- config

FOLDS = [0]            # folds this kernel trains, sequentially
N_FOLDS = 5
EPOCHS = 8
SEED = 1234

N_SLOT = 6
CACHE_SLICES = 16      # what the cache holds
TRAIN_SLICES = 8       # sampled per slot per epoch: augmentation + halves compute
CACHE_SIZE = 256       # stored resolution
IMG = 224              # training resolution (224-288 is the reported sweet spot)

BATCH_STUDIES = 2
ACCUM = 4
LR = 3e-4
WD = 1e-2
WARMUP_FRAC = 0.1

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
          "Medial OA", "Lateral OA", "PF OA", "Effusion",
          "Synovitis", "Baker's", "Contusion", "Fracture"]
N_LAB = len(LABELS)

CACHE_DIRS = None      # resolved by search; layout differs across Kaggle envs
# swapped to labels_v2_repaired.csv by the pusher for the repaired retrain. The
# repaired file differs ONLY on silent Synovitis and Effusion cells; every cell the
# radiologist stated explicitly is byte-identical between the two.
# Arm B is this same file with ENCODER="imagenet" and MODEL_SEED=777, rendered by
# the pusher. Deriving it from arm A rather than maintaining a separate script is
# deliberate: the standalone arm B leaked host RAM at ~45MB/step and was SIGKILLed
# on every fold, while this file has run 5 folds at a stable 0.30 s/step. Keeping
# the module graph byte-identical and swapping only the weight VALUES means arm B
# cannot have a different memory or speed profile than arm A.
#
# Decorrelation retained: pretraining source (ImageNet vs RadImageNet) and init
# seed. Given up versus the original design: the pooling and normalisation axes.
# That is a smaller decorrelation delta, but it is one that actually runs.
ENCODER = "rad"          # "rad" | "imagenet"
MODEL_SEED = None        # None -> use SEED; arm B sets 777
RUN_PREFIX = "armA"      # output filename prefix; arm B sets "armB"
IMNET_MEAN, IMNET_STD = 0.449, 0.226

LABELS_FILE = "labels_v1_opus45.csv"

N_CACHE_SHARDS = 4
OUT = "/kaggle/working"
INPUT_ROOT = "/kaggle/input"

T0 = time.time()


def log(m):
    print("[{0:7.1f}s] {1}".format(time.time() - T0, m), flush=True)


def print_tree(max_depth=3):
    for dirpath, dirnames, filenames in os.walk(INPUT_ROOT):
        depth = dirpath[len(INPUT_ROOT):].count(os.sep)
        if depth >= max_depth:
            dirnames[:] = []
        pad = "  " * depth
        log("{0}{1}/".format(pad, os.path.basename(dirpath) or INPUT_ROOT))
        for f in sorted(filenames)[:8]:
            log("{0}  {1}".format(pad, f))


def find_file(filename, required=True):
    """Kaggle mounts inputs flat OR nested under datasets/<owner>/ and
    notebooks/<owner>/ depending on the environment. Searching instead of
    hardcoding avoids silently training on a partial corpus."""
    for dirpath, _d, filenames in os.walk(INPUT_ROOT):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    if required:
        print_tree()
        raise SystemExit("could not find {0} under {1}".format(filename, INPUT_ROOT))
    return None


# ---------------------------------------------------------------- data

class Corpus:
    """Memory-maps the cache shards and indexes studies across them."""

    def __init__(self):
        self.shards, self.present, self.index = [], [], {}
        for i in range(N_CACHE_SHARDS):
            cp = find_file("cache_shard{0}.npy".format(i), required=False)
            if cp is None:
                log("MISSING cache_shard{0}.npy".format(i))
                continue
            d = os.path.dirname(cp)
            arr = np.load(cp, mmap_mode="r")
            pres = np.load(os.path.join(d, "present_shard{0}.npy".format(i)))
            st = pd.read_csv(os.path.join(d, "studies_shard{0}.csv".format(i)))
            si = len(self.shards)
            self.shards.append(arr)
            self.present.append(pres)
            for k, u in enumerate(st["StudyInstanceUID"]):
                self.index[str(u)] = (si, k)
            log("shard {0}: {1} studies, fill {2:.1%}, shape {3}".format(
                i, len(st), pres.mean(), arr.shape))
        if not self.shards:
            print_tree()
            raise SystemExit("no cache shards attached")
        log("corpus: {0} studies total".format(len(self.index)))
        # all four shards must be present or the model trains on 75% of the data
        # while the log looks healthy
        if len(self.shards) != N_CACHE_SHARDS:
            raise SystemExit("expected {0} shards, attached {1}".format(
                N_CACHE_SHARDS, len(self.shards)))

    def get(self, uid):
        si, k = self.index[uid]
        return self.shards[si][k], self.present[si][k]


class DS(torch.utils.data.Dataset):
    def __init__(self, corpus, uids, targets, train):
        self.c, self.uids, self.y, self.train = corpus, uids, targets, train

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        uid = self.uids[i]
        vol, pres = self.c.get(uid)
        vol = np.asarray(vol)                      # (6, 16, 256, 256)

        if self.train:
            sel = np.sort(np.random.choice(CACHE_SLICES, TRAIN_SLICES, replace=False))
        else:
            sel = np.linspace(0, CACHE_SLICES - 1, TRAIN_SLICES).round().astype(int)
        vol = vol[:, sel]

        if self.train:
            m = CACHE_SIZE - IMG
            oy, ox = np.random.randint(0, m + 1), np.random.randint(0, m + 1)
        else:
            oy = ox = (CACHE_SIZE - IMG) // 2
        vol = vol[:, :, oy:oy + IMG, ox:ox + IMG]

        x = torch.from_numpy(np.ascontiguousarray(vol)).float()
        if self.train:
            # mild intensity jitter only -- see module docstring re: no h-flip
            x.mul_(1.0 + 0.12 * (torch.rand(1).item() - 0.5))
            x.add_(15.0 * (torch.rand(1).item() - 0.5))
            x.clamp_(0, 255)
        # RadImageNet preprocessing is [-1, 1], NOT ImageNet mean/std. Using
        # ImageNet stats with these weights degrades the features.
        x = normalize_(x)

        return x, torch.from_numpy(pres.astype(np.float32)), \
            torch.from_numpy(self.y[i].astype(np.float32))


# ---------------------------------------------------------------- model

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
        return self.backbone(x).mean(dim=(2, 3))     # (N, 2048)


def normalize_(x):
    """RadImageNet expects [-1, 1]; ImageNet weights expect ImageNet mean/std.
    Using the wrong one degrades the features badly, so it follows ENCODER."""
    if ENCODER == "rad":
        return x.div_(127.5).sub_(1.0)
    return x.div_(255.0).sub_(IMNET_MEAN).div_(IMNET_STD)


def load_radimagenet():
    """Ziad Ahmed isolated RadImageNet vs ImageNet as a single variable and
    measured +0.063, the largest reported gain in this competition. A silent
    failure here costs all of it while the log still looks healthy, so the load is
    strict and the parameter count is asserted."""
    p = find_file("ResNet50.pt")
    enc = RadEncoder()
    sd = torch.load(p, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    enc.load_state_dict(sd, strict=True)
    n = sum(q.numel() for q in enc.parameters())
    log("RadImageNet loaded strict from {0}; {1} params".format(p, n))
    if n != 23508032:
        raise SystemExit("Rad encoder parameter count is {0}, expected 23508032 "
                         "-- structure drifted".format(n))
    return enc


def load_imagenet_resnet50():
    """Same trunk structure as RadEncoder, ImageNet weights instead.

    Structural identity is the point: arm B must differ from arm A only in the
    VALUES of the weights, so that its memory and throughput are guaranteed to
    match a configuration already proven over 5 folds.
    """
    import torchvision
    enc = RadEncoder()
    src = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    trunk = nn.Sequential(*list(src.children())[:-2])
    enc.backbone.load_state_dict(trunk.state_dict(), strict=True)
    n = sum(q.numel() for q in enc.parameters())
    log("ImageNet ResNet50 loaded strict; {0} params".format(n))
    if n != 23508032:
        raise SystemExit("ImageNet encoder param count is {0}, expected 23508032 "
                         "-- structure drifted from arm A".format(n))
    return enc


def load_encoder():
    if ENCODER == "rad":
        return load_radimagenet()
    if ENCODER == "imagenet":
        return load_imagenet_resnet50()
    raise SystemExit("unknown ENCODER {0!r}".format(ENCODER))


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


# ---------------------------------------------------------------- metric

def auc(y, s):
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    p, n = (y == 1).sum(), (y == 0).sum()
    if p == 0 or n == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * n)


# ---------------------------------------------------------------- train

def main():
    # Fold isolation. The first multi-fold attempt ran every fold in one process
    # and was OOM-killed ("Killed") partway through the second fold: models,
    # optimizers, DataParallel replicas and persistent DataLoader workers
    # accumulated across folds. Running each fold in a fresh subprocess returns
    # all of it to the OS on exit, and keeps the whole run server-side so no local
    # machine has to stay awake between folds.
    one = os.environ.get("RSNA_ONE_FOLD")
    if one is None and len(FOLDS) > 1:
        import subprocess
        rc_all = {}
        for f in FOLDS:
            log("")
            log("======== launching fold {0} in a clean subprocess ========".format(f))
            env = dict(os.environ)
            env["RSNA_ONE_FOLD"] = str(f)
            r = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env)
            rc_all[f] = r.returncode
            log("fold {0} subprocess exited {1}".format(f, r.returncode))
        log("ALL FOLDS DONE: {0}".format(rc_all))
        bad = [f for f, rc in rc_all.items() if rc != 0]
        if bad:
            log("WARNING folds that failed: {0}".format(bad))
        return

    folds = [int(one)] if one is not None else list(FOLDS)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    corpus = Corpus()
    lab = pd.read_csv(find_file(LABELS_FILE))
    lab["STUDY_UID"] = lab["STUDY_UID"].astype(str)
    lab = lab[lab["STUDY_UID"].isin(corpus.index)].reset_index(drop=True)
    log("labelled studies present in cache: {0}".format(len(lab)))

    y = lab[LABELS].to_numpy(np.float32)
    uids = lab["STUDY_UID"].tolist()

    # the 58 hand-labelled studies are the only ground truth we have. Per fold
    # only ~12 land in validation, which is far too few to select on, but logging
    # it keeps an honest signal visible next to the soft-label proxy.
    GOLD_COLS = ["GOLD_ACL", "GOLD_MCL", "GOLD_MEDIAL_MENISCUS",
                 "GOLD_LATERAL_MENISCUS", "GOLD_MEDIAL_OA", "GOLD_LATERAL_OA",
                 "GOLD_PF_OA", "GOLD_EFFUSION", "GOLD_SYNOVITIS", "GOLD_BAKERS",
                 "GOLD_CONTUSION", "GOLD_FRACTURE"]
    has_gold = lab["IS_GOLD"].fillna(False).to_numpy(bool) \
        if "IS_GOLD" in lab.columns else np.zeros(len(lab), bool)
    ygold = lab[GOLD_COLS].to_numpy(np.float32) if all(
        c in lab.columns for c in GOLD_COLS) else None
    log("gold studies available: {0}".format(int(has_gold.sum())))

    for f in folds:
        log("")
        log("################ FOLD {0} ################".format(f))
        try:
            run_fold(f, corpus, lab, y, uids, has_gold, ygold)
        except Exception:
            import traceback
            log("FOLD {0} FAILED:\n{1}".format(f, traceback.format_exc()))
            raise
    log("DONE folds {0}".format(folds))


def run_fold(FOLD, corpus, lab, y, uids, has_gold, ygold):
    # MODEL_SEED controls initialisation only. The fold split below MUST stay on
    # SEED so arm A and arm B validate on exactly the same studies -- otherwise
    # their OOF predictions cannot be blended and repair would leak across folds.
    torch.manual_seed((MODEL_SEED if MODEL_SEED is not None else SEED) + FOLD)
    np.random.seed((MODEL_SEED if MODEL_SEED is not None else SEED) + FOLD)

    rng = np.random.RandomState(SEED)
    fold_id = rng.permutation(len(uids)) % N_FOLDS
    tr_i = np.where(fold_id != FOLD)[0]
    va_i = np.where(fold_id == FOLD)[0]
    log("fold {0}: train {1}, val {2}".format(FOLD, len(tr_i), len(va_i)))

    dtr = DS(corpus, [uids[i] for i in tr_i], y[tr_i], True)
    dva = DS(corpus, [uids[i] for i in va_i], y[va_i], False)
    ltr = torch.utils.data.DataLoader(
        dtr, batch_size=BATCH_STUDIES, shuffle=True, num_workers=2,
        pin_memory=True, drop_last=True, persistent_workers=False)
    lva = torch.utils.data.DataLoader(
        dva, batch_size=BATCH_STUDIES, shuffle=False, num_workers=2,
        pin_memory=True, persistent_workers=False)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net().to(dev)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        log("DataParallel over {0} GPUs".format(torch.cuda.device_count()))

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = max(1, len(ltr) // ACCUM) * EPOCHS
    warm = int(steps * WARMUP_FRAC)

    def lr_at(s):
        if s < warm:
            return s / max(warm, 1)
        t = (s - warm) / max(steps - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

    best, best_oof = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad(set_to_none=True)
        run, seen = 0.0, 0
        for bi, (x, pres, t) in enumerate(ltr):
            x, pres, t = x.to(dev, non_blocking=True), pres.to(dev), t.to(dev)
            with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                out = model(x, pres)
                # soft-target BCE: the fractional Opus probability IS the target
                loss = F.binary_cross_entropy_with_logits(out, t)
            scaler.scale(loss / ACCUM).backward()
            if (bi + 1) % ACCUM == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            run += loss.item() * len(x)
            seen += len(x)
            if (bi + 1) % 200 == 0:
                log("ep{0} {1}/{2} loss {3:.4f}".format(
                    ep, bi + 1, len(ltr), run / max(seen, 1)))

        model.eval()
        P = []
        with torch.no_grad():
            for x, pres, t in lva:
                x, pres = x.to(dev), pres.to(dev)
                with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                    P.append(torch.sigmoid(model(x, pres)).float().cpu().numpy())
        P = np.concatenate(P)

        # validate against the SOFT extracted labels thresholded only for the
        # metric. This is a proxy: real selection uses gold + LB.
        yv = y[va_i][:len(P)]
        aucs = [auc((yv[:, j] > 0.5).astype(int), P[:, j]) for j in range(N_LAB)]
        macro = float(np.nanmean(aucs))
        log("ep{0} loss {1:.4f} val macro AUC {2:.4f}".format(
            ep, run / max(seen, 1), macro))
        for j, nm in enumerate(LABELS):
            log("    {0:18s} {1:.4f}".format(nm, aucs[j]))

        # honest-but-tiny gold signal on whatever gold rows are in this fold
        if ygold is not None:
            gmask = has_gold[va_i][:len(P)]
            if gmask.sum() >= 8:
                gv = ygold[va_i][:len(P)][gmask]
                ga = [auc(gv[:, j], P[gmask][:, j]) for j in range(N_LAB)]
                log("    [gold n={0}] macro {1:.4f}".format(
                    int(gmask.sum()), float(np.nanmean(ga))))

        if macro > best:
            best, best_oof = macro, P
            torch.save({
                "model": (model.module if hasattr(model, "module") else model).state_dict(),
                "fold": FOLD, "epoch": ep, "macro": macro,
                "labels": LABELS, "img": IMG, "slices": TRAIN_SLICES,
            }, os.path.join(OUT, "{0}_fold{1}.pt".format(RUN_PREFIX, FOLD)))
            log("saved checkpoint (best {0:.4f})".format(best))

    oof = pd.DataFrame(best_oof, columns=LABELS)
    oof.insert(0, "STUDY_UID", [uids[i] for i in va_i][:len(best_oof)])
    oof.to_csv(os.path.join(OUT, "{0}_oof_fold{1}.csv".format(RUN_PREFIX, FOLD)),
               index=False)
    log("DONE fold {0}: best macro AUC {1:.4f}".format(FOLD, best))

    # second layer of defence behind subprocess isolation: release the big objects
    # explicitly so a same-process multi-fold run also stays bounded
    import gc
    del ltr, lva, dtr, dva, model, opt, sched, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
