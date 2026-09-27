"""
RSNA Knee -- training arm B: ImageNet CoAtNet + transformer slot pooling.

Purpose is DECORRELATION, not raw strength. Charles Savas measured +0.004 from a
second arm at correlation 0.83 with the first; above roughly 0.95 a blend adds
nothing. So arm B deliberately differs from arm A on four independent axes:

  backbone       CoAtNet (timm, ImageNet) vs arm A's ResNet50 (RadImageNet)
  pretraining    natural images vs radiology images -- different feature priors
  pooling        mean over slices then a transformer encoder over the 6 slot
                 tokens, vs arm A's gated attention at both levels
  normalisation  ImageNet mean/std vs arm A's [-1, 1]
  init seed      777 vs 1234

CRITICAL: the FOLD SPLIT seed stays identical to arm A (SPLIT_SEED = 1234). The
out-of-fold predictions of the two arms must cover exactly the same validation
studies or they cannot be blended, and label repair would leak across folds.
Only the model init seed differs.

Internet is enabled on this TRAINING kernel so timm can fetch pretrained weights.
The submission kernel stays offline and loads our saved checkpoint from a dataset.
If the download fails the script falls back to RadImageNet ResNet50 with arm B's
different head, which is less decorrelated but still usable -- and it says so
loudly rather than silently training something unexpected.
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

FOLDS = [0]
N_FOLDS = 5
EPOCHS = 8
SPLIT_SEED = 1234      # MUST match arm A so OOF rows align
MODEL_SEED = 777       # differs from arm A on purpose

N_SLOT = 6
CACHE_SLICES = 16
TRAIN_SLICES = 8
CACHE_SIZE = 256
IMG = 224              # coatnet_0_rw_224 expects exactly 224

BATCH_STUDIES = 2
ACCUM = 4
LR = 2e-4              # transformer head prefers a slightly lower LR
WD = 5e-2
WARMUP_FRAC = 0.1

# CoAtNet was the first choice and had to be abandoned: on Kaggle's T4s it ran at
# 2.57 s/step versus ResNet50's 0.30 s/step -- about 8.5x slower at comparable
# FLOPs, because Turing has no fast attention kernels. One fold would have taken
# ~10h. These are all pure-convolution ImageNet backbones, which keeps the
# pretraining-source decorrelation while staying affordable.
BACKBONES = ["convnext_tiny", "efficientnet_b0", "resnet50"]

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
          "Medial OA", "Lateral OA", "PF OA", "Effusion",
          "Synovitis", "Baker's", "Contusion", "Fracture"]
N_LAB = len(LABELS)

N_CACHE_SHARDS = 4
OUT = "/kaggle/working"
INPUT_ROOT = "/kaggle/input"

IMNET_MEAN, IMNET_STD = 0.449, 0.226

T0 = time.time()


def log(m):
    print("[{0:7.1f}s] {1}".format(time.time() - T0, m), flush=True)


def mem():
    """Host RSS and container-available memory.

    Arm B's first run died with exit -9 on all five folds, which is a host-RAM
    OOM kill rather than a CUDA error, and the log gave no way to see it coming.
    Printing this alongside the loss turns the next failure into a measurement.
    """
    try:
        rss = 0
        with open("/proc/self/status") as fh:
            for ln in fh:
                if ln.startswith("VmRSS:"):
                    rss = int(ln.split()[1]) / 1048576.0
        avail = 0
        with open("/proc/meminfo") as fh:
            for ln in fh:
                if ln.startswith("MemAvailable:"):
                    avail = int(ln.split()[1]) / 1048576.0
        g = torch.cuda.memory_reserved() / 1073741824.0 if torch.cuda.is_available() else 0
        return "rss {0:.1f}G avail {1:.1f}G gpu {2:.1f}G".format(rss, avail, g)
    except Exception as exc:
        return "mem probe failed: {0}".format(exc)


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
    for dirpath, _d, filenames in os.walk(INPUT_ROOT):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    if required:
        print_tree()
        raise SystemExit("could not find {0}".format(filename))
    return None


# ---------------------------------------------------------------- data

class Corpus:
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
            log("shard {0}: {1} studies, fill {2:.1%}".format(i, len(st), pres.mean()))
        if len(self.shards) != N_CACHE_SHARDS:
            print_tree()
            raise SystemExit("expected {0} shards, attached {1}".format(
                N_CACHE_SHARDS, len(self.shards)))
        log("corpus: {0} studies".format(len(self.index)))

    def get(self, uid):
        si, k = self.index[uid]
        return self.shards[si][k], self.present[si][k]


class DS(torch.utils.data.Dataset):
    def __init__(self, corpus, uids, targets, train):
        self.c, self.uids, self.y, self.train = corpus, uids, targets, train

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        vol, pres = self.c.get(self.uids[i])
        vol = np.asarray(vol)
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

        x = torch.from_numpy(np.ascontiguousarray(vol)).float().div_(255.0)
        if self.train:
            # intensity jitter only. NO horizontal flip: the cache normalises every
            # knee to Left and Medial/Lateral OA plus the two menisci are defined
            # by side, so flipping would silently swap four labels.
            x.mul_(1.0 + 0.12 * (torch.rand(1).item() - 0.5))
            x.add_(0.06 * (torch.rand(1).item() - 0.5))
            x.clamp_(0, 1)
        # ImageNet normalisation, unlike arm A's [-1,1] -- another decorrelation axis
        x = (x - IMNET_MEAN) / IMNET_STD

        return x, torch.from_numpy(pres.astype(np.float32)), \
            torch.from_numpy(self.y[i].astype(np.float32))


# ---------------------------------------------------------------- model

def build_backbone():
    """timm CoAtNet, ImageNet pretrained. Falls back to RadImageNet ResNet50 only
    if the download fails, and reports which path was taken."""
    try:
        import timm
    except Exception as exc:
        log("timm unavailable ({0}); falling back to RadImageNet".format(exc))
        return build_rad_fallback()

    for name in BACKBONES:
        try:
            m = timm.create_model(name, pretrained=True, num_classes=0, in_chans=3)
            dim = m.num_features
            log("backbone {0} pretrained OK, feature dim {1}".format(name, dim))
            return m, dim, name
        except Exception as exc:
            log("backbone {0} failed: {1}".format(name, str(exc)[:200]))
    log("all timm backbones failed; falling back to RadImageNet")
    return build_rad_fallback()


def build_rad_fallback():
    import torchvision
    p = find_file("ResNet50.pt")

    class RadTrunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                *list(torchvision.models.resnet50(weights=None).children())[:-2])

        def forward(self, x):
            return self.backbone(x).mean(dim=(2, 3))

    t = RadTrunk()
    sd = torch.load(p, map_location="cpu", weights_only=True)
    t.load_state_dict(sd, strict=True)
    n = sum(q.numel() for q in t.parameters())
    if n != 23508032:
        raise SystemExit("Rad fallback param count {0}".format(n))
    log("FALLBACK: RadImageNet ResNet50 (arm B is LESS decorrelated than intended)")
    return t, 2048, "radimagenet_resnet50_fallback"


class Net(nn.Module):
    """Mean-pool slices, then a transformer encoder over the 6 slot tokens.

    Structurally different from arm A, which uses gated attention at both the
    slice and slot level. Absent slots are masked out of the attention so a study
    missing its axial non-fluid series (about 81% of them) contributes nothing
    there rather than a zero vector that the model would have to learn to ignore.
    """

    def __init__(self):
        super().__init__()
        self.enc, dim, self.bb_name = build_backbone()
        self.proj = nn.Linear(dim, 512)
        self.slot_embed = nn.Parameter(torch.randn(N_SLOT, 512) * 0.02)
        self.slot_score = nn.Linear(512, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(512), nn.Dropout(0.2), nn.Linear(512, N_LAB))

    def forward(self, x, pres):
        B, SL, S, H, W = x.shape
        f = self.enc(x.reshape(B * SL * S, 1, H, W).expand(-1, 3, -1, -1))
        f = f.reshape(B, SL, S, -1).mean(2)              # mean over slices
        t = self.proj(f) + self.slot_embed.unsqueeze(0)   # (B, 6, 512)
        # single tanh attention over slots. Arm A instead uses a gated v/u
        # attention at BOTH the slice and the slot level, so the two arms
        # aggregate very differently even where the features agree.
        a = self.slot_score(torch.tanh(t)).squeeze(-1)    # (B, 6)
        a = a.masked_fill(pres < 0.5, float("-inf"))
        a = torch.nan_to_num(torch.softmax(a, dim=1), nan=0.0)
        return self.head((t * a.unsqueeze(-1)).sum(1))


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
    # one fold per subprocess: a single process accumulated model, optimizer and
    # DataLoader memory across folds and was OOM-killed ("Killed") mid-fold on the
    # first multi-fold attempt
    one = os.environ.get("RSNA_ONE_FOLD")
    if one is None and len(FOLDS) > 1:
        import subprocess
        rc = {}
        for f in FOLDS:
            log("")
            log("======== launching fold {0} in a clean subprocess ========".format(f))
            env = dict(os.environ)
            env["RSNA_ONE_FOLD"] = str(f)
            r = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env)
            rc[f] = r.returncode
            log("fold {0} exited {1}".format(f, r.returncode))
        log("ALL FOLDS DONE: {0}".format(rc))
        bad = [f for f, v in rc.items() if v != 0]
        if bad:
            log("WARNING folds failed: {0}".format(bad))
        return

    folds = [int(one)] if one is not None else list(FOLDS)

    corpus = Corpus()
    lab = pd.read_csv(find_file("labels_v1_opus45.csv"))
    lab["STUDY_UID"] = lab["STUDY_UID"].astype(str)
    lab = lab[lab["STUDY_UID"].isin(corpus.index)].reset_index(drop=True)
    y = lab[LABELS].to_numpy(np.float32)
    uids = lab["STUDY_UID"].tolist()
    log("labelled studies present in cache: {0}".format(len(lab)))

    GOLD_COLS = ["GOLD_ACL", "GOLD_MCL", "GOLD_MEDIAL_MENISCUS",
                 "GOLD_LATERAL_MENISCUS", "GOLD_MEDIAL_OA", "GOLD_LATERAL_OA",
                 "GOLD_PF_OA", "GOLD_EFFUSION", "GOLD_SYNOVITIS", "GOLD_BAKERS",
                 "GOLD_CONTUSION", "GOLD_FRACTURE"]
    has_gold = lab["IS_GOLD"].fillna(False).to_numpy(bool)
    ygold = lab[GOLD_COLS].to_numpy(np.float32)

    for f in folds:
        log("")
        log("################ ARM B FOLD {0} ################".format(f))
        run_fold(f, corpus, y, uids, has_gold, ygold)
    log("DONE folds {0}".format(folds))


def run_fold(FOLD, corpus, y, uids, has_gold, ygold):
    torch.manual_seed(MODEL_SEED + FOLD)
    np.random.seed(MODEL_SEED + FOLD)

    # fold split uses SPLIT_SEED, identical to arm A, so the two arms' OOF
    # predictions cover the same validation studies and can be blended
    rng = np.random.RandomState(SPLIT_SEED)
    fold_id = rng.permutation(len(uids)) % N_FOLDS
    tr_i = np.where(fold_id != FOLD)[0]
    va_i = np.where(fold_id == FOLD)[0]
    log("fold {0}: train {1}, val {2}".format(FOLD, len(tr_i), len(va_i)))

    dtr = DS(corpus, [uids[i] for i in tr_i], y[tr_i], True)
    dva = DS(corpus, [uids[i] for i in va_i], y[va_i], False)
    # pin_memory=False: every arm B fold was SIGKILLed (exit -9) by the host OOM
    # killer around step 700, so host RAM is the constraint here and pinned
    # staging buffers are the one part of it we can simply give up.
    ltr = torch.utils.data.DataLoader(
        dtr, batch_size=BATCH_STUDIES, shuffle=True, num_workers=2,
        pin_memory=False, drop_last=True, persistent_workers=False)
    lva = torch.utils.data.DataLoader(
        dva, batch_size=BATCH_STUDIES, shuffle=False, num_workers=2,
        pin_memory=False, persistent_workers=False)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net().to(dev)
    bb = model.bb_name
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
                loss = F.binary_cross_entropy_with_logits(model(x, pres), t)
            scaler.scale(loss / ACCUM).backward()
            if (bi + 1) % ACCUM == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            run += loss.item() * len(x)
            seen += len(x)
            if (bi + 1) % 200 == 0:
                log("ep{0} {1}/{2} loss {3:.4f}  {4}".format(
                    ep, bi + 1, len(ltr), run / max(seen, 1), mem()))

        model.eval()
        P = []
        with torch.no_grad():
            for x, pres, t in lva:
                x, pres = x.to(dev), pres.to(dev)
                with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                    P.append(torch.sigmoid(model(x, pres)).float().cpu().numpy())
        P = np.concatenate(P)

        yv = y[va_i][:len(P)]
        aucs = [auc((yv[:, j] > 0.5).astype(int), P[:, j]) for j in range(N_LAB)]
        macro = float(np.nanmean(aucs))
        log("ep{0} loss {1:.4f} val macro AUC {2:.4f}".format(
            ep, run / max(seen, 1), macro))
        gm = has_gold[va_i][:len(P)]
        if gm.sum() >= 8:
            gv = ygold[va_i][:len(P)][gm]
            ga = [auc(gv[:, j], P[gm][:, j]) for j in range(N_LAB)]
            log("    [gold n={0}] macro {1:.4f}".format(
                int(gm.sum()), float(np.nanmean(ga))))

        if macro > best:
            best, best_oof = macro, P
            torch.save({
                "model": (model.module if hasattr(model, "module") else model).state_dict(),
                "fold": FOLD, "epoch": ep, "macro": macro, "backbone": bb,
                "labels": LABELS, "img": IMG, "slices": TRAIN_SLICES,
            }, os.path.join(OUT, "armB_fold{0}.pt".format(FOLD)))
            log("saved checkpoint (best {0:.4f})".format(best))

    oof = pd.DataFrame(best_oof, columns=LABELS)
    oof.insert(0, "STUDY_UID", [uids[i] for i in va_i][:len(best_oof)])
    oof.to_csv(os.path.join(OUT, "armB_oof_fold{0}.csv".format(FOLD)), index=False)
    log("DONE arm B fold {0}: best {1:.4f} backbone {2}".format(FOLD, best, bb))

    import gc
    del ltr, lva, dtr, dva, model, opt, sched, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
