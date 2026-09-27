"""
Per-label blend weights for merging our arms into the reference submission.

HOW THESE WERE DERIVED, AND WHAT THEY ARE NOT
---------------------------------------------
Two measured quantities from the complete 5-fold arm A OOF (4407 studies, 58 gold,
tools/eval_oof.py):

  absolute  our arm's gold AUC per label -- how good we are at all
  delta     our arm's gold AUC minus the LLM label's gold AUC -- whether we are
            reading something out of the pixels that the radiologist did not write

A high delta alone is not a reason to weight heavily: Synovitis has a good delta
(+0.057) but a weak absolute AUC (0.840), so it earns a moderate weight, not a
large one. Both have to be respectable.

THE LIMIT OF THIS EVIDENCE -- READ BEFORE TUNING
------------------------------------------------
delta compares our arm against the LLM LABELS, not against the reference baseline
we are blending into. The baseline is a strong image model in its own right and
also extracts pixel signal. We have NO out-of-fold predictions for it -- it was
trained on these same studies, so its predictions here are leaked and cannot be
used to fit anything. Therefore:

  * these weights are PRINCIPLED, NOT FITTED
  * "our arm beats the label on Effusion" does NOT establish "our arm beats the
    baseline on Effusion"
  * every weight is deliberately below 0.5, so the baseline stays the senior
    signal on all twelve labels and a wrong guess costs little
  * the weights must be confirmed by leaderboard probing before the final pick

A 50/50 blend measured on a 1-epoch arm B probe came out 0.015 WORSE than arm A
alone, which looked like evidence against blending. It was not -- it was evidence
that the probe was undertrained. With both arms fully trained the same 50/50 blend
is +0.0123 BETTER. Do not draw blend conclusions from an undertrained arm.

Also measured and rejected: out-of-fold LABEL REPAIR. Retraining arm A on repaired
Synovitis and Effusion targets moved gold macro 0.8955 -> 0.8891, and the two
repaired labels were themselves the biggest losers (Synovitis -0.038, Effusion
-0.015). The v1 labels are the ones in use. See rsna-knee-oof-findings.md.

Gold is ~3x enriched and several labels have under 15 positives, so treat the
ordering as real and the magnitudes as approximate.
"""

# label -> (weight on our arms, combined-arm gold AUC, that AUC minus label AUC)
#
# UPDATED after both arms finished. The numbers are now for the 50/50 rank blend
# of arm A + arm B (gold macro 0.9078), not arm A alone (0.8955). The blend is
# materially stronger on several labels than arm A was, so those weights went up:
#   ACL             0.912 -> 0.958
#   Medial Meniscus 0.934 -> 0.963
#   Fracture        0.932 -> 0.954
#   Lateral Meniscus 0.852 -> 0.878
# Measured: mean rank correlation between arms 0.756, and the blend beats the best
# single arm by +0.0123 macro -- roughly 3x the +0.004 reported by the community
# and 4x the 0.003 noise floor.
BLEND_W = {
    # strong absolutely AND reading what the report missed -> lean in hardest
    "Effusion":         (0.38, 0.970, +0.150),
    "Medial OA":        (0.32, 0.967, +0.062),
    "Contusion":        (0.32, 0.951, +0.086),
    "Fracture":         (0.30, 0.954, +0.064),
    "Baker's":          (0.28, 0.902, +0.071),
    # strong absolutely, roughly level with the label
    "Medial Meniscus":  (0.25, 0.963, +0.008),
    "Synovitis":        (0.25, 0.853, +0.070),
    "ACL":              (0.20, 0.958, -0.038),
    "Lateral Meniscus": (0.18, 0.878, +0.003),
    "Lateral OA":       (0.15, 0.849, +0.002),
    # the report is decisively better: ligament tears and PF OA are stated
    # explicitly, and our arms stay well behind. Near-zero, not zero, purely for
    # variance reduction.
    "MCL":              (0.08, 0.828, -0.164),
    "PF OA":            (0.08, 0.819, -0.126),
}

# Conservative fallback for the two-pick hedge: one submission at the weights
# above, one at everything halved. If the LB says the blend helps, the full
# weights are the aggressive pick; if it says it hurts, the halved one limits
# the damage and baseline-only remains the safe pick.
HALVED = {k: (round(v[0] / 2, 3), v[1], v[2]) for k, v in BLEND_W.items()}

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
          "Medial OA", "Lateral OA", "PF OA", "Effusion",
          "Synovitis", "Baker's", "Contusion", "Fracture"]


def as_vector(table=None):
    """Weights in _RSNA_LABELS order, which is the notebook's column order."""
    t = table or BLEND_W
    missing = [l for l in LABELS if l not in t]
    if missing:
        raise SystemExit("no weight for: {0}".format(missing))
    return [t[l][0] for l in LABELS]


if __name__ == "__main__":
    print("{0:18s} {1:>7s} {2:>8s} {3:>8s}".format(
        "finding", "weight", "our_auc", "vs_label"))
    print("-" * 45)
    for l in LABELS:
        w, a, d = BLEND_W[l]
        print("{0:18s} {1:7.2f} {2:8.3f} {3:+8.3f}".format(l, w, a, d))
    print()
    print("mean weight on our arms: {0:.3f}".format(
        sum(w for w, _, _ in BLEND_W.values()) / len(BLEND_W)))
    print("max weight:              {0:.2f}  (baseline stays senior everywhere)".format(
        max(w for w, _, _ in BLEND_W.values())))
    print()
    print("vector in _RSNA_LABELS order:")
    print(as_vector())
    print("halved (hedge pick):")
    print(as_vector(HALVED))
