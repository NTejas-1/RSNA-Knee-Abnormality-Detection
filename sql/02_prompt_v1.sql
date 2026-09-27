-- RSNA Knee 3-state label extraction prompt, v1.
-- Creates a view that renders one prompt per study.
--
-- Two design choices that matter:
--
-- 1. THREE-STATE OUTPUT. Every public label set collapses "radiologist said no"
--    and "radiologist said nothing" into a single 0. That conflation is exactly
--    why Synovitis sits at ~0.797 AUC for everyone: the reports are silent, not
--    negative, so the 0s are systematic false negatives. Emitting state
--    separately lets Phase 3 repair ONLY the silent cells with out-of-fold
--    predictions, which is the one intervention reported to move that label.
--
-- 2. HOST THRESHOLDS VERBATIM. The pinned competition overview specifies
--    quantitative criteria (>50% fibre disruption, >=1cm of >50% cartilage loss,
--    surface contact on >=2 images, acute-only MCL) and states that borderline
--    findings were graded NEGATIVE to favour specificity. Most extractors ignore
--    these and mislabel the boundary cases.
--
-- Short JSON keys ("s"/"p", p/n/u) are deliberate: they cut output tokens ~3x
-- across 4407 calls at Opus pricing.

USE DATABASE RSNA_KNEE;
USE SCHEMA LABELS;

CREATE OR REPLACE VIEW PROMPTS_V1 AS
SELECT
    STUDY_UID,
    IS_GOLD,
    $$You are a musculoskeletal radiologist labelling a knee MRI report for a research dataset.

The report may be in any language. Read it in its original language; do not translate before deciding.

For EACH of the 12 findings below, output two things:
  "s" = state, one of:
        "p" = the report affirms this finding is PRESENT and it meets the threshold
        "n" = the report EXPLICITLY states this finding is absent / normal / intact
        "u" = the report does not mention this finding at all (silence, not denial)
  "p" = probability in [0,1] that a radiologist reviewing the IMAGES would label it 1.
        Use the full continuous range. Do NOT round to 0 or 1.
        If s="u", give your best prior for an unmentioned finding in this clinical
        context rather than defaulting to 0 -- silence in a report is weak evidence
        of absence, not proof.

THRESHOLDS (apply these exactly; they define the reference labels):
- Borderline or "on the fence" findings were graded NEGATIVE to favour specificity.
- ACL: high-grade partial or full-thickness tear -- complete discontinuity, or >50% of
  fibres disrupted. Mild signal change, degeneration or thickening WITHOUT
  discontinuity is negative.
- MCL: high-grade partial or complete ACUTE tear with disrupted fibres and edema.
  Low-grade sprains and chronic or remote stress changes are negative.
- Medial Meniscus / Lateral Meniscus: abnormal signal definitely contacting the
  meniscal surface on AT LEAST TWO images, or a morphologic abnormality (truncated,
  diminutive, or displaced fragment). Intrasubstance degeneration that does not
  reach the surface is negative.
- Medial OA / Lateral OA / PF OA: a moderate or large area (roughly 1 cm or greater)
  of high-grade cartilage loss, meaning GREATER THAN 50% of cartilage thickness, in
  that specific compartment. Medial = medial tibiofemoral, Lateral = lateral
  tibiofemoral, PF = patellofemoral.
- Effusion: a MODERATE or LARGE amount of fluid distending the joint. Trace or small
  physiologic fluid is negative.
- Synovitis: inflammation and thickening of the synovial lining. Frequently present
  but UNREPORTED -- be careful to use "u" rather than "n" when the report is simply
  silent on the synovium.
- Baker's: a MODERATE or LARGE fluid collection in the characteristic popliteal
  location. Small cysts are negative.
- Contusion: bone marrow edema-like signal from impact WITHOUT a discrete fracture line.
- Fracture: an ACUTE cortical break or fracture line.

Return ONLY minified JSON, no prose, no markdown fence, with exactly these 12 keys:
{"ACL":{"s":"p|n|u","p":0.0},"MCL":{...},"MedialMeniscus":{...},"LateralMeniscus":{...},"MedialOA":{...},"LateralOA":{...},"PFOA":{...},"Effusion":{...},"Synovitis":{...},"Bakers":{...},"Contusion":{...},"Fracture":{...}}

REPORT:
$$ || REPORT AS PROMPT
FROM REPORTS;
