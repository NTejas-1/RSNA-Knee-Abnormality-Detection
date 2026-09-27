-- RSNA Knee: label-extraction workspace on account ORUWQCB-CG54164
-- Reports are Competition Data. Host permits sending them to a hosted LLM for
-- label extraction (discussion/733965, Rules 2.6.b) but redistribution is an
-- OPEN question -- never publish anything derived from this schema publicly.

CREATE DATABASE IF NOT EXISTS RSNA_KNEE;
CREATE SCHEMA IF NOT EXISTS RSNA_KNEE.LABELS;

USE DATABASE RSNA_KNEE;
USE SCHEMA LABELS;

-- Raw reports plus the 58 gold rows (gold cols are NULL for the other 4349).
CREATE TABLE IF NOT EXISTS REPORTS (
    STUDY_UID           VARCHAR PRIMARY KEY,
    REPORT              VARCHAR,
    REPORT_CHARS        NUMBER,
    GOLD_ACL            NUMBER,
    GOLD_MCL            NUMBER,
    GOLD_MEDIAL_MENISCUS  NUMBER,
    GOLD_LATERAL_MENISCUS NUMBER,
    GOLD_MEDIAL_OA      NUMBER,
    GOLD_LATERAL_OA     NUMBER,
    GOLD_PF_OA          NUMBER,
    GOLD_EFFUSION       NUMBER,
    GOLD_SYNOVITIS      NUMBER,
    GOLD_BAKERS         NUMBER,
    GOLD_CONTUSION      NUMBER,
    GOLD_FRACTURE       NUMBER,
    IS_GOLD             BOOLEAN
);

-- One row per study per extraction run. RAW_JSON keeps the model's full reply so
-- we can re-parse without re-paying for inference.
CREATE TABLE IF NOT EXISTS EXTRACTED_RAW (
    STUDY_UID      VARCHAR,
    RUN_TAG        VARCHAR,
    MODEL          VARCHAR,
    RAW_JSON       VARCHAR,
    EXTRACTED_AT   TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (STUDY_UID, RUN_TAG)
);

-- Parsed long form: 12 findings x {state, prob} per study.
-- STATE is the whole point of this design: it separates "radiologist said no"
-- from "radiologist said nothing", so Phase 3 can repair only the silent cells.
CREATE TABLE IF NOT EXISTS EXTRACTED_LABELS (
    STUDY_UID   VARCHAR,
    RUN_TAG     VARCHAR,
    FINDING     VARCHAR,
    STATE       VARCHAR,   -- positive | explicit_negative | not_mentioned
    PROB        FLOAT,     -- fractional, never rounded to 0/1
    PRIMARY KEY (STUDY_UID, RUN_TAG, FINDING)
);

CREATE STAGE IF NOT EXISTS RSNA_STAGE
    FILE_FORMAT = (TYPE = PARQUET);
