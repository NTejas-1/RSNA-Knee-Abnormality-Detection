"""
Inspect the reference submission notebook without touching it.

The notebook is minified single-line JSON, so the Read tool cannot show it -- it
has to be parsed. This prints a per-cell map plus the lines that matter for
integration: where the submission is written, what holds the predictions, and
where the integrity gates live.

The gates are the reason we APPEND rather than edit. The notebook asserts
_DINOV2_MATCHED_MEMBERS == 20, checks SHA256 weight fingerprints and sets
V18_CALIBRATOR_APPLIED; editing inside a gated arm risks silently disabling the
0.940 baseline we are building on.
"""
import io
import json
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(HERE, "ref_kernel",
                  "rsna-knee-speedy-raptors-coatnet-d4-sota.ipynb")

KEY = re.compile(
    r"submission|to_csv|sample_submission|SUBMISSION|"
    r"_DINOV2_MATCHED_MEMBERS|V18_CALIBRATOR_APPLIED|sha256|"
    r"^\s*def \w+|^\s*class \w+|LABELS|TARGET|pred", re.I)


def main():
    nb = json.load(io.open(NB, encoding="utf-8"))
    cells = nb["cells"]
    print("notebook: {0} cells".format(len(cells)))
    print()
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        lines = src.split("\n")
        head = next((l for l in lines if l.strip()), "")
        print("--- cell {0} [{1}] {2} lines, {3} chars".format(
            i, c.get("cell_type"), len(lines), len(src)))
        print("      first: {0}".format(head.strip()[:105]))
        hits = [l.strip() for l in lines if KEY.search(l)]
        for h in hits[:7]:
            print("      | {0}".format(h[:105]))
        if len(hits) > 7:
            print("      | ... {0} more matching lines".format(len(hits) - 7))


if __name__ == "__main__":
    main()
