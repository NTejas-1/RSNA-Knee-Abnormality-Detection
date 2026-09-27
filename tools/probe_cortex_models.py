"""Probe which Cortex LLM models are actually served in this account's region."""
import subprocess
import sys

CONN = "oruwqcb-kaggle"

MODELS = [
    "claude-opus-4-5",
    "claude-opus-4-1",
    "claude-4-opus",
    "claude-sonnet-4-5",
    "claude-3-7-sonnet",
    "claude-3-5-sonnet",
    "openai-gpt-4.1",
    "openai-gpt-5",
    "llama3.1-70b",
    "mistral-large2",
]


def probe(model):
    q = "SELECT AI_COMPLETE('{0}', 'Reply with exactly: OK') AS r".format(model)
    p = subprocess.run(
        ["snow", "sql", "-c", CONN, "-q", q],
        capture_output=True, text=True, shell=True,
    )
    out = (p.stdout or "") + (p.stderr or "")
    low = out.lower()
    available = ("| r" in low or "ok" in low) and "error" not in low
    reason = ""
    for ln in out.splitlines():
        l = ln.lower()
        if "not available" in l or "unknown model" in l or "error" in l or "not supported" in l:
            reason = ln.strip()[:120]
            break
    return available, reason


if __name__ == "__main__":
    print("region probe via connection:", CONN)
    for m in MODELS:
        ok, reason = probe(m)
        print("{0:6s} {1:20s} {2}".format("AVAIL" if ok else "NO", m, reason))
        sys.stdout.flush()
