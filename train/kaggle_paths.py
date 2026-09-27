"""Locate Kaggle inputs by searching for a known filename.

Kaggle mounts inputs in two different layouts depending on account/environment:
    /kaggle/input/<slug>/file                       (flat)
    /kaggle/input/datasets/<owner>/<slug>/file      (nested, datasets)
    /kaggle/input/notebooks/<owner>/<slug>/file     (nested, kernel outputs)

Hardcoding either layout silently yields "file not found" or, worse, a partially
attached corpus. This searches instead, and fails loudly with the actual tree.
"""
import os

ROOT = "/kaggle/input"


def walk_tree(max_depth=4):
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        depth = dirpath[len(ROOT):].count(os.sep)
        if depth >= max_depth:
            dirnames[:] = []
        out.append((dirpath, sorted(dirnames), sorted(filenames)[:12]))
    return out


def print_tree(max_depth=3):
    print("== /kaggle/input tree (depth {0}) ==".format(max_depth))
    for dirpath, dirnames, filenames in walk_tree(max_depth):
        depth = dirpath[len(ROOT):].count(os.sep)
        pad = "  " * depth
        print("{0}{1}/".format(pad, os.path.basename(dirpath) or ROOT))
        for f in filenames:
            print("{0}  {1}".format(pad, f))


def find_file(filename, required=True):
    """Return full path of the first file matching `filename` anywhere under
    /kaggle/input, else None (or raise)."""
    for dirpath, _dirnames, filenames in os.walk(ROOT):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    if required:
        print_tree()
        raise SystemExit("could not find {0} under {1}".format(filename, ROOT))
    return None


def find_dir_containing(filename, required=True):
    p = find_file(filename, required=required)
    return os.path.dirname(p) if p else None
