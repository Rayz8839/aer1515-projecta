#!/usr/bin/env python3
"""Download the 10 CA-1M validation sequences used by the Boxer paper.

The list is from Boxer, Supp. B.1: "the first 10 sequences in the validation split".
Each tar is extracted to <out>/ca1m-val-<id>/, which is where boxer's CALoader
looks (boxer/sample_data/ by default). About 5 GB in total.

CA-1M is released under CC-BY-NC-ND 4.0. Pass --accept-license to confirm you
accept those terms (the script won't prompt, so it can run unattended).

    python ca1m_eval/download_ca1m_eval.py --accept-license
    python ca1m_eval/download_ca1m_eval.py --accept-license --ids 42897552   # just one
"""

import argparse
import glob
import os
import shutil
import sys
import tarfile
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "boxer", "sample_data")
URL = "https://ml-site.cdn-apple.com/datasets/ca1m/val/ca1m-val-{vid}.tar"

# Boxer paper, Supplementary B.1 (in the paper's order).
BOXER_VAL_IDS = [
    "45662921", "45261179", "47115543", "45261143", "45261615",
    "42897545", "45261133", "42897552", "45663113", "42897521",
]


def is_complete(seq_dir: str) -> bool:
    has_world = glob.glob(os.path.join(seq_dir, "**/world.gt/instances.json"), recursive=True)
    has_imgs = glob.glob(os.path.join(seq_dir, "**/*.wide/image.png"), recursive=True)
    return bool(has_world) and bool(has_imgs)


def download(url: str, dest: str) -> None:
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write(f"\r  {done / 2**20:7.1f}/{total / 2**20:.1f} MB")
                sys.stdout.flush()
    print()
    os.replace(tmp, dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT, help="extraction root (default: boxer/sample_data)")
    ap.add_argument("--ids", nargs="*", default=BOXER_VAL_IDS, help="video ids (default: Boxer's 10)")
    ap.add_argument("--accept-license", action="store_true", help="accept CA-1M CC-BY-NC-ND 4.0 terms")
    ap.add_argument("--keep-tar", action="store_true", help="keep the .tar after extraction")
    args = ap.parse_args()

    if not args.accept_license:
        sys.exit("CA-1M is CC-BY-NC-ND 4.0 (https://creativecommons.org/licenses/by-nc-nd/4.0/). "
                 "Re-run with --accept-license.")

    os.makedirs(args.out, exist_ok=True)
    for i, vid in enumerate(args.ids, 1):
        name = f"ca1m-val-{vid}"
        seq_dir = os.path.join(args.out, name)
        print(f"[{i}/{len(args.ids)}] {name}")
        if is_complete(seq_dir):
            print("  already extracted, skipping")
            continue
        tar_path = os.path.join(args.out, name + ".tar")
        if not os.path.exists(tar_path):
            download(URL.format(vid=vid), tar_path)
        print("  extracting...")
        shutil.rmtree(seq_dir, ignore_errors=True)
        os.makedirs(seq_dir)
        with tarfile.open(tar_path) as tar:
            tar.extractall(seq_dir, filter="data")
        if not is_complete(seq_dir):
            sys.exit(f"  {seq_dir} looks incomplete after extraction")
        n = len(glob.glob(os.path.join(seq_dir, "**/*.wide/image.png"), recursive=True))
        print(f"  ok: {n} frames")
        if not args.keep_tar:
            os.remove(tar_path)
    print("done")


if __name__ == "__main__":
    main()
