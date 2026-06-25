#!/usr/bin/env python3
"""concat_shards.py — stitch a directory of tokenized .npy shards into one
big .npy file the existing zwt TokenLoader (single-file mmap) can consume.

Usage:
  python3 zwt/scripts/concat_shards.py <shards_dir> <out_path> [--exclude '*.tmp.npy']

Defaults:
  * Sorts shards lexicographically (stable order).
  * Skips files matching *.tmp.npy and any subdirectories.
  * Asserts dtype consistency across shards (fails loud if mixed).
  * Streams the copy via np.memmap on both ends — no full-data RAM load.
    A 26 GB concat is bound by SSD throughput, not memory.

Notes:
  * Output file is a real .npy (proper header), not raw bytes — np.load works.
  * 1-D shards are concatenated end-to-end. 2-D shards (rows x seq) are
    flattened first so the result is a single 1-D token stream — that's
    what TokenLoader expects.
  * Existing output file is overwritten without prompt. Make sure the
    target path is what you want.

Example:
  python3 zwt/scripts/concat_shards.py \\
    /home/exouser/data/dolma-tokenized \\
    /home/exouser/data/dolma_combined.npy
  export ZWT_TOKENS=/home/exouser/data/dolma_combined.npy
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import os
import sys
import time

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards_dir", help="directory containing *.npy shards")
    ap.add_argument("out_path",   help="path to write the combined .npy file")
    ap.add_argument("--exclude", action="append", default=["*.tmp.npy"],
                    help="glob pattern(s) to exclude (repeatable). "
                         "Default: *.tmp.npy")
    ap.add_argument("--pattern", default="*.npy",
                    help="shard file glob (default: *.npy)")
    ap.add_argument("--dry-run", action="store_true",
                    help="inspect shards (count, dtype, total tokens, GB) "
                         "and exit without writing the output file")
    args = ap.parse_args()

    shards_dir = os.path.abspath(args.shards_dir)
    if not os.path.isdir(shards_dir):
        print(f"concat_shards: not a directory: {shards_dir}", file=sys.stderr)
        return 1

    all_paths = sorted(glob.glob(os.path.join(shards_dir, args.pattern)))
    files = []
    for p in all_paths:
        if not os.path.isfile(p):
            continue
        bn = os.path.basename(p)
        skip = any(fnmatch.fnmatch(bn, ex) for ex in args.exclude)
        if skip:
            print(f"  skip (excluded): {bn}")
            continue
        files.append(p)

    if not files:
        print(f"concat_shards: no shards matched {args.pattern} in {shards_dir}",
              file=sys.stderr)
        return 1

    # Pass 1: validate dtype + sum element count.
    dtype = None
    total_elems = 0
    shapes: list[tuple] = []
    print(f"{len(files)} shards in {shards_dir}")
    for f in files:
        a = np.load(f, mmap_mode="r")
        if dtype is None:
            dtype = a.dtype
        elif a.dtype != dtype:
            print(f"concat_shards: dtype mismatch — "
                  f"{os.path.basename(f)} is {a.dtype}, expected {dtype}",
                  file=sys.stderr)
            return 1
        shapes.append(tuple(a.shape))
        total_elems += int(a.size)

    total_bytes = total_elems * dtype.itemsize
    print(f"  dtype: {dtype}")
    print(f"  total: {total_elems:,} tokens, {total_bytes/1e9:.2f} GB")
    print(f"  out:   {args.out_path}")

    if args.dry_run:
        print("--dry-run: not writing. Drop the flag to actually concat.")
        return 0

    # Pre-flight free-space check on the output dir. Don't start a
    # multi-minute copy that will fail half-way.
    out_dir = os.path.dirname(os.path.abspath(args.out_path)) or "."
    st = os.statvfs(out_dir)
    free_bytes = st.f_bavail * st.f_frsize
    if free_bytes < total_bytes + (64 << 20):  # +64 MiB headroom
        print(f"concat_shards: only {free_bytes/1e9:.1f} GB free in {out_dir} — "
              f"need {total_bytes/1e9:.1f} GB. Aborting.", file=sys.stderr)
        return 1

    # Pass 2: open output as np.memmap, stream shards in. open_memmap writes
    # the .npy header for us so the resulting file is np.load-compatible.
    out = np.lib.format.open_memmap(
        args.out_path, mode="w+", dtype=dtype, shape=(total_elems,))

    written = 0
    t0 = time.time()
    for i, f in enumerate(files, start=1):
        a = np.load(f, mmap_mode="r").reshape(-1)  # flatten 1-D or 2-D
        n = int(a.size)
        out[written:written + n] = a
        written += n
        secs = time.time() - t0
        rate = (written * dtype.itemsize) / max(secs, 1e-9) / 1e9
        print(f"  [{i:>2}/{len(files)}] {os.path.basename(f)}  "
              f"+{n:>11,} tok  ({written:,}/{total_elems:,})  "
              f"{rate:5.2f} GB/s")

    out.flush()
    del out  # release the memmap so the file handle closes cleanly

    elapsed = time.time() - t0
    print(f"\nDone: {args.out_path}  "
          f"({total_bytes/1e9:.2f} GB in {elapsed:.1f}s, "
          f"{total_bytes/elapsed/1e9:.2f} GB/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
