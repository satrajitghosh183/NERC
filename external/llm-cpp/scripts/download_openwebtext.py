#!/usr/bin/env python3
"""
scripts/download_openwebtext.py

Download the OpenWebText corpus (≈8M documents, ~40 GB raw text) from the
HuggingFace `Skylion007/openwebtext` mirror and write it out as plain
`.txt` chunk files that this repo's `build/prepare_data` binary can
tokenize. We chunk so that prepare_data can stream-tokenize each file in
parallel without ever materializing the whole 40 GB string in memory.

Usage:
    python3 scripts/download_openwebtext.py [output_dir]

    # Default output_dir is the Jetstream volume path:
    #   /media/volume/Prep_and_Voice_Training/data/openwebtext_raw
    python3 scripts/download_openwebtext.py
    python3 scripts/download_openwebtext.py ./data/openwebtext_raw

--- Reads ---
    sys.argv[1] (optional) — output directory.
    HuggingFace cache (~/.cache/huggingface) — the `datasets` library
                       caches the parquet shards here on first run.

--- Writes / Side effects ---
    <out_dir>/chunk_0000.txt, chunk_0001.txt, …  (100k docs per file,
        documents joined with a blank line as paragraph separator).
    <out_dir>/                — created if missing.

--- Calls ---
    datasets.load_dataset (HuggingFace) — actual network IO + parquet
                                          decode happens inside this call.

--- Role in workflow ---
    One-shot data-prep step. After this finishes you typically pipe the
    resulting *.txt files into `./build/prepare_data` to produce a
    tokenized .npy that olmo_train memory-maps at training time.
"""
import os
import sys


def main():
    """Download OpenWebText and shard to plain-text chunk files."""
    # argv[1] (optional): override output directory. Default is the Jetstream
    # data volume — overridden when running on a workstation with no /media.
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/media/volume/Prep_and_Voice_Training/data/openwebtext_raw"
    os.makedirs(out_dir, exist_ok=True)

    # `datasets` is a heavy dependency; bail out with a friendly hint if it
    # isn't installed (rather than crashing with a bare ImportError).
    try:
        from datasets import load_dataset
    except ImportError:
        print("pip install datasets")
        sys.exit(1)

    # Stream-load the corpus. num_proc=8 parallelizes the parquet decode
    # across 8 worker processes — bandwidth-bound on most boxes anyway.
    print(f"Downloading OpenWebText to {out_dir} ...")
    ds = load_dataset("Skylion007/openwebtext", split="train", num_proc=8)
    print(f"Got {len(ds)} documents")

    # Shard into 100k-document chunks. This bounds individual file size to
    # roughly 500 MB, which is friendly to both `prepare_data` parallelism
    # and the underlying filesystem (no single 40 GB file).
    chunk_size = 100000
    for i in range(0, len(ds), chunk_size):
        end = min(i + chunk_size, len(ds))
        # `select` returns a lazy view, not a copy — cheap.
        chunk = ds.select(range(i, end))
        # Zero-pad the chunk index to 4 digits so files sort lexicographically.
        path = os.path.join(out_dir, f"chunk_{i // chunk_size:04d}.txt")
        with open(path, "w") as f:
            # Write each document followed by a blank line; downstream
            # tokenizers treat the blank line as a hard document boundary.
            for row in chunk:
                f.write(row["text"] + "\n\n")
        print(f"  Wrote {path} ({end - i} docs)")

    print("Done")


if __name__ == "__main__":
    main()
