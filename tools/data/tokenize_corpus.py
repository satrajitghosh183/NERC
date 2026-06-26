#!/usr/bin/env python3
"""
Tokenize the merged shader corpus to a uint16 .npy with a trained BPE
(GLSL-aware or GPT-2), the format OLMo-corecpp's TokenDataset memory-maps.

Also reports the real chars/token and total token count — the headline number for
the tokenizer win (GPT-2 gets ~1.99 chars/token on this corpus; the GLSL BPE should
be much higher, i.e. fewer tokens for the same shader content).

USAGE
  python3 tokenize_corpus.py --corpus corpus_merged/corpus.txt \
      --bpe glsl_bpe --out corpus_merged/shaders_glslbpe.npy
  # compare against GPT-2 with --bpe <dir with gpt2 vocab.json+merges.txt>
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bpe import make_encoder, B2U  # reuse the exact BPE encode + byte map

EOT = "<|endoftext|>"


def load_bpe(bpe_dir):
    vocab = json.load(open(os.path.join(bpe_dir, "vocab.json")))
    merges = []
    with open(os.path.join(bpe_dir, "merges.txt")) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            a, b = line.rstrip("\n").split(" ")
            merges.append((a, b))
    return vocab, merges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--bpe", required=True, help="dir with vocab.json + merges.txt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vocab, merges = load_bpe(args.bpe)
    encode = make_encoder(vocab, merges)
    eot_id = vocab.get(EOT, len(vocab))          # reserve an id for the doc separator
    vocab_size = max(len(vocab), eot_id + 1)

    text = open(args.corpus, encoding="utf-8", errors="replace").read()
    docs = text.split(EOT)
    ids = []
    for i, doc in enumerate(docs):
        if doc.strip():
            ids.extend(encode(doc))
            ids.append(eot_id)
        if (i + 1) % 5000 == 0:
            print(f"  tokenized {i+1}/{len(docs)} docs, {len(ids):,} tokens", flush=True)

    arr = np.array(ids, dtype=np.uint16 if vocab_size <= 65536 else np.uint32)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    arr.tofile(args.out) if args.out.endswith(".bin") else np.save(args.out, arr)

    chars = len(text)
    cpt = chars / len(ids) if ids else 0
    print(f"\n[tokenize] {len(docs):,} docs -> {len(ids):,} tokens")
    print(f"[tokenize] vocab_size={vocab_size}  chars/token={cpt:.3f}  "
          f"(GPT-2 baseline on this corpus ~1.99)")
    print(f"[tokenize] saved {args.out}  ({arr.nbytes/1e6:.1f}MB, dtype={arr.dtype})")
    json.dump({"tokens": len(ids), "docs": len(docs), "vocab_size": vocab_size,
               "chars_per_token": round(cpt, 3), "bpe": args.bpe},
              open(os.path.splitext(args.out)[0] + "_meta.json", "w"), indent=2)


if __name__ == "__main__":
    main()
