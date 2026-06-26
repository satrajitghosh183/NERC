#!/usr/bin/env python3
"""
From-scratch byte-level BPE trainer for the OmniTrace shader corpus.

WHY: GPT-2's BPE is English-centric and wastes tokens on GLSL (the overnight run's
47.6M "tokens" were inflated by this). We keep GPT-2's byte-level scheme + standard
pretokenization regex (so the result is a drop-in vocab.json + merges.txt for
OLMo-corecpp's BPETokenizer), but LEARN the merges on shader code — so `vec3`,
`float`, `gl_FragCoord`, `fragColor`, `texture`, `fract`, `0.5`, etc. collapse to
single tokens. Net effect: more shader *content* per context window, fewer tokens
to model, better from-scratch learning at the same data scale.

The BPE training itself is implemented from scratch (incremental pair-count updates,
the "no slapping packages" path). `regex` is used only for the GPT-2 pretokenization
pattern; falls back to stdlib `re` (a code-oriented approximation) if unavailable.

USAGE
  # train on the merged corpus (dir of .txt/.frag/.glsl, recursive):
  python3 train_bpe.py --corpus ~/shader_data --vocab-size 16384 --out ~/shader_data/glsl_bpe
  # verify round-trip + measure compression vs GPT-2 on a sample:
  python3 train_bpe.py --self-test
  # then point the trainer at out/{vocab.json,merges.txt} and retokenize.
"""
import argparse, glob, json, os, sys, time
from collections import Counter, defaultdict

try:
    import regex as _re
    GPT2_PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
except ImportError:                                            # stdlib fallback
    import re as _re
    GPT2_PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""


def bytes_to_unicode():
    """GPT-2's reversible byte<->unicode map (keeps tokens printable in vocab.json)."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


B2U = bytes_to_unicode()
PAT = _re.compile(GPT2_PAT)


def pretokenize_freqs(files, max_bytes=0):
    """Scan corpus -> Counter of pre-token (as tuple of byte-unicode chars) -> freq."""
    wf = Counter()
    scanned = 0
    for fp in files:
        try:
            text = open(fp, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        for tok in PAT.findall(text):
            b = tok.encode("utf-8")
            scanned += len(b)
            wf[tuple(B2U[x] for x in b)] += 1
        if max_bytes and scanned > max_bytes:
            break
    return wf, scanned


def train(files, vocab_size, max_bytes=0, verbose=True):
    """Incremental byte-level BPE. Returns (vocab dict, merges list)."""
    wf, scanned = pretokenize_freqs(files, max_bytes)
    if verbose:
        print(f"[bpe] {len(wf)} unique pre-tokens, {scanned/1e6:.1f}MB scanned", flush=True)

    # base vocab: all 256 byte-unicode chars
    vocab = {}
    for b in range(256):
        vocab.setdefault(B2U[b], len(vocab))

    # word state: parallel arrays for speed
    words = [list(w) for w in wf]                  # current segmentation
    freqs = [wf[w] for w in wf]

    # global pair counts + which words contain each pair
    pair_cnt = Counter()
    pair_words = defaultdict(set)
    for wi, seg in enumerate(words):
        f = freqs[wi]
        for a, b in zip(seg, seg[1:]):
            pair_cnt[(a, b)] += f
            pair_words[(a, b)].add(wi)

    merges = []
    t0 = time.time()
    while len(vocab) < vocab_size and pair_cnt:
        (A, B), cnt = max(pair_cnt.items(), key=lambda kv: kv[1])
        if cnt < 2:
            break
        new = A + B
        merges.append((A, B))
        vocab[new] = len(vocab)

        # re-segment only the words that contain (A,B); update pair counts incrementally
        for wi in list(pair_words[(A, B)]):
            seg = words[wi]
            f = freqs[wi]
            # remove this word's current pair contributions
            for a, b in zip(seg, seg[1:]):
                pair_cnt[(a, b)] -= f
                if pair_cnt[(a, b)] <= 0:
                    del pair_cnt[(a, b)]
                    pair_words[(a, b)].discard(wi)
                else:
                    pair_words[(a, b)].discard(wi)
            # merge A,B -> new
            out = []
            i = 0
            while i < len(seg):
                if i < len(seg) - 1 and seg[i] == A and seg[i + 1] == B:
                    out.append(new); i += 2
                else:
                    out.append(seg[i]); i += 1
            words[wi] = out
            # add back the new pair contributions
            for a, b in zip(out, out[1:]):
                pair_cnt[(a, b)] += f
                pair_words[(a, b)].add(wi)

        if verbose and len(vocab) % 1000 == 0:
            print(f"[bpe] vocab={len(vocab)} merges={len(merges)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return vocab, merges


def write_gpt2(vocab, merges, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vocab, open(os.path.join(out_dir, "vocab.json"), "w"))
    with open(os.path.join(out_dir, "merges.txt"), "w") as f:
        f.write("#version: 0.2\n")
        for a, b in merges:
            f.write(f"{a} {b}\n")
    print(f"[bpe] wrote vocab({len(vocab)}) + merges({len(merges)}) -> {out_dir}", flush=True)


def make_encoder(vocab, merges):
    """Return an encode(text)->List[int] using the trained vocab+merges (for testing)."""
    rank = {(a, b): i for i, (a, b) in enumerate(merges)}

    def bpe(word):
        seg = list(word)
        while len(seg) > 1:
            best, bi = None, None
            for i, p in enumerate(zip(seg, seg[1:])):
                r = rank.get(p)
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if bi is None:
                break
            seg[bi:bi + 2] = [seg[bi] + seg[bi + 1]]
        return seg

    def encode(text):
        ids = []
        for tok in PAT.findall(text):
            w = tuple(B2U[x] for x in tok.encode("utf-8"))
            for piece in bpe(w):
                ids.append(vocab[piece])
        return ids

    return encode


# --------------------------------------------------------------------- self-test
SAMPLE = """// Shader: blue fire
// tags: fire, noise, 2d
void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    vec2 uv = fragCoord / iResolution.xy;
    vec3 col = vec3(0.0);
    float n = fract(sin(dot(uv, vec2(12.9898, 78.233))) * 43758.5453);
    col += vec3(0.1, 0.3, 0.9) * smoothstep(0.2, 0.9, n);
    fragColor = vec4(col, 1.0);
}
"""


def self_test():
    files = []
    import tempfile
    d = tempfile.mkdtemp()
    # replicate the sample to give the trainer something to merge
    for i in range(50):
        fp = os.path.join(d, f"s{i}.txt")
        open(fp, "w").write(SAMPLE)
        files.append(fp)
    vocab, merges = train(files, vocab_size=600, verbose=False)
    enc = make_encoder(vocab, merges)
    ids = enc(SAMPLE)
    # round-trip: decode ids back to text via reverse vocab + byte map
    u2b = {u: b for b, u in B2U.items()}
    inv = {i: t for t, i in vocab.items()}
    decoded = bytes(u2b[ch] for i in ids for ch in inv[i]).decode("utf-8", errors="replace")
    ok = decoded == SAMPLE
    raw_bytes = len(SAMPLE.encode("utf-8"))
    print(f"[self-test] vocab={len(vocab)} merges={len(merges)}")
    print(f"[self-test] {raw_bytes} bytes -> {len(ids)} tokens "
          f"({raw_bytes/len(ids):.2f} bytes/token)")
    print(f"[self-test] round-trip exact: {'PASS' if ok else 'FAIL'}")
    # show a few GLSL tokens that became single units
    glsl_units = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])
                  if any(k in t for k in ("vec", "float", "frag", "col", "uv"))][:12]
    print(f"[self-test] learned GLSL tokens (sample): {glsl_units}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="From-scratch GLSL byte-level BPE trainer")
    ap.add_argument("--corpus", help="dir of shader text (recursive: .txt/.frag/.glsl)")
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--out", default=os.path.expanduser("~/shader_data/glsl_bpe"))
    ap.add_argument("--max-bytes", type=int, default=0, help="cap corpus bytes (0=all)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if self_test() else 1)
    if not args.corpus:
        sys.exit("ERROR: --corpus required (or use --self-test)")
    files = []
    for ext in ("*.txt", "*.frag", "*.glsl", "*.fragment", "*.comp"):
        files += glob.glob(os.path.join(args.corpus, "**", ext), recursive=True)
    if not files:
        sys.exit(f"ERROR: no shader text files under {args.corpus}")
    print(f"[bpe] {len(files)} corpus files", flush=True)
    vocab, merges = train(files, args.vocab_size, args.max_bytes)
    write_gpt2(vocab, merges, args.out)


if __name__ == "__main__":
    main()
