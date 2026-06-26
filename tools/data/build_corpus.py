#!/usr/bin/env python3
"""
Unified shader-corpus builder for the OmniTrace shader-LM.

Ingests every source into ONE normalized, deduped corpus in the `// Shader: <name>`
instruction format (the format the eval prompts use — fixing the mismatch that gave the
from-scratch model 0/8). Sources (all key-free):

  --vipitis DIR     github.com/Vipitis/shadertoys-dataset  data/annotated/*.jsonl
                    (rich per-pass fields: image_code/common_code/buffer_*_code + metadata)
  --mizu JSONL      hf datasets/mizuamedesu/shader-sft-dataset  (chat: system/user/assistant,
                    assistant = ```glsl ...``` ; user = natural-language prompt)
  --shaders21k DIR  the on-box shaders21k codes/ (*.fragment) — optional, when on the box
  --extra DIR       any dir of *.txt/*.frag/*.glsl already in `// Shader:` form

Dedup: by Shadertoy id (when known) AND by normalized-code SHA1. Filters: min code length,
must contain mainImage/void main. Outputs:
  out/corpus.txt        all records joined by <|endoftext|>  (for LM tokenization)
  out/train.jsonl       {prompt, code, name, source, id}      (for SFT / RL / eval)
  out/val.jsonl         held-out split
  out/stats.json        counts, dedup rate, byte/char totals, GPT-2-token estimate
"""
import argparse, ast, glob, hashlib, json, os, re, sys
from collections import Counter

EOT = "<|endoftext|>"
CODE_FENCE = re.compile(r"```(?:glsl|c|cpp)?\s*(.*?)```", re.S)
THINK = re.compile(r"<think>.*?</think>", re.S)
HDR_ID = re.compile(r"//\s*ID:\s*([A-Za-z0-9]+)")
HDR_NAME = re.compile(r"//\s*(?:Shadertoy|Shader|Title):\s*(.+)")


# ------------------------------------------------------------------- normalize
def norm_code_hash(code):
    """Hash on whitespace-normalized code so trivial reformatting still dedupes."""
    c = re.sub(r"\s+", " ", code).strip()
    return hashlib.sha1(c.encode("utf-8", "replace")).hexdigest()


def has_entry(code):
    return ("mainImage" in code) or re.search(r"\bvoid\s+main\s*\(", code)


def make_record(name, desc, tags, code):
    """Assemble the canonical instruction record. Returns (record_text, prompt)."""
    name = (name or "untitled").strip().replace("\n", " ")[:120]
    header = f"// Shader: {name}\n"
    prompt = f"// Shader: {name}\n"
    desc_lines = (desc or "").strip().splitlines()
    if desc_lines:
        d = desc_lines[0][:200]
        header += f"// {d}\n"; prompt += f"// {d}\n"
    if tags:
        t = ", ".join(tags)[:200]
        header += f"// tags: {t}\n"; prompt += f"// tags: {t}\n"
    return header + code.rstrip() + "\n", prompt


# --------------------------------------------------------------------- sources
def parse_tags(raw):
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(t) for t in raw]
    try:
        v = ast.literal_eval(raw)
        return [str(t) for t in v] if isinstance(v, (list, tuple)) else [str(v)]
    except Exception:
        return [raw] if isinstance(raw, str) else []


def from_vipitis(d):
    """Yield dicts from Vipitis annotated/*.jsonl (assemble multipass code)."""
    for fp in sorted(glob.glob(os.path.join(d, "data", "annotated", "*.jsonl"))
                     or glob.glob(os.path.join(d, "*.jsonl"))):
        for line in open(fp, errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            parts = [r.get(k, "") for k in
                     ("common_code", "buffer_a_code", "buffer_b_code", "buffer_c_code",
                      "buffer_d_code", "cube_a_code", "image_code")]
            code = "\n\n".join(p for p in parts if p and p.strip())
            if not code.strip():
                continue
            yield {"id": r.get("id"), "name": r.get("name"),
                   "desc": r.get("description", ""), "tags": parse_tags(r.get("tags")),
                   "code": code, "source": "vipitis", "license": r.get("license", "")}


def from_mizu(fp):
    """Yield dicts from mizuamedesu chat jsonl (strip fence + <think>)."""
    for line in open(fp, errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            msgs = json.loads(line)["messages"]
        except Exception:
            continue
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        asst = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        asst = THINK.sub("", asst)
        m = CODE_FENCE.search(asst)
        code = (m.group(1) if m else asst).strip()
        if not code:
            continue
        sid = (HDR_ID.search(code) or [None, None])[1] if HDR_ID.search(code) else None
        nm = HDR_NAME.search(code)
        name = nm.group(1).strip() if nm else (user[:60] if user else "untitled")
        yield {"id": sid, "name": name, "desc": user.strip(), "tags": [],
               "code": code, "source": "mizuamedesu", "license": ""}


def from_frag_dir(d, source):
    """Yield dicts from a dir of raw shader files (shaders21k codes/, etc.)."""
    for fp in glob.glob(os.path.join(d, "*")):
        if not os.path.isfile(fp):
            continue
        try:
            code = open(fp, errors="replace").read()
        except Exception:
            continue
        if code.strip():
            yield {"id": os.path.basename(fp).split(".")[0], "name": os.path.basename(fp),
                   "desc": "", "tags": [], "code": code, "source": source, "license": ""}


# ------------------------------------------------------------------------ main
def build(sources, out, min_len, val_frac):
    os.makedirs(out, exist_ok=True)
    seen_hash, seen_id = set(), set()
    kept, dup, thin, noentry = 0, 0, 0, 0
    by_source = Counter()
    chars = 0
    rows = []
    for gen in sources:
        for r in gen:
            code = r["code"]
            if len(code) < min_len:
                thin += 1; continue
            if not has_entry(code):
                noentry += 1; continue
            sid = r.get("id")
            if sid and sid in seen_id:
                dup += 1; continue
            h = norm_code_hash(code)
            if h in seen_hash:
                dup += 1; continue
            seen_hash.add(h)
            if sid:
                seen_id.add(sid)
            record, prompt = make_record(r["name"], r["desc"], r["tags"], code)
            rows.append({"prompt": prompt, "code": code, "record": record,
                         "name": r["name"], "source": r["source"], "id": sid})
            kept += 1; by_source[r["source"]] += 1; chars += len(record)

    # deterministic shuffle (hash of id/name) so split is stable across runs
    rows.sort(key=lambda x: hashlib.sha1((str(x["id"]) + x["name"]).encode()).hexdigest())
    n_val = int(len(rows) * val_frac)
    val, train = rows[:n_val], rows[n_val:]

    with open(os.path.join(out, "corpus.txt"), "w") as f:
        for r in train:
            f.write(r["record"] + EOT + "\n")
    for split, data in (("train", train), ("val", val)):
        with open(os.path.join(out, f"{split}.jsonl"), "w") as f:
            for r in data:
                f.write(json.dumps({k: r[k] for k in ("prompt", "code", "name", "source", "id")}) + "\n")

    stats = {"kept": kept, "dup": dup, "thin": thin, "no_entry": noentry,
             "by_source": dict(by_source), "train": len(train), "val": len(val),
             "chars": chars, "est_gpt2_tokens": round(chars / 3.7)}  # ~3.7 char/tok on code+meta
    json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=2)
    print(json.dumps(stats, indent=2))
    print(f"\n-> {out}/corpus.txt  (+ train.jsonl/val.jsonl/stats.json)")
    return stats


def main():
    ap = argparse.ArgumentParser(description="Unified shader corpus builder")
    ap.add_argument("--vipitis", help="Vipitis shadertoys-dataset repo dir")
    ap.add_argument("--mizu", help="mizuamedesu train jsonl")
    ap.add_argument("--mizu-eval", help="mizuamedesu eval jsonl (folded into corpus too)")
    ap.add_argument("--shaders21k", help="shaders21k codes/ dir (on box)")
    ap.add_argument("--extra", help="extra dir of // Shader: txt")
    ap.add_argument("--out", default="corpus_merged")
    ap.add_argument("--min-len", type=int, default=60)
    ap.add_argument("--val-frac", type=float, default=0.02)
    args = ap.parse_args()

    gens = []
    if args.vipitis:    gens.append(from_vipitis(args.vipitis))
    if args.mizu:       gens.append(from_mizu(args.mizu))
    if args.mizu_eval:  gens.append(from_mizu(args.mizu_eval))
    if args.shaders21k: gens.append(from_frag_dir(args.shaders21k, "shaders21k"))
    if args.extra:      gens.append(from_frag_dir(args.extra, "extra"))
    if not gens:
        sys.exit("ERROR: give at least one of --vipitis/--mizu/--shaders21k/--extra")
    build(gens, args.out, args.min_len, args.val_frac)


if __name__ == "__main__":
    main()
