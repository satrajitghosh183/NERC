#!/usr/bin/env python3
"""
scripts/conf_to_json.py

Convert one of this repo's INI-style training configs (`conf/*.conf`) into
a flat JSON config consumed by the standalone `chat` C++ binary.

Why this exists:
  Training (`olmo_train`) reads the full multi-section INI directly.
  Inference / chat (`build/chat`) only needs the *model* hyperparameters
  plus a handful of MTP / multi-res fields, and it expects them as JSON
  for simplicity. This script is the bridge between the two formats.

Usage:
    python3 scripts/conf_to_json.py <input.conf> [output.json]
    # If output.json is omitted, JSON is written to stdout.

Examples:
    python3 scripts/conf_to_json.py conf/olmo.conf configs/olmo.json
    python3 scripts/conf_to_json.py conf/olmo_125M.conf > /tmp/m.json

--- Reads ---
    sys.argv[1]  — path to a .conf file (INI-style, sections in [brackets],
                   key/value pairs separated by tab or '=', '#' starts a
                   trailing comment)

--- Writes ---
    sys.argv[2]  — path to JSON output. If absent, writes to stdout
                   (/dev/stdout) — note this is POSIX-only.

--- Role in workflow ---
    Invoked once per .conf by the kzin_run.sh / jetstream_run.sh
    `generate_json_configs` phase, just before chat-quality evaluation.
"""
import sys, json


def parse_conf(path):
    """Parse an INI-ish `.conf` file into a dict-of-dicts: {section: {k: v}}.

    Recognized syntax:
      - `# ...` lines and blank lines are ignored.
      - `[section_name]` starts a new section.
      - `key<TAB>value` or `key = value` are key/value pairs in the
        current section. The first occurrence of TAB or '=' splits the
        line; values are stripped of trailing `#…` comments.
    """
    sections = {}
    current = ""  # name of the section we are currently filling
    with open(path) as f:
        for line in f:
            line = line.strip()
            # Skip blanks and full-line comments.
            if not line or line.startswith('#'):
                continue
            # New section header.
            if line.startswith('[') and line.endswith(']'):
                current = line[1:-1]
                sections[current] = {}
                continue
            # Key/value: try TAB first (the .conf style we usually write),
            # then '=' as a fallback for hand-edited files.
            for sep in ['\t', '=']:
                if sep in line:
                    key, val = line.split(sep, 1)
                    key = key.strip()
                    # Drop inline `# comment` and surrounding whitespace.
                    val = val.split('#')[0].strip()
                    sections[current][key] = val
                    break
    return sections


# Parse the input config. argv[1] is the .conf path (positional, required).
conf = parse_conf(sys.argv[1])

# Pull the three sections we care about. `.get(..., {})` keeps the script
# robust if a section is missing; per-key defaults below cover the rest.
m = conf.get('model', {})         # architecture (d_model, n_layers, …)
o = conf.get('optimization', {})  # multi_res toggle, etc.
d = conf.get('data', {})          # bpe_vocab path for multi-res tokenizer

# Build the JSON dict. Defaults below mirror conf/olmo.conf (the 30M preset)
# so the chat binary won't crash if a key is missing from the source .conf.
config = {
    # ── Core transformer geometry ────────────────────────────────────
    "d_model": int(m.get('d_model', 256)),
    "vocab_size": int(m.get('vocab_size', 50257)),
    "n_layers": int(m.get('n_layers', 4)),
    "n_heads": int(m.get('n_heads', 8)),
    # n_kv_heads = -1 means "same as n_heads" (no GQA).
    "n_kv_heads": int(m.get('n_kv_heads', -1)),
    # head_dim = -1 means "derive as d_model / n_heads".
    "head_dim": int(m.get('head_dim', -1)),
    # RoPE base (θ); 500000 is the OLMo2 / Llama-3 value, 10000 is GPT-NeoX.
    "rope_theta": int(m.get('rope_theta', 500000)),
    "layer_norm_eps": float(m.get('layer_norm_eps', 1e-6)),
    "init_std": float(m.get('init_std', 0.02)),
    # use_qk_norm: applies RMSNorm to Q and K before the attention dot product
    # (an OLMo2 stability trick). Stored as 0/1 in conf, exposed as bool here.
    "use_qk_norm": bool(int(m.get('use_qk_norm', 1))),
    # ── Multi-token prediction (MTP) heads ───────────────────────────
    "num_mtp_heads": int(m.get('num_mtp_heads', 0)),
    "mtp_loss_weight": float(m.get('mtp_loss_weight', 0.1)),
    # ── DC-MRE (multi-resolution embeddings) ─────────────────────────
    "use_multi_res": bool(int(o.get('multi_res', 0))),
    # Path to GPT-2 vocab.json — the multi-res embedder needs it to extract
    # character trigrams from BPE tokens at chat time.
    "bpe_vocab_path": d.get('bpe_vocab', ''),
    # Hash-bucket sizes for the char-trigram and phrase-pair side embeddings.
    # These are intentionally hard-coded (not in the .conf) to keep the chat
    # binary's input format minimal and stable across runs.
    "multi_res_char_buckets": 4096,
    "multi_res_phrase_buckets": 8192,
    "multi_res_inner_dim": 64,
}

# argv[2] is the JSON output path; default to stdout via /dev/stdout.
out = sys.argv[2] if len(sys.argv) > 2 else "/dev/stdout"
with open(out, 'w') as f:
    # indent=2 keeps the file diff-friendly when checked into git.
    json.dump(config, f, indent=2)
    f.write('\n')  # trailing newline (POSIX text-file convention)
