#!/usr/bin/env python3
"""Read an OLMo-corecpp INI .conf and emit the COMPLETE JSON config that the
llm-cpp `chat` binary needs (conf_to_json.py drops hidden_size_multiplier /
block_type, which makes the FFN mismatch on load — this includes everything)."""
import sys, json, re
conf, out = sys.argv[1], sys.argv[2]
kv = {}
for line in open(conf):
    line = line.split("#")[0].strip()
    if not line or line.startswith("["): continue
    parts = re.split(r"[\s=]+", line, maxsplit=1)
    if len(parts) == 2: kv[parts[0]] = parts[1].strip()

def gi(k, d):  # int
    try: return int(kv.get(k, d))
    except: return d
def gf(k, d):
    try: return float(kv.get(k, d))
    except: return d
def gs(k, d): return kv.get(k, d)

cfg = {
  "d_model": gi("d_model", 768), "vocab_size": gi("vocab_size", 50304),
  "n_layers": gi("n_layers", 12), "n_heads": gi("n_heads", 12),
  "n_kv_heads": gi("n_kv_heads", -1), "head_dim": gi("head_dim", -1),
  "rope_theta": gf("rope_theta", 500000), "layer_norm_eps": gf("layer_norm_eps", 1e-5),
  "init_std": gf("init_std", 0.02),
  "use_qk_norm": gi("use_qk_norm", 0) == 1,
  "use_head_qk_norm": gi("use_head_qk_norm", 0) == 1,
  "block_type": gs("block_type", "reordered_norm"),
  "layer_norm_type": gs("layer_norm_type", "rms_norm"),
  "attention_backend": gs("attention_backend", "sdpa"),
  "hidden_size_multiple_of": gi("hidden_size_multiple_of", 256),
  "hidden_size_multiplier": gf("hidden_size_multiplier", 1.0),
  "num_mtp_heads": gi("num_mtp_heads", 0),
  "mtp_loss_weight": gf("mtp_loss_weight", 0.1),
  "use_multi_res": gi("multi_res", 0) == 1,
  "bpe_vocab_path": "data/gpt2/vocab.json",
}
json.dump(cfg, open(out, "w"), indent=2)
print("wrote", out, "->", json.dumps(cfg))
