#!/usr/bin/env python3
"""
Generation quality evaluation across frameworks.

Trains models, generates text from 10 prompts, grades responses automatically.
Outputs a JSON report + human-readable table.

Usage:
    python eval_generation.py \
        --llm-cpp-chat ./build/chat \
        --llm-cpp-checkpoint checkpoints/30M.pt \
        --llm-cpp-config configs/30M.json \
        --vocab-file data/gpt2/vocab.json \
        --merges-file data/gpt2/merges.txt \
        --cpp-llm-chat ../cpp-llm-clean/build/chat \
        --cpp-llm-checkpoint ../cpp-llm-clean/checkpoints/30M.pt \
        --cpp-llm-config ../cpp-llm-clean/configs/30M.json \
        --device cuda \
        --output results/generation_eval.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ── Story prompts (appropriate for TinyStories-trained models) ──
PROMPTS = [
    "Once upon a time there was a little cat who",
    "The girl walked into the forest and found a",
    "One day a boy named Tom went to the park and",
    "There was a big red ball in the garden that",
    "The little bird sang a beautiful song and then",
    "Lily and her friend went to the beach to find",
    "A tiny mouse lived in a small house near the",
    "The sun was shining bright when Sam decided to",
    "In a magical kingdom far away there lived a kind",
    "The dog wagged his tail because he was so happy to",
]

MAX_TOKENS = 200
TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.9
REP_PENALTY = 1.1


@dataclass
class GenerationResult:
    framework: str
    prompt: str
    response: str
    tokens_generated: int = 0
    tok_per_s: float = 0.0
    wall_time_s: float = 0.0
    score: float = 0.0
    score_breakdown: dict = field(default_factory=dict)


# ── Automated Grading ──

def detect_repetition(text: str) -> float:
    """Return fraction of text that is repeated n-grams (3-gram)."""
    words = text.lower().split()
    if len(words) < 6:
        return 0.0
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    counts = Counter(trigrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / max(len(trigrams), 1)


def vocabulary_diversity(text: str) -> float:
    """Unique words / total words. Higher = more diverse."""
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def sentence_completeness(text: str) -> float:
    """Fraction of sentences that end with proper punctuation."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences:
        return 0.0
    complete = sum(1 for s in sentences if s.strip() and s.strip()[-1] in '.!?')
    return complete / len(sentences)


def prompt_relevance(prompt: str, response: str) -> float:
    """Simple keyword overlap between prompt and response."""
    prompt_words = set(prompt.lower().split())
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at',
                  'to', 'for', 'of', 'is', 'was', 'were', 'that', 'who',
                  'there', 'then', 'so', 'his', 'her'}
    prompt_words -= stop_words
    response_words = set(response.lower().split())
    if not prompt_words:
        return 0.5
    overlap = len(prompt_words & response_words) / len(prompt_words)
    return min(overlap, 1.0)


def grade_response(prompt: str, response: str) -> tuple[float, dict]:
    """
    Grade a generation response on a 0-10 scale.

    Breakdown:
      Coherence     (0-3): penalized by repetition
      Fluency       (0-2): vocabulary diversity
      Relevance     (0-2): keyword overlap with prompt
      Completeness  (0-2): proper sentence endings
      Length        (0-1): generates enough meaningful text
    """
    words = response.split()
    n_words = len(words)

    # Coherence: 3 - (repetition_ratio * 6), clamp [0, 3]
    rep = detect_repetition(response)
    coherence = max(0.0, min(3.0, 3.0 - rep * 6.0))

    # Fluency: diversity * 2, clamp [0, 2]
    div = vocabulary_diversity(response)
    fluency = max(0.0, min(2.0, div * 2.5))

    # Relevance: overlap * 2, clamp [0, 2]
    rel = prompt_relevance(prompt, response)
    relevance = max(0.0, min(2.0, rel * 2.5))

    # Completeness: sentence_completeness * 2
    comp = sentence_completeness(response)
    completeness = max(0.0, min(2.0, comp * 2.0))

    # Length: 1.0 if >= 30 words, scaled down below that
    length_score = min(1.0, n_words / 30.0) if n_words > 0 else 0.0

    total = coherence + fluency + relevance + completeness + length_score
    breakdown = {
        "coherence": round(coherence, 2),
        "fluency": round(fluency, 2),
        "relevance": round(relevance, 2),
        "completeness": round(completeness, 2),
        "length": round(length_score, 2),
        "word_count": n_words,
        "repetition_ratio": round(rep, 3),
        "vocab_diversity": round(div, 3),
    }
    return round(total, 2), breakdown


# ── Framework Runners ──

def run_llm_cpp_chat(chat_binary: str, checkpoint: str, config_json: str,
                     vocab_file: str, merges_file: str, prompt: str,
                     device: str = "cuda") -> GenerationResult:
    """Run llm-cpp chat binary with a single prompt, parse output."""
    cmd = [
        chat_binary,
        "--checkpoint", checkpoint,
        "--config", config_json,
        "--vocab-file", vocab_file,
        "--merges-file", merges_file,
        "--device", device,
        "--max-tokens", str(MAX_TOKENS),
        "--temperature", str(TEMPERATURE),
        "--top-k", str(TOP_K),
        "--top-p", str(TOP_P),
        "--repetition-penalty", str(REP_PENALTY),
    ]

    start = time.time()
    try:
        proc = subprocess.run(
            cmd, input=prompt + "\nquit\n", capture_output=True,
            text=True, timeout=120
        )
        elapsed = time.time() - start
        output = proc.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return GenerationResult(
            framework="llm-cpp", prompt=prompt,
            response=f"[ERROR: {e}]", wall_time_s=time.time() - start
        )

    # Parse: "Model: <text>\n[N tokens, X.X tok/s]"
    response_text = ""
    tokens_gen = 0
    tok_s = 0.0

    model_match = re.search(r'Model:\s*(.*?)(?:\n\[|\Z)', output, re.DOTALL)
    if model_match:
        response_text = model_match.group(1).strip()

    stats_match = re.search(r'\[(\d+)\s+tokens?,\s+([\d.]+)\s+tok/s', output)
    if stats_match:
        tokens_gen = int(stats_match.group(1))
        tok_s = float(stats_match.group(2))

    score, breakdown = grade_response(prompt, response_text)

    return GenerationResult(
        framework="llm-cpp (Experimental)",
        prompt=prompt,
        response=response_text,
        tokens_generated=tokens_gen,
        tok_per_s=tok_s,
        wall_time_s=round(elapsed, 3),
        score=score,
        score_breakdown=breakdown,
    )


def run_cpp_llm_chat(chat_binary: str, checkpoint: str, config_json: str,
                     vocab_file: str, merges_file: str, prompt: str,
                     device: str = "cuda") -> GenerationResult:
    """Run cpp-llm-clean chat binary. Same interface as llm-cpp."""
    cmd = [
        chat_binary,
        "--checkpoint", checkpoint,
        "--config", config_json,
        "--vocab-file", vocab_file,
        "--merges-file", merges_file,
        "--device", device,
        "--max-tokens", str(MAX_TOKENS),
        "--temperature", str(TEMPERATURE),
    ]

    start = time.time()
    try:
        proc = subprocess.run(
            cmd, input=prompt + "\nquit\n", capture_output=True,
            text=True, timeout=120
        )
        elapsed = time.time() - start
        output = proc.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return GenerationResult(
            framework="cpp-llm", prompt=prompt,
            response=f"[ERROR: {e}]", wall_time_s=time.time() - start
        )

    response_text = ""
    tokens_gen = 0
    tok_s = 0.0

    model_match = re.search(r'Model:\s*(.*?)(?:\n\[|\Z)', output, re.DOTALL)
    if model_match:
        response_text = model_match.group(1).strip()

    stats_match = re.search(r'\[(\d+)\s+tokens?,\s+([\d.]+)\s+tok/s', output)
    if stats_match:
        tokens_gen = int(stats_match.group(1))
        tok_s = float(stats_match.group(2))

    score, breakdown = grade_response(prompt, response_text)

    return GenerationResult(
        framework="cpp-llm (clean)",
        prompt=prompt,
        response=response_text,
        tokens_generated=tokens_gen,
        tok_per_s=tok_s,
        wall_time_s=round(elapsed, 3),
        score=score,
        score_breakdown=breakdown,
    )


def run_python_generate(model_config: dict, checkpoint_dir: str,
                        vocab_file: str, merges_file: str, prompt: str,
                        device: str = "cuda") -> GenerationResult:
    """
    Generate text with a Python OLMo-core model.
    Runs as a subprocess to isolate the Python environment.
    """
    # Write inline generation script
    gen_script = f'''
import sys, json, time, torch, torch.nn.functional as F

# Minimal BPE decode
class SimpleBPE:
    def __init__(self, vocab_path, merges_path):
        with open(vocab_path) as f:
            self.vocab = json.load(f)
        self.id_to_token = {{v: k for k, v in self.vocab.items()}}
        self.eos_id = self.vocab.get("<|endoftext|>", 50256)
    def encode(self, text):
        # Rough char-level fallback for prompt encoding
        tokens = []
        for ch in text:
            if ch in self.vocab:
                tokens.append(self.vocab[ch])
            elif "\\u0120" + ch.lower() in self.vocab:
                tokens.append(self.vocab["\\u0120" + ch.lower()])
        return tokens if tokens else [0]
    def decode(self, ids):
        out = []
        for i in ids:
            tok = self.id_to_token.get(i, "")
            tok = tok.replace("\\u0120", " ").replace("\\u010a", "\\n")
            # Handle byte tokens
            if tok.startswith("\\u00") and len(tok) == 6:
                try:
                    tok = chr(int(tok[4:], 16))
                except ValueError:
                    pass
            out.append(tok)
        return "".join(out)

try:
    from olmo_core.nn.transformer import TransformerConfig, Transformer
    config = json.loads("""{json.dumps(model_config)}""")
    tc = TransformerConfig(
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        vocab_size=config["vocab_size"],
    )
    model = tc.build(device="{device}")
    # Load checkpoint if exists
    ckpt_path = "{checkpoint_dir}"
    if ckpt_path and __import__("os").path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="{device}", weights_only=False)
        if isinstance(state, dict):
            model.load_state_dict(state, strict=False)
    model.eval()
except Exception as e:
    # Fallback: create a simple transformer with PyTorch
    import torch.nn as nn
    class MiniTransformer(nn.Module):
        def __init__(self, vocab_size, d_model, n_layers, n_heads):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, batch_first=True)
            self.transformer = nn.TransformerEncoder(layer, n_layers)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        def forward(self, x):
            h = self.embed(x)
            mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
            h = self.transformer(h, mask=mask, is_causal=True)
            return self.lm_head(h)
    config = json.loads("""{json.dumps(model_config)}""")
    model = MiniTransformer(config["vocab_size"], config["d_model"],
                            config["n_layers"], config["n_heads"]).to("{device}")
    ckpt_path = "{checkpoint_dir}"
    if ckpt_path and __import__("os").path.exists(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location="{device}", weights_only=False)
            if isinstance(state, dict):
                model.load_state_dict(state, strict=False)
        except Exception:
            pass
    model.eval()

# Tokenize prompt
bpe = SimpleBPE("{vocab_file}", "{merges_file}")
prompt_text = """{prompt}"""
prompt_ids = bpe.encode(prompt_text)
input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="{device}")

# Generate
torch.manual_seed(42)
start = time.time()
tokens_gen = 0
with torch.no_grad():
    for _ in range({MAX_TOKENS}):
        logits = model(input_ids)[:, -1, :]
        logits = logits / {TEMPERATURE}
        topk = torch.topk(logits, {TOP_K})
        probs = F.softmax(topk.values, dim=-1)
        idx = torch.multinomial(probs, 1)
        next_token = topk.indices.gather(-1, idx)
        if next_token.item() == bpe.eos_id:
            break
        input_ids = torch.cat([input_ids, next_token], dim=1)
        tokens_gen += 1

elapsed = time.time() - start
generated_ids = input_ids[0, len(prompt_ids):].tolist()
text = bpe.decode(generated_ids)
tok_s = tokens_gen / elapsed if elapsed > 0 else 0
print(f"PYOUT|||{{text}}|||{{tokens_gen}}|||{{tok_s:.1f}}|||{{elapsed:.3f}}")
'''

    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", gen_script],
            capture_output=True, text=True, timeout=180,
        )
        elapsed = time.time() - start
        output = proc.stdout.strip()
    except (subprocess.TimeoutExpired, Exception) as e:
        return GenerationResult(
            framework="Python OLMo-core", prompt=prompt,
            response=f"[ERROR: {e}]", wall_time_s=time.time() - start
        )

    # Parse PYOUT|||text|||tokens|||tok_s|||elapsed
    response_text = ""
    tokens_gen = 0
    tok_s = 0.0

    for line in output.split("\n"):
        if line.startswith("PYOUT|||"):
            parts = line.split("|||")
            if len(parts) >= 5:
                response_text = parts[1]
                tokens_gen = int(parts[2])
                tok_s = float(parts[3])
                elapsed = float(parts[4])

    if not response_text and proc.stderr:
        response_text = f"[STDERR: {proc.stderr[:200]}]"

    score, breakdown = grade_response(prompt, response_text)

    return GenerationResult(
        framework="Python OLMo-core",
        prompt=prompt,
        response=response_text,
        tokens_generated=tokens_gen,
        tok_per_s=tok_s,
        wall_time_s=round(elapsed, 3),
        score=score,
        score_breakdown=breakdown,
    )


# ── Report Generation ──

def print_table(results_by_framework: dict[str, list[GenerationResult]]):
    """Print a comparison table."""
    frameworks = list(results_by_framework.keys())

    print("\n" + "=" * 80)
    print("GENERATION QUALITY EVALUATION")
    print("=" * 80)

    # Per-prompt comparison
    for i, prompt in enumerate(PROMPTS):
        print(f"\n--- Prompt {i+1}: \"{prompt[:50]}...\"")
        for fw in frameworks:
            results = results_by_framework[fw]
            if i < len(results):
                r = results[i]
                print(f"  [{fw}] Score: {r.score}/10 | "
                      f"{r.tokens_generated} tok @ {r.tok_per_s:.1f} tok/s")
                response_preview = r.response[:120].replace('\n', ' ')
                print(f"    \"{response_preview}...\"")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Framework':<25} {'Avg Score':>10} {'Avg tok/s':>12} {'Avg Time':>10}")
    print("-" * 60)

    for fw in frameworks:
        results = results_by_framework[fw]
        valid = [r for r in results if not r.response.startswith("[ERROR")]
        if not valid:
            print(f"{fw:<25} {'N/A':>10} {'N/A':>12} {'N/A':>10}")
            continue
        avg_score = sum(r.score for r in valid) / len(valid)
        avg_tok_s = sum(r.tok_per_s for r in valid) / len(valid)
        avg_time = sum(r.wall_time_s for r in valid) / len(valid)
        print(f"{fw:<25} {avg_score:>10.2f} {avg_tok_s:>12.1f} {avg_time:>10.3f}s")

    # Score breakdown averages
    print(f"\n{'Framework':<25} {'Coher':>7} {'Fluency':>8} {'Relev':>7} "
          f"{'Compl':>7} {'Length':>7}")
    print("-" * 65)
    for fw in frameworks:
        results = results_by_framework[fw]
        valid = [r for r in results if r.score_breakdown]
        if not valid:
            continue
        n = len(valid)
        c = sum(r.score_breakdown.get("coherence", 0) for r in valid) / n
        f = sum(r.score_breakdown.get("fluency", 0) for r in valid) / n
        rl = sum(r.score_breakdown.get("relevance", 0) for r in valid) / n
        cp = sum(r.score_breakdown.get("completeness", 0) for r in valid) / n
        ln = sum(r.score_breakdown.get("length", 0) for r in valid) / n
        print(f"{fw:<25} {c:>7.2f} {f:>8.2f} {rl:>7.2f} {cp:>7.2f} {ln:>7.2f}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Generation quality evaluation")
    # llm-cpp (Experimental)
    parser.add_argument("--llm-cpp-chat", help="Path to llm-cpp chat binary")
    parser.add_argument("--llm-cpp-checkpoint", help="llm-cpp model checkpoint")
    parser.add_argument("--llm-cpp-config", help="llm-cpp model config JSON")
    # cpp-llm (clean)
    parser.add_argument("--cpp-llm-chat", help="Path to cpp-llm chat binary")
    parser.add_argument("--cpp-llm-checkpoint", help="cpp-llm model checkpoint")
    parser.add_argument("--cpp-llm-config", help="cpp-llm model config JSON")
    # Python
    parser.add_argument("--python-config", help="JSON string or file with model config")
    parser.add_argument("--python-checkpoint", help="Python model checkpoint")
    # Shared
    parser.add_argument("--vocab-file", required=True, help="GPT-2 vocab.json")
    parser.add_argument("--merges-file", required=True, help="GPT-2 merges.txt")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output", default="results/generation_eval.json",
                        help="Output JSON file")
    args = parser.parse_args()

    results_by_framework = {}

    # ── llm-cpp (Experimental) ──
    if args.llm_cpp_chat and args.llm_cpp_checkpoint and args.llm_cpp_config:
        print(f"\n>>> Evaluating llm-cpp (Experimental) generation...")
        results = []
        for i, prompt in enumerate(PROMPTS):
            print(f"  Prompt {i+1}/10...", end=" ", flush=True)
            r = run_llm_cpp_chat(
                args.llm_cpp_chat, args.llm_cpp_checkpoint,
                args.llm_cpp_config, args.vocab_file, args.merges_file,
                prompt, args.device
            )
            results.append(r)
            print(f"Score: {r.score}/10, {r.tok_per_s:.1f} tok/s")
        results_by_framework["llm-cpp (Experimental)"] = results

    # ── cpp-llm (clean) ──
    if args.cpp_llm_chat and args.cpp_llm_checkpoint and args.cpp_llm_config:
        print(f"\n>>> Evaluating cpp-llm (clean) generation...")
        results = []
        for i, prompt in enumerate(PROMPTS):
            print(f"  Prompt {i+1}/10...", end=" ", flush=True)
            r = run_cpp_llm_chat(
                args.cpp_llm_chat, args.cpp_llm_checkpoint,
                args.cpp_llm_config, args.vocab_file, args.merges_file,
                prompt, args.device
            )
            results.append(r)
            print(f"Score: {r.score}/10, {r.tok_per_s:.1f} tok/s")
        results_by_framework["cpp-llm (clean)"] = results

    # ── Python OLMo-core ──
    if args.python_config:
        print(f"\n>>> Evaluating Python OLMo-core generation...")
        if os.path.isfile(args.python_config):
            with open(args.python_config) as f:
                model_config = json.load(f)
        else:
            model_config = json.loads(args.python_config)

        results = []
        for i, prompt in enumerate(PROMPTS):
            print(f"  Prompt {i+1}/10...", end=" ", flush=True)
            r = run_python_generate(
                model_config, args.python_checkpoint or "",
                args.vocab_file, args.merges_file,
                prompt, args.device
            )
            results.append(r)
            print(f"Score: {r.score}/10, {r.tok_per_s:.1f} tok/s")
        results_by_framework["Python OLMo-core"] = results

    if not results_by_framework:
        print("No frameworks configured. Use --llm-cpp-chat, --cpp-llm-chat, "
              "or --python-config.")
        sys.exit(1)

    # Print comparison table
    print_table(results_by_framework)

    # Save JSON report
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    report = {
        "prompts": PROMPTS,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "frameworks": {},
    }
    for fw, results in results_by_framework.items():
        valid = [r for r in results if not r.response.startswith("[ERROR")]
        report["frameworks"][fw] = {
            "results": [asdict(r) for r in results],
            "avg_score": round(sum(r.score for r in valid) / max(len(valid), 1), 2),
            "avg_tok_per_s": round(sum(r.tok_per_s for r in valid) / max(len(valid), 1), 1),
            "num_prompts": len(results),
            "num_errors": len(results) - len(valid),
        }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {args.output}")


if __name__ == "__main__":
    main()
