#!/usr/bin/env python3
"""scripts/race/olmo_train_race.py

Python race side. Trains OLMo-core's llama2_271M architecture (dialed to
n_layers=12 / n_heads=8 → ~257M, matching race_250m_cpp.conf's backbone)
on data/race_tokens.npy and writes per-step metrics to a CSV the race
analyzer reads.

Built directly on olmo-python/src/examples/llm/train.py's verified API.
Strips the network-dependent callbacks (W&B, Comet, C4 + HellaSwag
evaluators) so it runs offline on the race box. Adds a SpeedMonitor for
throughput and a tiny CSV logger.

Launch (single GPU):

    torchrun --nproc-per-node=1 scripts/race/olmo_train_race.py \\
        --data data/race_tokens.npy \\
        --steps 1000 \\
        --metrics-csv scripts/race/results/python_train/metrics.csv

This is invoked for you by scripts/race/05_train_python.sh.
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Dict, List

from olmo_core.config import DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import Callback, ConfigSaverCallback
from olmo_core.train.callbacks.speed_monitor import SpeedMonitorCallback
from olmo_core.train.common import TRAIN_CE_LOSS_METRIC
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all


# Throughput metric keys recorded by SpeedMonitorCallback.
_TPS_KEYS = ("throughput/device/TPS (actual avg)", "throughput/device/TPS")


class RaceMetricsCallback(Callback):
    """Writes (step, loss, tok_per_s) to a CSV on every metrics log.

    Uses the same column names the C++ side's metrics.csv uses so
    08_analyze.py treats both identically.
    """

    def __init__(self, csv_path: str):
        super().__init__()
        self.csv_path = csv_path
        self._rows: List[Dict[str, Any]] = []
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=["step", "loss", "tok_per_s"]).writeheader()

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        loss = metrics.get(TRAIN_CE_LOSS_METRIC)
        tps = None
        for k in _TPS_KEYS:
            if k in metrics:
                tps = metrics[k]
                break
        if loss is None:
            return
        row = {"step": step, "loss": float(loss),
               "tok_per_s": float(tps) if tps is not None else float("nan")}
        self._rows.append(row)
        # Append incrementally so a crashed run still leaves partial data.
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=["step", "loss", "tok_per_s"]).writerow(row)
        if step % 10 == 0:
            tps_s = f"{row['tok_per_s']:.0f}" if row["tok_per_s"] == row["tok_per_s"] else "—"
            print(f"  step {step:>5} | loss {row['loss']:.4f} | tok/s {tps_s}", flush=True)


def build_config(args) -> Dict[str, Any]:
    tok = TokenizerConfig.gpt2()  # padded_vocab_size() → 50304

    # Model: llama2_271M architecture, dialed to match race_250m_cpp.conf.
    model_config = TransformerConfig.llama2_271M(
        vocab_size=tok.padded_vocab_size(),
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        rope_theta=10_000,
    )

    dataset_config = NumpyFSLDatasetConfig(
        paths=[args.data],
        sequence_length=args.seq_len,
        tokenizer=tok,
        work_dir=args.work_dir,
    )

    # global_batch_size and rank_microbatch_size are in TOKENS.
    # Match the C++ side: microbatch=4 instances × seq, global=32 instances × seq.
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=args.global_batch_instances * args.seq_len,
        seed=args.seed,
        num_workers=4,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=args.microbatch_instances * args.seq_len,
        max_sequence_length=args.seq_len,
        optim=AdamWConfig(
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0)),
            ],
        ),
        compile_model=args.compile,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
        max_grad_norm=args.max_grad_norm,
        scheduler=CosWithWarmup(warmup_steps=args.warmup_steps),
    )

    trainer_config = (
        TrainerConfig(
            save_folder=args.save_folder,
            save_overwrite=True,
            metrics_collect_interval=args.log_interval,
            cancel_check_interval=50,
            max_duration=Duration.steps(args.steps),
        )
        .with_callback("speed_monitor", SpeedMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("race_metrics", RaceMetricsCallback(args.metrics_csv))
    )

    return dict(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/race_tokens.npy")
    ap.add_argument("--metrics-csv", default="scripts/race/results/python_train/metrics.csv")
    ap.add_argument("--save-folder", default="scripts/race/results/python_ckpt")
    ap.add_argument("--work-dir", default="/tmp/race-dataset-cache")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--n-layers", type=int, default=12)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--microbatch-instances", type=int, default=4)
    ap.add_argument("--global-batch-instances", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3.0e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compile", action="store_true", default=True,
                    help="torch.compile the model (fair Python-side optimization)")
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    args = ap.parse_args()

    prepare_training_environment()
    try:
        seed_all(args.seed)
        cfg = build_config(args)
        model = cfg["model"].build(init_device="meta")
        train_module = cfg["train_module"].build(model)
        dataset = cfg["dataset"].build()
        data_loader = cfg["data_loader"].build(
            dataset, dp_process_group=train_module.dp_process_group)
        trainer = cfg["trainer"].build(train_module, data_loader)

        # Report the actual parameter count so the race log shows the match.
        n_params = cfg["model"].num_params
        print(f"[race] OLMo-core model: {n_params/1e6:.1f}M params "
              f"(d=1024, layers={args.n_layers}, heads={args.n_heads}, "
              f"vocab={cfg['model'].vocab_size}, untied)", flush=True)

        trainer.fit()
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
