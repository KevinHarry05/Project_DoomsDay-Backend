"""
PatchTST: patch-based transformer encoder, direct multi-horizon forecasting.

Architecture (scaled down from the team's notebook - see config.yaml for the
full rationale on every size choice):

    [lookback, n_features] input window
        -> split into overlapping patches (patch_length, patch_stride)
        -> linear patch embedding to d_model
        -> standard Transformer encoder (n_layers, n_heads, dim_feedforward)
        -> flatten -> linear projection head -> [n_horizons] direct output

This is ONE shared/global model trained pooled across all 12 regions (matching
the notebook's own design), not 12 independent per-region transformers - far
cheaper to train and better able to learn cross-region demand structure.

Everything downstream of training (contract shaping, DB writing) treats this
the same as every other track: one ModelRunContract + forecast rows per
region, even though the weights are shared. That sharing is recorded
explicitly in model_runs.metadata.pooled_global_model so nobody mistakes it
for 12 independently-fit models later.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError("torch is required for the PatchTST track") from exc

from .config import Config
from .contracts import ModelRunContract
from .evaluation import evaluate_split
from .features import Origin, RegionSequenceSet, build_region_sequences, window_and_targets
from .utils import Timer, utcnow

logger = logging.getLogger(__name__)


# -- model --------------------------------------------------------------------

class PatchTST(nn.Module):
    def __init__(self, n_features: int, lookback: int, n_horizons: int, cfg: Config):
        super().__init__()
        self.patch_len = int(cfg.get("model.patch_length", 16))
        self.stride = int(cfg.get("model.patch_stride", 8))
        d_model = int(cfg.get("model.d_model", 32))
        n_heads = int(cfg.get("model.n_heads", 2))
        n_layers = int(cfg.get("model.n_layers", 2))
        dim_ff = int(cfg.get("model.dim_feedforward", 64))
        dropout = float(cfg.get("model.dropout", 0.10))

        self.num_patches = (lookback - self.patch_len) // self.stride + 1
        if self.num_patches < 1:
            raise ValueError("lookback too short for patch_length/patch_stride")

        self.patch_embed = nn.Linear(self.patch_len * n_features, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(self.num_patches * d_model, n_horizons)

    def _to_patches(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: [batch, lookback, n_features] -> [batch, num_patches, patch_len * n_features]
        patches = x.unfold(1, self.patch_len, self.stride)  # [batch, num_patches, n_features, patch_len]
        patches = patches.permute(0, 1, 3, 2).contiguous()  # [batch, num_patches, patch_len, n_features]
        b, p, pl, f = patches.shape
        return patches.view(b, p, pl * f)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        patches = self._to_patches(x)
        emb = self.patch_embed(patches) + self.pos_embed
        enc = self.encoder(emb)
        enc = self.norm(enc)
        flat = enc.reshape(enc.shape[0], -1)
        return self.head(flat)  # [batch, n_horizons]


class _SeqDataset(Dataset):
    def __init__(self, samples: List[np.ndarray], labels: List[np.ndarray]):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i], self.labels[i]


# -- training result containers (mirror statistical_forecasting's shape) ------

@dataclass
class HorizonSeries:
    predictions: pd.DataFrame
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class RegionResult:
    region_code: str
    run: ModelRunContract
    horizons: Dict[int, HorizonSeries]
    status: str = "SUCCESS"
    failure_reason: Optional[str] = None
    fit_seconds: float = 0.0
    n_origins: int = 0


@dataclass
class GlobalTrainResult:
    model: "PatchTST"
    sequence_sets: Dict[str, RegionSequenceSet]
    horizons: List[int]
    fit_seconds: float
    n_train_samples: int
    n_val_samples: int
    training_start: pd.Timestamp
    training_end: pd.Timestamp
    n_features: int
    n_params: int
    epochs_run: int
    best_val_loss: float
    stopped_reason: str


def _collect_samples(seq_sets: Dict[str, RegionSequenceSet], cfg: Config, horizons: List[int], split: str):
    X, Y = [], []
    for seq in seq_sets.values():
        for origin in seq.origins:
            if origin.split_name != split:
                continue
            window, targets = window_and_targets(seq, origin, cfg, horizons)
            y = np.array(
                [seq.scaler.transform(np.array([targets[h][1]]))[0] if np.isfinite(targets[h][1]) else 0.0
                 for h in horizons],
                dtype=np.float32,
            )
            X.append(window)
            Y.append(y)
    return X, Y


def train_global(regions_data: Dict[str, pd.DataFrame], cfg: Config, horizons: List[int]) -> GlobalTrainResult:
    torch.manual_seed(int(cfg.get("training.seed", 42)))

    seq_sets: Dict[str, RegionSequenceSet] = {}
    for region, df in regions_data.items():
        built = build_region_sequences(df, cfg, horizons)
        if built is not None:
            seq_sets[region] = built
    if not seq_sets:
        raise RuntimeError("No region produced usable sequences")

    n_features = len(cfg.get("sequence.feature_columns"))
    lookback = int(cfg.get("sequence.lookback_hours", 168))

    X_train, Y_train = _collect_samples(seq_sets, cfg, horizons, "train")
    X_val, Y_val = _collect_samples(seq_sets, cfg, horizons, "val")
    logger.info("Pooled training samples: train=%d val=%d (across %d regions)",
                len(X_train), len(X_val), len(seq_sets))

    model = PatchTST(n_features=n_features, lookback=lookback, n_horizons=len(horizons), cfg=cfg)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("PatchTST params: %s", f"{n_params:,}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("training.learning_rate", 1e-3)),
        weight_decay=float(cfg.get("training.weight_decay", 1e-4)),
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=1)
    loss_fn = nn.MSELoss()

    batch_size = int(cfg.get("training.batch_size", 64))
    max_epochs = int(cfg.get("training.max_epochs", 12))
    patience = int(cfg.get("training.patience", 3))
    grad_clip = float(cfg.get("training.grad_clip", 1.0))
    max_minutes = float(cfg.get("training.max_train_minutes", 25))

    train_ds = _SeqDataset(X_train, Y_train)
    val_ds = _SeqDataset(X_val, Y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    best_val = float("inf")
    best_state = None
    epochs_since_improve = 0
    stopped_reason = "max_epochs_reached"
    epochs_run = 0

    training_start = utcnow()
    wall_start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            running += float(loss.item())
            n_batches += 1
            if (time.perf_counter() - wall_start) / 60.0 > max_minutes:
                stopped_reason = "max_train_minutes_reached"
                break
        train_loss = running / max(n_batches, 1)

        model.eval()
        val_running, val_batches = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_running += float(loss_fn(pred, yb).item())
                val_batches += 1
        val_loss = val_running / max(val_batches, 1)
        sched.step(val_loss)
        epochs_run = epoch

        elapsed_min = (time.perf_counter() - wall_start) / 60.0
        logger.info("Epoch %2d/%d  train_loss=%.5f  val_loss=%.5f  elapsed=%.1fmin",
                    epoch, max_epochs, train_loss, val_loss, elapsed_min)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                stopped_reason = "early_stopping"
                break

        if stopped_reason == "max_train_minutes_reached":
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    fit_seconds = time.perf_counter() - wall_start
    training_end = utcnow()

    logger.info("Training stopped: %s (epochs=%d, best_val_loss=%.5f, %.1fs)",
                stopped_reason, epochs_run, best_val, fit_seconds)

    return GlobalTrainResult(
        model=model, sequence_sets=seq_sets, horizons=horizons, fit_seconds=fit_seconds,
        n_train_samples=len(X_train), n_val_samples=len(X_val),
        training_start=training_start, training_end=training_end,
        n_features=n_features, n_params=n_params, epochs_run=epochs_run,
        best_val_loss=best_val, stopped_reason=stopped_reason,
    )


def evaluate_region(global_result: GlobalTrainResult, region_code: str, cfg: Config) -> RegionResult:
    seq = global_result.sequence_sets.get(region_code)
    if seq is None:
        return RegionResult(
            region_code=region_code,
            run=_dummy_run(region_code, cfg, global_result),
            horizons={}, status="FAILED",
            failure_reason="No usable sequences (insufficient rows for lookback/horizon)",
        )

    model = global_result.model
    model.eval()
    horizons = global_result.horizons
    smape_eps = float(cfg.get("evaluation.smape_epsilon", 1.0))

    rows_by_split: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    with torch.no_grad():
        for origin in seq.origins:
            window, targets = window_and_targets(seq, origin, cfg, horizons)
            x = torch.from_numpy(window).unsqueeze(0)
            pred_scaled = model(x).squeeze(0).numpy()
            pred_mw = seq.scaler.inverse(pred_scaled)
            for i, h in enumerate(horizons):
                target_ts, actual = targets[h]
                rows_by_split[origin.split_name].append({
                    "forecast_timestamp": origin.forecast_timestamp,
                    "target_timestamp": target_ts,
                    "horizon_hours": h,
                    "predicted_demand_mw": float(pred_mw[i]),
                    "actual_demand_mw": float(actual) if np.isfinite(actual) else None,
                    "split_name": origin.split_name,
                })

    run = _build_run_contract(region_code, cfg, global_result, seq)
    horizon_map: Dict[int, HorizonSeries] = {}
    all_rows = rows_by_split["train"] + rows_by_split["val"] + rows_by_split["test"]
    full_df = pd.DataFrame(all_rows)

    for h in horizons:
        h_df = full_df[full_df["horizon_hours"] == h].copy()
        metrics: Dict[str, Dict[str, float]] = {}
        for split_name in ("train", "val", "test"):
            part = h_df[(h_df["split_name"] == split_name) & h_df["actual_demand_mw"].notna()]
            if part.empty:
                continue
            eval_frame = part.rename(columns={"actual_demand_mw": "y"})
            metrics[split_name] = evaluate_split(
                eval_frame, pred_col="predicted_demand_mw", target_col="y",
                smape_epsilon=smape_eps, horizon_hours=None,
            )
        horizon_map[h] = HorizonSeries(predictions=h_df, metrics=metrics)

    return RegionResult(
        region_code=region_code, run=run, horizons=horizon_map, status="SUCCESS",
        fit_seconds=global_result.fit_seconds, n_origins=len(seq.origins),
    )


def _build_run_contract(region_code: str, cfg: Config, gr: GlobalTrainResult, seq: RegionSequenceSet) -> ModelRunContract:
    return ModelRunContract(
        model_name=cfg["project.model_name"],
        model_type=cfg["project.model_type"],
        model_version=cfg["project.model_version"],
        region_code=region_code,
        training_start=gr.training_start,
        training_end=gr.training_end,
        horizons=gr.horizons,
        feature_version="patchtst-seq-1.0.0",
        code_version=cfg["project.code_version"],
        status="SUCCESS",
        n_features=gr.n_features,
        n_training_rows=gr.n_train_samples,
        metadata={
            "pooled_global_model": True,
            "n_regions_pooled": len(gr.sequence_sets),
            "n_params": gr.n_params,
            "epochs_run": gr.epochs_run,
            "stopped_reason": gr.stopped_reason,
            "best_val_loss": gr.best_val_loss,
            "lookback_hours": int(cfg.get("sequence.lookback_hours", 168)),
            "origin_cadence_hours": int(cfg.get("sequence.origin_cadence_hours", 24)),
            "scaler_mean": seq.scaler.mean,
            "scaler_std": seq.scaler.std,
        },
    )


def _dummy_run(region_code: str, cfg: Config, gr: GlobalTrainResult) -> ModelRunContract:
    return ModelRunContract(
        model_name=cfg["project.model_name"], model_type=cfg["project.model_type"],
        model_version=cfg["project.model_version"], region_code=region_code,
        training_start=gr.training_start, training_end=gr.training_end,
        horizons=gr.horizons, feature_version="patchtst-seq-1.0.0",
        code_version=cfg["project.code_version"], status="FAILED",
        failure_reason="No usable sequences", metadata={"pooled_global_model": True},
    )
