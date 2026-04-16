"""
train_module.py — Inverse-PINN Odometry Correction
Training loop, experiment tracking (lightweight JSON), reproducibility.

Exports:
    @dataclass TrainConfig
    set_seed(seed: int) -> None
    build_optimizer(model, cfg) -> torch.optim.Optimizer
    build_scheduler(optimizer, cfg) -> torch.optim.lr_scheduler._LRScheduler
    compute_data_weights(df_train, target_cols) -> (w_x, w_yaw)
    JSONLogger         : in-process history logger -> history.json
    EarlyStopping      : patience-based stopper on val L_data
    train_one_epoch(...)
    evaluate(...)
    train(model, loss_fn, loaders, cfg, out_dir) -> dict
        Writes: out_dir/history.json, out_dir/best.pt, out_dir/config.json

Works on GPU (CUDA) with AMP (fp16) or CPU (AMP auto-disabled).
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    # Optimization
    epochs: int = 300
    mlp_lr: float = 1e-3
    coeff_lr_mult: float = 0.3        # coeff_lr = mlp_lr * coeff_lr_mult
    weight_decay: float = 1e-4        # applied to MLP only; 0 for coeffs
    grad_clip_max_norm: float = 1.0
    # Scheduler (CosineAnnealingLR over epochs)
    scheduler: str = "cosine"          # 'cosine' | 'none'
    cosine_eta_min_ratio: float = 0.01 # eta_min = mlp_lr * ratio
    # Early stopping (on val L_data)
    early_stop_patience: int = 20
    early_stop_min_delta: float = 0.0
    # Runtime
    seed: int = 42
    device: str = "auto"               # 'auto' | 'cuda' | 'cpu'
    amp: bool = True                   # mixed precision if CUDA available
    deterministic: bool = True         # torch cudnn deterministic
    log_every: int = 1                 # print every N epochs
    # Checkpoint
    out_dir: str = "./runs/pinn"
    save_best: bool = True
    # Labels
    run_name: str = "pinn_default"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Cublas determinism for matmul (PyTorch >=1.8); safe to set, no-op if unsupported
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def resolve_device(pref: str = "auto") -> torch.device:
    if pref == "cuda":
        return torch.device("cuda")
    if pref == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Optimizer / Scheduler
# ---------------------------------------------------------------------------
def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW with separate param groups for MLP and physical coefficients."""
    # Detect InversePINN vs baselines
    has_coeffs = hasattr(model, "coeff_parameters") and hasattr(model, "mlp_parameters")
    if has_coeffs:
        mlp_params = list(model.mlp_parameters())
        coeff_params = list(model.coeff_parameters())
        groups = [
            {"params": mlp_params, "lr": cfg.mlp_lr, "weight_decay": cfg.weight_decay,
             "name": "mlp"},
            {"params": coeff_params, "lr": cfg.mlp_lr * cfg.coeff_lr_mult,
             "weight_decay": 0.0, "name": "coeff"},
        ]
    else:
        groups = [
            {"params": [p for p in model.parameters() if p.requires_grad],
             "lr": cfg.mlp_lr, "weight_decay": cfg.weight_decay, "name": "all"},
        ]
    return torch.optim.AdamW(groups, lr=cfg.mlp_lr, weight_decay=cfg.weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(cfg.epochs, 1),
            eta_min=cfg.mlp_lr * cfg.cosine_eta_min_ratio,
        )
    return None


# ---------------------------------------------------------------------------
# Data loss weights from GT std
# ---------------------------------------------------------------------------
def compute_data_weights(
    df_train,
    target_cols=("gt_d_x", "gt_d_yaw"),
    eps: float = 1e-6,
) -> Tuple[float, float]:
    """Return per-axis inverse-variance weights from the training split."""
    arr = np.stack([df_train[c].to_numpy(dtype=np.float64) for c in target_cols], axis=-1)
    var = arr.var(axis=0) + eps
    w = 1.0 / var
    return float(w[0]), float(w[1])


# ---------------------------------------------------------------------------
# JSON logger + Early stopping
# ---------------------------------------------------------------------------
class JSONLogger:
    """Accumulate per-epoch rows and dump to JSON on flush()."""

    def __init__(self, path: str, run_name: str, config: Optional[dict] = None):
        self.path = path
        self.run_name = run_name
        self.config = config or {}
        self.rows: List[Dict] = []
        self.coeff_history: List[Dict] = []
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def log_epoch(self, row: Dict) -> None:
        self.rows.append(row)

    def log_coefficients(self, epoch: int, coeffs: Dict[str, float]) -> None:
        self.coeff_history.append({"epoch": int(epoch), **coeffs})

    def flush(self) -> None:
        payload = {
            "run_name": self.run_name,
            "config": self.config,
            "epochs": self.rows,
            "coefficients": self.coeff_history,
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=float)


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = math.inf if mode == "min" else -math.inf
        self.bad = 0
        self.stopped = False

    def step(self, value: float) -> bool:
        """Return True if improved."""
        improved = (
            (self.mode == "min" and value < self.best - self.min_delta)
            or (self.mode == "max" and value > self.best + self.min_delta)
        )
        if improved:
            self.best = value
            self.bad = 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                self.stopped = True
        return improved


# ---------------------------------------------------------------------------
# Forward helpers (supports InversePINN and baselines uniformly)
# ---------------------------------------------------------------------------
def _model_forward(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Call model(x) or model(x, x_raw=...) as appropriate."""
    name = type(model).__name__
    if name in ("IdentityBaseline", "LinearBaseline"):
        return model(batch["x"], x_raw=batch["x_raw"])
    return model(batch["x"])


def _model_coeffs(model: nn.Module) -> Optional[Dict[str, torch.Tensor]]:
    if hasattr(model, "coefficients"):
        return model.coefficients()
    return None


# ---------------------------------------------------------------------------
# Train / Eval loops
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_clip: float = 1.0,
    amp: bool = False,
) -> Dict[str, float]:
    model.train()
    running = {"total": 0.0, "data": 0.0, "stationary": 0.0,
               "nonholonomic": 0.0, "coeff": 0.0, "magnitude": 0.0}
    n_batches = 0
    t0 = time.time()

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)

        if amp and scaler is not None:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                pred = _model_forward(model, batch)
                coeffs = _model_coeffs(model)
                total, parts = loss_fn(
                    pred, batch["y"], batch["x_raw"],
                    stationary=batch.get("stationary"), coeffs=coeffs,
                )
            scaler.scale(total).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip,
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = _model_forward(model, batch)
            coeffs = _model_coeffs(model)
            total, parts = loss_fn(
                pred, batch["y"], batch["x_raw"],
                stationary=batch.get("stationary"), coeffs=coeffs,
            )
            total.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip,
                )
            optimizer.step()

        for k in running:
            running[k] += float(parts[k].detach().cpu())
        n_batches += 1

    elapsed = time.time() - t0
    out = {k: v / max(n_batches, 1) for k, v in running.items()}
    out["elapsed_s"] = elapsed
    return out


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loss_fn: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    running = {"total": 0.0, "data": 0.0, "stationary": 0.0,
               "nonholonomic": 0.0, "coeff": 0.0, "magnitude": 0.0}
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        pred = _model_forward(model, batch)
        coeffs = _model_coeffs(model)
        total, parts = loss_fn(
            pred, batch["y"], batch["x_raw"],
            stationary=batch.get("stationary"), coeffs=coeffs,
        )
        for k in running:
            running[k] += float(parts[k].detach().cpu())
        n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in running.items()}


# ---------------------------------------------------------------------------
# Main train()
# ---------------------------------------------------------------------------
def train(
    model: nn.Module,
    loss_fn: nn.Module,
    loaders: Dict[str, DataLoader],
    cfg: Optional[TrainConfig] = None,
    extra_config: Optional[dict] = None,
) -> Dict:
    """Full training with early stopping, AMP, best checkpoint, JSON history.

    Returns a dict with keys: history_path, best_path, best_val_data,
                              best_epoch, stopped_early, coefficients.
    """
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed, cfg.deterministic)
    device = resolve_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    model = model.to(device)
    loss_fn = loss_fn.to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    use_amp = bool(cfg.amp and device.type == "cuda")
    amp_scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    # Logger
    config_dump = {"train": asdict(cfg), **(extra_config or {})}
    history_path = os.path.join(cfg.out_dir, "history.json")
    config_path = os.path.join(cfg.out_dir, "config.json")
    best_path = os.path.join(cfg.out_dir, "best.pt")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dump, f, indent=2, default=str)

    logger = JSONLogger(history_path, cfg.run_name, config=config_dump)
    stopper = EarlyStopping(cfg.early_stop_patience, cfg.early_stop_min_delta, mode="min")

    best_val_data = math.inf
    best_epoch = -1

    print(f"[train] device={device} amp={use_amp} epochs={cfg.epochs} "
          f"seed={cfg.seed} run={cfg.run_name}")
    print(f"[train] out_dir={cfg.out_dir}")

    try:
        # GPU memory snapshot if available
        if device.type == "cuda":
            print(f"[train] cuda_mem_total={torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")
    except Exception:
        pass

    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(
            model, loss_fn, loaders["train"], optimizer, device,
            scaler=amp_scaler, grad_clip=cfg.grad_clip_max_norm, amp=use_amp,
        )
        va = evaluate(model, loss_fn, loaders["val"], device)

        if scheduler is not None:
            scheduler.step()

        # coefficient snapshot
        coeffs_snap: Dict[str, float] = {}
        if hasattr(model, "coefficient_values"):
            coeffs_snap = model.coefficient_values()
            logger.log_coefficients(epoch, coeffs_snap)

        cur_lr_mlp = optimizer.param_groups[0]["lr"]
        cur_lr_coeff = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else cur_lr_mlp

        row = {
            "epoch": epoch,
            "lr_mlp": cur_lr_mlp,
            "lr_coeff": cur_lr_coeff,
            "train_total": tr["total"],
            "train_data": tr["data"],
            "train_stationary": tr["stationary"],
            "train_nonholonomic": tr["nonholonomic"],
            "train_coeff": tr["coeff"],
            "train_magnitude": tr["magnitude"],
            "train_elapsed_s": tr["elapsed_s"],
            "val_total": va["total"],
            "val_data": va["data"],
            "val_stationary": va["stationary"],
            "val_nonholonomic": va["nonholonomic"],
            "val_coeff": va["coeff"],
            "val_magnitude": va["magnitude"],
            **{f"coef_{k}": v for k, v in coeffs_snap.items()},
        }
        if device.type == "cuda":
            row["cuda_mem_alloc_gb"] = torch.cuda.memory_allocated() / 1e9
            row["cuda_mem_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        logger.log_epoch(row)

        # best checkpoint on val L_data
        improved = stopper.step(va["data"])
        if improved:
            best_val_data = va["data"]
            best_epoch = epoch
            if cfg.save_best:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_data": va["data"],
                        "val_total": va["total"],
                        "coefficients": coeffs_snap,
                        "config": config_dump,
                    },
                    best_path,
                )

        if epoch % cfg.log_every == 0 or improved or stopper.stopped:
            msg = (
                f"[epoch {epoch:4d}/{cfg.epochs}] "
                f"tr_total={tr['total']:.4e} tr_data={tr['data']:.4e} "
                f"va_data={va['data']:.4e} "
                f"tr_stat={tr['stationary']:.3e} tr_nh={tr['nonholonomic']:.3e} "
                f"lr={cur_lr_mlp:.2e}"
            )
            if coeffs_snap:
                msg += (
                    f" | b={coeffs_snap.get('b', 0):.3f} "
                    f"sr={coeffs_snap.get('s_r', 0):.3f} "
                    f"asum={coeffs_snap.get('alpha_sum', 0):.3f}"
                )
            if improved:
                msg += " *"
            print(msg)

        if stopper.stopped:
            print(f"[train] early stop at epoch {epoch} (best epoch={best_epoch}, "
                  f"val_data={best_val_data:.4e})")
            break

        logger.flush()  # periodic flush for crash resilience

    logger.flush()

    return {
        "history_path": history_path,
        "best_path": best_path if cfg.save_best else None,
        "best_val_data": best_val_data,
        "best_epoch": best_epoch,
        "stopped_early": stopper.stopped,
        "final_coefficients": (
            model.coefficient_values() if hasattr(model, "coefficient_values") else {}
        ),
    }


# ---------------------------------------------------------------------------
# Requested public API wrapper
# ---------------------------------------------------------------------------
def train_api(
    model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Optional[TrainConfig] = None,
    save_dir: str = "./runs/pinn",
) -> Dict:
    """Spec-compliant wrapper: train(model, loss_fn, train_loader, val_loader, cfg, save_dir)

    Returns a dict with keys:
        history   : the in-memory history payload (epochs rows + coefficients)
        best_path : path to best.pt (or None if save_best=False)

    This wraps the richer `train(...)` entry point that takes a `loaders` dict.
    """
    cfg = cfg or TrainConfig()
    # override out_dir
    cfg_dict = asdict(cfg)
    cfg_dict["out_dir"] = save_dir
    cfg2 = TrainConfig(**cfg_dict)

    loaders = {"train": train_loader, "val": val_loader}
    result = train(model, loss_fn, loaders, cfg2)

    # Load back the JSON payload so callers get a single dict
    history_payload: Dict = {}
    try:
        with open(result["history_path"], "r", encoding="utf-8") as f:
            history_payload = json.load(f)
    except Exception:
        history_payload = {}
    history_payload["best_epoch"] = result["best_epoch"]
    history_payload["best_val_data"] = result["best_val_data"]
    history_payload["stopped_early"] = result["stopped_early"]
    history_payload["final_coefficients"] = result["final_coefficients"]

    return {"history": history_payload, "best_path": result["best_path"]}


def load_best(
    model: nn.Module,
    best_path: str,
    map_location: Optional[str] = None,
) -> Dict:
    """Load best checkpoint weights into ``model`` and return the checkpoint dict."""
    ckpt = torch.load(best_path, map_location=map_location or "cpu")
    state = ckpt.get("model_state") or ckpt.get("model_state_dict")
    if state is None:
        raise KeyError("checkpoint has no model_state/model_state_dict")
    model.load_state_dict(state)
    return ckpt


# ---------------------------------------------------------------------------
# Hyperparameter grid suggestion (manual, commented defaults)
# ---------------------------------------------------------------------------
SUGGESTED_GRID = {
    # Uncomment and iterate manually or wrap with Optuna if needed.
    # "w_nonholonomic": [0.05, 0.1, 0.3],
    # "w_stationary":   [0.5, 1.0, 2.0],
    # "mlp_lr":         [5e-4, 1e-3, 2e-3],
    # "hidden_dim":     [64, 96, 128],
}


# ---------------------------------------------------------------------------
# Smoke test: random tensors, 5 epochs
# ---------------------------------------------------------------------------
def _smoke_test() -> None:
    """Run 5 epochs on synthetic random tensors to validate the loop."""
    import tempfile
    from torch.utils.data import TensorDataset, DataLoader

    # Late import to avoid hard coupling
    try:
        from model_module import build_model, build_loss
    except ImportError:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from model_module import build_model, build_loss

    set_seed(0)
    N, B = 256, 32
    x = torch.randn(N, 6)
    x_raw = torch.randn(N, 6) * 0.05
    y = torch.stack([x_raw[:, 0] * 0.98, x_raw[:, 5] * 1.02], dim=-1)
    # stationary: rows whose raw translational+angular norm is small
    stat = (x_raw.norm(dim=-1) < 0.02)
    dt = torch.ones(N) * 0.02

    class _DictDS(torch.utils.data.Dataset):
        def __len__(self): return N
        def __getitem__(self, i):
            return {"x": x[i], "x_raw": x_raw[i], "y": y[i],
                    "stationary": stat[i], "dt": dt[i]}

    ds = _DictDS()
    # split by index slices (time-ordered)
    n_tr = int(N * 0.7)
    n_va = int(N * 0.15)
    tr_idx = list(range(0, n_tr))
    va_idx = list(range(n_tr, n_tr + n_va))
    te_idx = list(range(n_tr + n_va, N))

    def _subset_loader(idx_list, batch_size=B, shuffle=False):
        sub = torch.utils.data.Subset(ds, idx_list)
        return DataLoader(sub, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    loaders = {
        "train": _subset_loader(tr_idx, shuffle=False),
        "val":   _subset_loader(va_idx),
        "test":  _subset_loader(te_idx),
    }

    model = build_model({"name": "pinn",
                         "model": {"hidden_dim": 32, "hidden_blocks": 2, "dropout": 0.1}})
    loss_fn = build_loss({"loss": {"data_weights": (1.0, 10.0)}})

    tmp = tempfile.mkdtemp(prefix="pinn_smoke_")
    cfg = TrainConfig(
        epochs=5, mlp_lr=1e-3, coeff_lr_mult=0.3, weight_decay=1e-4,
        early_stop_patience=10, out_dir=tmp, run_name="smoke",
        amp=False, deterministic=True, log_every=1, seed=0,
    )
    result = train(model, loss_fn, loaders, cfg,
                   extra_config={"note": "smoke-test"})

    assert os.path.exists(result["history_path"]), "history.json missing"
    assert result["best_path"] is None or os.path.exists(result["best_path"]), "best.pt missing"
    print(f"[smoke] best_val_data={result['best_val_data']:.6f} "
          f"best_epoch={result['best_epoch']} "
          f"final_coeffs={result['final_coefficients']}")
    print(f"[smoke] OK -> {tmp}")


if __name__ == "__main__":
    _smoke_test()
