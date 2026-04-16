"""
eval_module.py — Inverse-PINN Odometry Correction Evaluation

Evaluation utilities. CPU-safe, no external deps beyond numpy/pandas/torch/matplotlib.

Public API:
    predict_deltas(model, loader, device='cpu')
        -> dict(pred_dx, pred_dyaw, gt_dx, gt_dyaw, odom_dx, odom_dyaw, stationary)

    per_step_metrics(pred_dx, pred_dyaw, gt_dx, gt_dyaw, odom_dx, odom_dyaw)
        -> dict of MAE / RMSE (pred-vs-gt and odom-vs-gt identity baseline)

    build_trajectories(pred_dx, pred_dyaw, gt_dx, gt_dy, gt_dyaw,
                       odom_dx, odom_dy, odom_dyaw)
        -> dict of three trajectories {'gt','odom','pinn'} each (x, y, yaw)

    trajectory_metrics(traj, ref, rpe_k=10) -> dict(ATE_RMSE, FPE, heading_err_RMSE, RPE_trans, RPE_rot)

    stationary_residual(pred_dx, pred_dyaw, stationary) -> dict(mean_abs_dx, max_abs_dx, ...)

    coefficient_summary(final_coeffs, history) -> dict with final + last50-std convergence metrics

    plot_trajectories(trajs, save_path=None) -> matplotlib Figure
    plot_error_over_time(trajs, save_path=None) -> matplotlib Figure

    run_full_evaluation(model, loader, data_artifacts, save_dir,
                        history=None, coeffs=None, device='cpu') -> dict
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _wrap(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))


def _accumulate(d_x: np.ndarray, d_y: np.ndarray, d_yaw: np.ndarray,
                x0: float = 0.0, y0: float = 0.0, yaw0: float = 0.0):
    """Local fallback copy of data_module.accumulate_trajectory (2D Scout)."""
    d_x = np.asarray(d_x, dtype=np.float64).reshape(-1)
    d_y = np.asarray(d_y, dtype=np.float64).reshape(-1)
    d_yaw = np.asarray(d_yaw, dtype=np.float64).reshape(-1)
    n = len(d_x)
    x = np.empty(n + 1)
    y = np.empty(n + 1)
    yaw = np.empty(n + 1)
    x[0], y[0], yaw[0] = x0, y0, yaw0
    for k in range(n):
        c = math.cos(yaw[k]); s = math.sin(yaw[k])
        x[k + 1] = x[k] + d_x[k] * c - d_y[k] * s
        y[k + 1] = y[k] + d_x[k] * s + d_y[k] * c
        yaw[k + 1] = _wrap(np.array([yaw[k] + d_yaw[k]]))[0]
    return x, y, yaw


def _get_accumulate():
    """Prefer the canonical implementation in data_module; fallback otherwise."""
    try:
        from data_module import accumulate_trajectory  # type: ignore
        return accumulate_trajectory
    except Exception:
        try:
            from .data_module import accumulate_trajectory  # type: ignore
            return accumulate_trajectory
        except Exception:
            return _accumulate


# ---------------------------------------------------------------------------
# 1) Predict all deltas from a loader
# ---------------------------------------------------------------------------
def predict_deltas(
    model: "nn.Module",
    loader,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """Run model over the entire loader and collect per-step arrays.

    Returns dict of 1D numpy arrays (length = total samples in loader):
        pred_dx, pred_dyaw  - model predictions in original units
        gt_dx,   gt_dyaw    - supervised targets
        odom_dx, odom_dyaw  - raw odometry (identity baseline)
        stationary          - bool mask
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for predict_deltas()")
    device_t = torch.device(device)
    model = model.to(device_t)
    model.eval()

    preds, gts, odoms, stats = [], [], [], []
    is_baseline = type(model).__name__ in ("IdentityBaseline", "LinearBaseline")

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device_t) for k, v in batch.items()}
            if is_baseline:
                pred = model(batch["x"], x_raw=batch["x_raw"])
            else:
                pred = model(batch["x"])
            preds.append(pred.detach().cpu().numpy())
            gts.append(batch["y"].detach().cpu().numpy())
            odoms.append(batch["x_raw"].detach().cpu().numpy())
            stats.append(batch["stationary"].detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    odoms = np.concatenate(odoms, axis=0)
    stats = np.concatenate(stats, axis=0).astype(bool)

    # INPUT_COLS order: [d_x, d_y, d_z, d_rolling, d_pitch, d_yaw]
    return {
        "pred_dx": preds[:, 0].astype(np.float64),
        "pred_dyaw": preds[:, 1].astype(np.float64),
        "gt_dx": gts[:, 0].astype(np.float64),
        "gt_dyaw": gts[:, 1].astype(np.float64),
        "odom_dx": odoms[:, 0].astype(np.float64),
        "odom_dy": odoms[:, 1].astype(np.float64),
        "odom_dyaw": odoms[:, 5].astype(np.float64),
        "stationary": stats,
    }


# ---------------------------------------------------------------------------
# 2) Per-step MAE / RMSE
# ---------------------------------------------------------------------------
def _mae(a, b): return float(np.mean(np.abs(a - b)))
def _rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))


def per_step_metrics(
    pred_dx: np.ndarray, pred_dyaw: np.ndarray,
    gt_dx: np.ndarray, gt_dyaw: np.ndarray,
    odom_dx: np.ndarray, odom_dyaw: np.ndarray,
) -> Dict[str, float]:
    """MAE/RMSE for PINN prediction and identity-odometry baseline."""
    out = {
        "pinn_mae_dx": _mae(pred_dx, gt_dx),
        "pinn_mae_dyaw": _mae(pred_dyaw, gt_dyaw),
        "pinn_rmse_dx": _rmse(pred_dx, gt_dx),
        "pinn_rmse_dyaw": _rmse(pred_dyaw, gt_dyaw),
        "odom_mae_dx": _mae(odom_dx, gt_dx),
        "odom_mae_dyaw": _mae(odom_dyaw, gt_dyaw),
        "odom_rmse_dx": _rmse(odom_dx, gt_dx),
        "odom_rmse_dyaw": _rmse(odom_dyaw, gt_dyaw),
    }
    # relative improvement (%) — positive = PINN better
    def _imp(a, b):
        return float((b - a) / b * 100.0) if b > 1e-12 else 0.0
    out["improvement_rmse_dx_pct"] = _imp(out["pinn_rmse_dx"], out["odom_rmse_dx"])
    out["improvement_rmse_dyaw_pct"] = _imp(out["pinn_rmse_dyaw"], out["odom_rmse_dyaw"])
    return out


# ---------------------------------------------------------------------------
# 3) Trajectory construction
# ---------------------------------------------------------------------------
def build_trajectories(
    pred_dx: np.ndarray, pred_dyaw: np.ndarray,
    gt_dx: np.ndarray, gt_dy: np.ndarray, gt_dyaw: np.ndarray,
    odom_dx: np.ndarray, odom_dy: np.ndarray, odom_dyaw: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Accumulate 3 trajectories from the aligned delta arrays.

    The PINN model predicts only (d_x, d_yaw) — for trajectory accumulation
    we set d_y=0 (non-holonomic Scout assumption). Odom uses its own d_y
    (usually near zero); GT uses its d_y.
    """
    accumulate = _get_accumulate()
    zeros = np.zeros_like(pred_dx)
    gt_x, gt_y, gt_yaw = accumulate(gt_dx, gt_dy, gt_dyaw)
    od_x, od_y, od_yaw = accumulate(odom_dx, odom_dy, odom_dyaw)
    pn_x, pn_y, pn_yaw = accumulate(pred_dx, zeros, pred_dyaw)
    return {
        "gt":   {"x": gt_x, "y": gt_y, "yaw": gt_yaw},
        "odom": {"x": od_x, "y": od_y, "yaw": od_yaw},
        "pinn": {"x": pn_x, "y": pn_y, "yaw": pn_yaw},
    }


# ---------------------------------------------------------------------------
# 4) Trajectory metrics: ATE, FPE, Heading Error, RPE
# ---------------------------------------------------------------------------
def trajectory_metrics(
    traj: Dict[str, np.ndarray],
    ref: Dict[str, np.ndarray],
    rpe_k: int = 10,
) -> Dict[str, float]:
    """Absolute & Relative pose-error metrics vs a reference trajectory.

    ATE_RMSE : sqrt(mean(|p - p_ref|^2)) (x, y only)
    FPE      : ||p_N - p_ref_N||
    heading_err_RMSE : RMSE on wrapped (yaw - yaw_ref)
    RPE_trans, RPE_rot : mean over k-step relative motion differences

    All inputs assumed same length (N+1 poses including origin).
    """
    x, y, yaw = traj["x"], traj["y"], traj["yaw"]
    xr, yr, yr_yaw = ref["x"], ref["y"], ref["yaw"]

    n = min(len(x), len(xr))
    x, y, yaw = x[:n], y[:n], yaw[:n]
    xr, yr, yr_yaw = xr[:n], yr[:n], yr_yaw[:n]

    dxy = np.sqrt((x - xr) ** 2 + (y - yr) ** 2)
    ate = float(np.sqrt(np.mean(dxy ** 2)))
    fpe = float(dxy[-1])
    head_err = _wrap(yaw - yr_yaw)
    heading_rmse = float(np.sqrt(np.mean(head_err ** 2)))

    # RPE (k-step relative translation and rotation errors)
    k = max(1, int(rpe_k))
    if n > k:
        # displacement over k steps
        d_traj = np.stack([x[k:] - x[:-k], y[k:] - y[:-k]], axis=-1)
        d_ref = np.stack([xr[k:] - xr[:-k], yr[k:] - yr[:-k]], axis=-1)
        rpe_trans = float(np.sqrt(np.mean(np.sum((d_traj - d_ref) ** 2, axis=-1))))
        d_yaw_traj = _wrap(yaw[k:] - yaw[:-k])
        d_yaw_ref = _wrap(yr_yaw[k:] - yr_yaw[:-k])
        rpe_rot = float(np.sqrt(np.mean(_wrap(d_yaw_traj - d_yaw_ref) ** 2)))
    else:
        rpe_trans = float("nan"); rpe_rot = float("nan")

    # Total path length of reference (for normalization context)
    path_len = float(np.sum(np.sqrt(np.diff(xr) ** 2 + np.diff(yr) ** 2)))

    return {
        "ATE_RMSE": ate,
        "FPE": fpe,
        "heading_err_RMSE": heading_rmse,
        f"RPE{k}_trans": rpe_trans,
        f"RPE{k}_rot": rpe_rot,
        "ref_path_length": path_len,
        "FPE_over_path_pct": float(fpe / path_len * 100.0) if path_len > 1e-9 else float("nan"),
    }


# ---------------------------------------------------------------------------
# 5) Stationary residual
# ---------------------------------------------------------------------------
def stationary_residual(
    pred_dx: np.ndarray, pred_dyaw: np.ndarray, stationary: np.ndarray,
) -> Dict[str, float]:
    stationary = np.asarray(stationary, dtype=bool)
    n_stat = int(stationary.sum())
    if n_stat == 0:
        return {"n_stationary": 0, "mean_abs_dx": float("nan"),
                "max_abs_dx": float("nan"), "mean_abs_dyaw": float("nan"),
                "max_abs_dyaw": float("nan")}
    sx = np.abs(pred_dx[stationary])
    sy = np.abs(pred_dyaw[stationary])
    return {
        "n_stationary": n_stat,
        "stationary_ratio": float(stationary.mean()),
        "mean_abs_dx": float(sx.mean()),
        "max_abs_dx": float(sx.max()),
        "mean_abs_dyaw": float(sy.mean()),
        "max_abs_dyaw": float(sy.max()),
    }


# ---------------------------------------------------------------------------
# 6) Coefficient summary + convergence
# ---------------------------------------------------------------------------
def coefficient_summary(
    final_coeffs: Optional[Dict[str, float]],
    history: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"final": dict(final_coeffs) if final_coeffs else {}}
    if history is None:
        return out
    coeff_hist: List[Dict] = history.get("coefficients") or []
    if not coeff_hist:
        return out

    keys = ["b", "s_r", "alpha_sum"]
    tail = coeff_hist[-min(50, len(coeff_hist)):]
    conv: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = np.array([row.get(k, np.nan) for row in tail], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        conv[k] = {
            "last": float(vals[-1]),
            "last50_mean": float(vals.mean()),
            "last50_std": float(vals.std()),
            "last50_cv": float(vals.std() / (abs(vals.mean()) + 1e-12)),
        }
    out["convergence"] = conv
    # physical deviation from Scout priors (alpha_sum prior = 2.0 for symmetric L=R=1)
    priors = {"b": 0.49, "s_r": 1.0, "alpha_sum": 2.0}
    out["deviation_from_prior"] = {
        k: (conv.get(k, {}).get("last", float("nan")) - p)
        for k, p in priors.items()
    }
    # Derived: mean left/right wheel scale = alpha_sum / 2
    if "alpha_sum" in conv:
        out["alpha_mean"] = conv["alpha_sum"]["last"] / 2.0
    return out


# ---------------------------------------------------------------------------
# 7) Plots
# ---------------------------------------------------------------------------
def plot_trajectories(
    trajs: Dict[str, Dict[str, np.ndarray]],
    save_path: Optional[str] = None,
    title: str = "Trajectory: GT vs Odom vs PINN",
):
    if not _HAS_MPL:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(8, 9),
                             gridspec_kw={"height_ratios": [3, 1]})
    ax1, ax2 = axes

    colors = {"gt": "k", "odom": "tab:red", "pinn": "tab:blue"}
    styles = {"gt": "-", "odom": "--", "pinn": "-"}
    for name, tr in trajs.items():
        ax1.plot(tr["x"], tr["y"],
                 color=colors.get(name, None), linestyle=styles.get(name, "-"),
                 label=name.upper(), linewidth=1.5)
        ax1.scatter([tr["x"][0]], [tr["y"][0]], color=colors.get(name, "k"),
                    marker="o", s=25)
        ax1.scatter([tr["x"][-1]], [tr["y"][-1]], color=colors.get(name, "k"),
                    marker="s", s=25)
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]")
    ax1.set_title(title)
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.grid(True, alpha=0.3); ax1.legend()

    for name, tr in trajs.items():
        ax2.plot(tr["yaw"],
                 color=colors.get(name, None), linestyle=styles.get(name, "-"),
                 label=name.upper(), linewidth=1.2)
    ax2.set_xlabel("step"); ax2.set_ylabel("yaw [rad]")
    ax2.grid(True, alpha=0.3); ax2.legend()

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_error_over_time(
    trajs: Dict[str, Dict[str, np.ndarray]],
    save_path: Optional[str] = None,
    title: str = "Cumulative position error vs GT",
):
    if not _HAS_MPL:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ref = trajs["gt"]
    for name in ("odom", "pinn"):
        if name not in trajs:
            continue
        tr = trajs[name]
        n = min(len(tr["x"]), len(ref["x"]))
        err = np.sqrt((tr["x"][:n] - ref["x"][:n]) ** 2
                      + (tr["y"][:n] - ref["y"][:n]) ** 2)
        ax.plot(err, label=name.upper(), linewidth=1.3)
    ax.set_xlabel("step"); ax.set_ylabel("position error [m]")
    ax.set_title(title); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 8) Top-level orchestration
# ---------------------------------------------------------------------------
def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def run_full_evaluation(
    model: "nn.Module",
    loader,
    save_dir: str,
    history: Optional[Dict[str, Any]] = None,
    final_coeffs: Optional[Dict[str, float]] = None,
    device: str = "cpu",
    rpe_k: int = 10,
    tag: str = "test",
) -> Dict[str, Any]:
    """End-to-end evaluation.

    Produces inside save_dir:
        metrics.json      — all scalar metrics
        trajectory.png    — 2D trajectories + yaw time series
        error_over_time.png — cumulative translational error
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1) predict
    arr = predict_deltas(model, loader, device=device)

    # 2) per-step metrics
    ps = per_step_metrics(
        arr["pred_dx"], arr["pred_dyaw"],
        arr["gt_dx"], arr["gt_dyaw"],
        arr["odom_dx"], arr["odom_dyaw"],
    )

    # 3) build 3 trajectories (need gt_dy from odom side? we only have gt_dx/dyaw via loader targets)
    # We treat gt_dy ≈ 0 (non-holonomic + body frame); this matches the model's assumption.
    # For Odom side we use odom_dy (may be small noise).
    gt_dy = np.zeros_like(arr["gt_dx"])
    trajs = build_trajectories(
        arr["pred_dx"], arr["pred_dyaw"],
        arr["gt_dx"], gt_dy, arr["gt_dyaw"],
        arr["odom_dx"], arr["odom_dy"], arr["odom_dyaw"],
    )

    # 4) trajectory metrics (reference = GT)
    tm_odom = trajectory_metrics(trajs["odom"], trajs["gt"], rpe_k=rpe_k)
    tm_pinn = trajectory_metrics(trajs["pinn"], trajs["gt"], rpe_k=rpe_k)

    # 5) stationary residual
    sres = stationary_residual(arr["pred_dx"], arr["pred_dyaw"], arr["stationary"])

    # 6) coefficient summary
    cs = coefficient_summary(final_coeffs, history)

    # 7) plots
    traj_png = os.path.join(save_dir, f"{tag}_trajectory.png")
    err_png = os.path.join(save_dir, f"{tag}_error_over_time.png")
    try:
        plot_trajectories(trajs, save_path=traj_png,
                          title=f"[{tag}] Trajectory: GT vs Odom vs PINN")
        plot_error_over_time(trajs, save_path=err_png,
                             title=f"[{tag}] Cumulative position error vs GT")
    except Exception as e:
        print(f"[eval] plotting failed: {e}")

    # 8) assemble
    metrics = {
        "tag": tag,
        "n_samples": int(len(arr["pred_dx"])),
        "per_step": ps,
        "trajectory": {
            "odom_vs_gt": tm_odom,
            "pinn_vs_gt": tm_pinn,
            "improvement": {
                "ATE_RMSE_reduction_pct": (
                    (tm_odom["ATE_RMSE"] - tm_pinn["ATE_RMSE"])
                    / max(tm_odom["ATE_RMSE"], 1e-12) * 100.0
                ),
                "FPE_reduction_pct": (
                    (tm_odom["FPE"] - tm_pinn["FPE"])
                    / max(tm_odom["FPE"], 1e-12) * 100.0
                ),
                "heading_RMSE_reduction_pct": (
                    (tm_odom["heading_err_RMSE"] - tm_pinn["heading_err_RMSE"])
                    / max(tm_odom["heading_err_RMSE"], 1e-12) * 100.0
                ),
            },
        },
        "stationary_residual": sres,
        "coefficients": cs,
        "plots": {"trajectory": traj_png, "error_over_time": err_png},
    }

    json_path = os.path.join(save_dir, f"{tag}_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(metrics), f, indent=2, default=float)
    metrics["metrics_json"] = json_path
    print(f"[eval] wrote {json_path}")

    return {"metrics": metrics, "arrays": arr, "trajectories": trajs}


# ---------------------------------------------------------------------------
# 9) Checkpoint loader convenience
# ---------------------------------------------------------------------------
def load_model_from_checkpoint(
    model: "nn.Module", ckpt_path: str, map_location: str = "cpu",
) -> Dict[str, Any]:
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch required")
    ckpt = torch.load(ckpt_path, map_location=map_location)
    state = ckpt.get("model_state") or ckpt.get("model_state_dict")
    if state is None:
        raise KeyError("checkpoint has no model_state/model_state_dict")
    model.load_state_dict(state)
    return ckpt


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _smoke_test() -> None:
    """Random-tensor smoke test: predict -> metrics -> trajectories -> plots."""
    import tempfile
    from torch.utils.data import Dataset, DataLoader

    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from model_module import build_model  # type: ignore

    torch.manual_seed(0)
    np.random.seed(0)
    N = 200
    x = torch.randn(N, 6) * 0.5
    x_raw = torch.randn(N, 6) * 0.05
    x_raw[:20] = 0.0  # stationary block
    # target: GT roughly = scaled odom
    y = torch.stack([x_raw[:, 0] * 0.95, x_raw[:, 5] * 1.05], dim=-1)
    stat = torch.zeros(N, dtype=torch.bool); stat[:20] = True
    dt = torch.ones(N) * 0.02

    class DS(Dataset):
        def __len__(self): return N
        def __getitem__(self, i):
            return {"x": x[i], "x_raw": x_raw[i], "y": y[i],
                    "stationary": stat[i], "dt": dt[i]}

    loader = DataLoader(DS(), batch_size=32, shuffle=False)

    model = build_model({"name": "pinn",
                         "model": {"hidden_dim": 32, "hidden_blocks": 2, "dropout": 0.0}})
    # fake history / coeffs
    fake_history = {"coefficients": [
        {"epoch": e, "b": 0.49 + 0.001 * e, "s_r": 1.0 + 0.0005 * e,
         "alpha_sum": 2.0} for e in range(1, 60)
    ]}

    tmp = tempfile.mkdtemp(prefix="pinn_eval_smoke_")
    out = run_full_evaluation(
        model, loader, save_dir=tmp,
        history=fake_history,
        final_coeffs=model.coefficient_values(),
        device="cpu", rpe_k=10, tag="smoke",
    )
    m = out["metrics"]
    print(f"[smoke] n={m['n_samples']}")
    print(f"[smoke] pinn rmse dx={m['per_step']['pinn_rmse_dx']:.4e} "
          f"dyaw={m['per_step']['pinn_rmse_dyaw']:.4e}")
    print(f"[smoke] odom rmse dx={m['per_step']['odom_rmse_dx']:.4e} "
          f"dyaw={m['per_step']['odom_rmse_dyaw']:.4e}")
    print(f"[smoke] ATE pinn={m['trajectory']['pinn_vs_gt']['ATE_RMSE']:.4e} "
          f"odom={m['trajectory']['odom_vs_gt']['ATE_RMSE']:.4e}")
    print(f"[smoke] FPE pinn={m['trajectory']['pinn_vs_gt']['FPE']:.4e} "
          f"odom={m['trajectory']['odom_vs_gt']['FPE']:.4e}")
    print(f"[smoke] stationary residual: {m['stationary_residual']}")
    print(f"[smoke] coeff final={m['coefficients']['final']}")
    print(f"[smoke] artifacts dir={tmp}")
    print("[smoke] OK")


if __name__ == "__main__":
    _smoke_test()
