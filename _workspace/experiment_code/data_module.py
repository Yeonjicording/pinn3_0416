"""
data_module.py — Inverse-PINN Odometry Correction

Data pipeline utilities for paired Ground Truth / Odometry CSV files.

Runs in Google Colab and locally. Exposes:
    upload_data_colab() -> (gt_path, odom_path)
    load_and_align(gt_path, odom_path, tol=None, method='nearest') -> pd.DataFrame
    detect_stationary(df, thresh_trans=1e-4, thresh_ang=1e-4) -> np.ndarray[bool]
    split_timeseries(df, ratios=(0.7, 0.15, 0.15)) -> (train, val, test)
    StandardScaler1D (fit/transform/inverse_transform)
    fit_scaler(train_df, cols) -> StandardScaler1D
    make_tensors(df, scaler, stationary_mask=None) -> dict[str, torch.Tensor]
    class OdomDataset(torch.utils.data.Dataset)
    build_dataloaders(df_train, df_val, df_test, scaler, batch_size, ...) -> dict
    accumulate_trajectory(d_x, d_y, d_yaw, x0=0.0, y0=0.0, yaw0=0.0) -> (x, y, yaw)

No heavy external deps. Only: numpy, pandas, torch.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Constants / schema
# ---------------------------------------------------------------------------
EXPECTED_COLS: List[str] = [
    "t", "d_x", "d_y", "d_z", "d_rolling", "d_pitch", "d_yaw",
]
INPUT_COLS: List[str] = ["d_x", "d_y", "d_z", "d_rolling", "d_pitch", "d_yaw"]
TARGET_COLS: List[str] = ["d_x", "d_yaw"]  # taken from GT side


# ---------------------------------------------------------------------------
# Colab upload helper
# ---------------------------------------------------------------------------
def upload_data_colab(
    gt_hint: str = "gt",
    odom_hint: str = "odom",
    save_dir: str = "/content",
) -> Tuple[str, str]:
    """Prompt user to upload 2 CSVs in Colab and return (gt_path, odom_path).

    Classification heuristic:
        - If filename contains `gt_hint` (case-insensitive) -> GT.
        - Else if contains `odom_hint` -> Odom.
        - Else: the first uploaded file = GT, second = Odom (with a printed warning).

    Outside Colab, raises RuntimeError.
    """
    try:
        from google.colab import files  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "google.colab.files not available. Pass file paths to load_and_align() directly."
        ) from e

    print("Please upload 2 CSV files: Ground Truth and Odometry.")
    uploaded = files.upload()  # dict[filename] = bytes
    names = list(uploaded.keys())
    if len(names) < 2:
        raise ValueError(f"Expected 2 files, got {len(names)}: {names}")

    os.makedirs(save_dir, exist_ok=True)
    paths: Dict[str, str] = {}
    for name in names:
        path = os.path.join(save_dir, name)
        with open(path, "wb") as f:
            f.write(uploaded[name])
        paths[name] = path

    gt_path: Optional[str] = None
    odom_path: Optional[str] = None
    for name, path in paths.items():
        low = name.lower()
        if gt_hint.lower() in low and gt_path is None:
            gt_path = path
        elif odom_hint.lower() in low and odom_path is None:
            odom_path = path

    if gt_path is None or odom_path is None:
        # fallback by upload order
        ordered = list(paths.values())
        gt_path = gt_path or ordered[0]
        odom_path = odom_path or (ordered[1] if ordered[1] != gt_path else ordered[0])
        print(
            f"[warn] Could not detect GT/Odom from filenames; "
            f"using first={os.path.basename(gt_path)} as GT, "
            f"second={os.path.basename(odom_path)} as Odom."
        )

    print(f"GT   : {gt_path}")
    print(f"Odom : {odom_path}")
    return gt_path, odom_path


# ---------------------------------------------------------------------------
# CSV IO + schema validation
# ---------------------------------------------------------------------------
def _read_csv_validated(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"File {path} missing required columns {missing}. Got {list(df.columns)}"
        )
    df = df[EXPECTED_COLS].copy()
    # drop NaN/Inf
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=EXPECTED_COLS)
    # ensure monotonic t (drop non-monotone rows, keep stable)
    t = df["t"].to_numpy()
    if len(t) < 2:
        return df.reset_index(drop=True)
    keep = np.concatenate([[True], np.diff(t) > 0])
    if not keep.all():
        df = df.iloc[keep].copy()
    # dedup identical t (keep first)
    df = df.drop_duplicates(subset=["t"], keep="first").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Timestamp alignment
# ---------------------------------------------------------------------------
def load_and_align(
    gt_path: str,
    odom_path: str,
    tol: Optional[float] = None,
    method: str = "nearest",
) -> pd.DataFrame:
    """Load two CSVs and align on timestamp.

    Returns a DataFrame with columns:
        t
        gt_d_x, gt_d_y, gt_d_z, gt_d_rolling, gt_d_pitch, gt_d_yaw
        od_d_x, od_d_y, od_d_z, od_d_rolling, od_d_pitch, od_d_yaw

    method='nearest' : merge_asof with direction='nearest' (default).
    method='cumdiff' : cumulate -> interpolate onto odom grid -> diff (physically consistent).

    tol : matching tolerance in same units as `t`. Defaults to 0.5 * median(Δt_odom).
    """
    gt = _read_csv_validated(gt_path)
    od = _read_csv_validated(odom_path)

    if len(gt) == 0 or len(od) == 0:
        raise ValueError("One of the input CSVs is empty after cleaning.")

    # intersect time range
    t_lo = max(gt["t"].min(), od["t"].min())
    t_hi = min(gt["t"].max(), od["t"].max())
    gt = gt[(gt["t"] >= t_lo) & (gt["t"] <= t_hi)].reset_index(drop=True)
    od = od[(od["t"] >= t_lo) & (od["t"] <= t_hi)].reset_index(drop=True)
    if len(gt) < 2 or len(od) < 2:
        raise ValueError("Insufficient overlap between GT and Odom time ranges.")

    dt_od = float(np.median(np.diff(od["t"].to_numpy())))
    if tol is None:
        tol = 0.5 * dt_od

    if method == "nearest":
        od_sorted = od.sort_values("t").reset_index(drop=True)
        gt_sorted = gt.sort_values("t").reset_index(drop=True)
        merged = pd.merge_asof(
            od_sorted.rename(columns={c: f"od_{c}" for c in INPUT_COLS}),
            gt_sorted.rename(columns={c: f"gt_{c}" for c in INPUT_COLS}),
            on="t",
            direction="nearest",
            tolerance=tol,
        )
        # drop rows where gt side is NaN (beyond tolerance)
        gt_cols = [f"gt_{c}" for c in INPUT_COLS]
        before = len(merged)
        merged = merged.dropna(subset=gt_cols).reset_index(drop=True)
        dropped = before - len(merged)
        if dropped > 0:
            print(f"[align/nearest] dropped {dropped}/{before} rows beyond tol={tol:.6g}")
        return merged

    elif method == "cumdiff":
        # Cumulate GT deltas -> interpolate onto odom t -> diff
        t_gt = gt["t"].to_numpy()
        t_od = od["t"].to_numpy()
        merged = pd.DataFrame({"t": t_od})
        for c in INPUT_COLS:
            cum_gt = np.concatenate([[0.0], np.cumsum(gt[c].to_numpy())])
            t_gt_ext = np.concatenate([[t_gt[0] - (t_gt[1] - t_gt[0])], t_gt])
            cum_on_od = np.interp(t_od, t_gt_ext, cum_gt)
            d_on_od = np.diff(cum_on_od, prepend=cum_on_od[0])
            merged[f"gt_{c}"] = d_on_od
            merged[f"od_{c}"] = od[c].to_numpy()
        return merged

    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Stationary mask
# ---------------------------------------------------------------------------
def detect_stationary(
    df: pd.DataFrame,
    thresh_trans: float = 1e-4,
    thresh_ang: float = 1e-4,
    side: str = "od",
) -> np.ndarray:
    """Return boolean mask where odometry (default) indicates stationary.

    side: 'od' or 'gt'.
    """
    prefix = f"{side}_"
    trans = np.sqrt(
        df[f"{prefix}d_x"] ** 2 + df[f"{prefix}d_y"] ** 2 + df[f"{prefix}d_z"] ** 2
    ).to_numpy()
    ang = np.sqrt(
        df[f"{prefix}d_rolling"] ** 2
        + df[f"{prefix}d_pitch"] ** 2
        + df[f"{prefix}d_yaw"] ** 2
    ).to_numpy()
    mask = (trans < thresh_trans) & (ang < thresh_ang)
    ratio = float(mask.mean()) if len(mask) > 0 else 0.0
    print(f"[stationary] ratio={ratio*100:.2f}% (n={int(mask.sum())}/{len(mask)})")
    return mask


# ---------------------------------------------------------------------------
# Time-series split
# ---------------------------------------------------------------------------
def split_timeseries(
    df: pd.DataFrame,
    ratios: Sequence[float] = (0.7, 0.15, 0.15),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")
    n = len(df)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])
    train = df.iloc[:n_tr].reset_index(drop=True)
    val = df.iloc[n_tr : n_tr + n_va].reset_index(drop=True)
    test = df.iloc[n_tr + n_va :].reset_index(drop=True)
    return train, val, test


# ---------------------------------------------------------------------------
# Scaler (lightweight, numpy-based; inverse-transform supported)
# ---------------------------------------------------------------------------
class StandardScaler1D:
    def __init__(self, eps: float = 1e-8):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.eps = eps
        self.cols: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame, cols: Sequence[str]) -> "StandardScaler1D":
        self.cols = list(cols)
        arr = df[self.cols].to_numpy(dtype=np.float64)
        self.mean_ = arr.mean(axis=0)
        self.std_ = arr.std(axis=0) + self.eps
        return self

    def transform(self, df_or_arr) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None, "fit() first"
        if isinstance(df_or_arr, pd.DataFrame):
            arr = df_or_arr[self.cols].to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(df_or_arr, dtype=np.float64)
        return (arr - self.mean_) / self.std_

    def inverse_transform(self, arr) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None, "fit() first"
        arr = np.asarray(arr, dtype=np.float64)
        return arr * self.std_ + self.mean_


def fit_scaler(train_df: pd.DataFrame, cols: Sequence[str]) -> StandardScaler1D:
    s = StandardScaler1D().fit(train_df, cols)
    return s


# ---------------------------------------------------------------------------
# Tensor packing
# ---------------------------------------------------------------------------
def make_tensors(
    df: pd.DataFrame,
    scaler: StandardScaler1D,
    stationary_mask: Optional[np.ndarray] = None,
) -> Dict[str, "torch.Tensor"]:
    """Pack aligned df into tensors.

    Returns dict with:
        x        : (N, 6) standardized odom input
        x_raw    : (N, 6) raw odom input (original units)
        y        : (N, 2) GT [d_x, d_yaw] in original units
        stationary : (N,) bool
        dt       : (N,) float (time step; first element copies second)
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch required for make_tensors()")
    od_cols = [f"od_{c}" for c in INPUT_COLS]
    gt_cols = [f"gt_{c}" for c in TARGET_COLS]

    x_raw = df[od_cols].to_numpy(dtype=np.float32)
    x_std = scaler.transform(df[od_cols]).astype(np.float32)
    y = df[gt_cols].to_numpy(dtype=np.float32)

    t = df["t"].to_numpy(dtype=np.float64)
    if len(t) >= 2:
        dt = np.diff(t, prepend=t[0] - (t[1] - t[0])).astype(np.float32)
    else:
        dt = np.array([0.0], dtype=np.float32)

    if stationary_mask is None:
        stationary_mask = np.zeros(len(df), dtype=bool)
    stat = np.asarray(stationary_mask, dtype=bool)

    return {
        "x": torch.from_numpy(x_std),
        "x_raw": torch.from_numpy(x_raw),
        "y": torch.from_numpy(y),
        "stationary": torch.from_numpy(stat),
        "dt": torch.from_numpy(dt),
    }


# ---------------------------------------------------------------------------
# PyTorch Dataset / DataLoader
# ---------------------------------------------------------------------------
if _HAS_TORCH:

    class OdomDataset(Dataset):
        """Simple index-based dataset over aligned odom/gt tensors."""

        def __init__(
            self,
            df: pd.DataFrame,
            scaler: StandardScaler1D,
            stationary_mask: Optional[np.ndarray] = None,
        ):
            tensors = make_tensors(df, scaler, stationary_mask)
            self.x = tensors["x"]
            self.x_raw = tensors["x_raw"]
            self.y = tensors["y"]
            self.stationary = tensors["stationary"]
            self.dt = tensors["dt"]

        def __len__(self) -> int:
            return self.x.shape[0]

        def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
            return {
                "x": self.x[idx],
                "x_raw": self.x_raw[idx],
                "y": self.y[idx],
                "stationary": self.stationary[idx],
                "dt": self.dt[idx],
            }

    def build_dataloaders(
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        scaler: StandardScaler1D,
        batch_size: int = 256,
        num_workers: int = 0,
        shuffle_train: bool = False,
        stationary_kwargs: Optional[dict] = None,
    ) -> Dict[str, DataLoader]:
        stationary_kwargs = stationary_kwargs or {}
        m_tr = detect_stationary(df_train, **stationary_kwargs)
        m_va = detect_stationary(df_val, **stationary_kwargs)
        m_te = detect_stationary(df_test, **stationary_kwargs)
        ds_tr = OdomDataset(df_train, scaler, m_tr)
        ds_va = OdomDataset(df_val, scaler, m_va)
        ds_te = OdomDataset(df_test, scaler, m_te)
        return {
            "train": DataLoader(
                ds_tr,
                batch_size=batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers,
                drop_last=False,
            ),
            "val": DataLoader(
                ds_va,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
            ),
            "test": DataLoader(
                ds_te,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
            ),
        }


# ---------------------------------------------------------------------------
# Trajectory accumulation
# ---------------------------------------------------------------------------
def _wrap_angle(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))


def accumulate_trajectory(
    d_x: np.ndarray,
    d_y: np.ndarray,
    d_yaw: np.ndarray,
    x0: float = 0.0,
    y0: float = 0.0,
    yaw0: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate body-frame deltas to world-frame (x, y, yaw) 2D trajectory.

    Uses yaw at beginning of each step (simple Euler integration).
    Scout differential drive; d_y is typically ~0 (non-holonomic) but supported.
    """
    d_x = np.asarray(d_x, dtype=np.float64).reshape(-1)
    d_y = np.asarray(d_y, dtype=np.float64).reshape(-1)
    d_yaw = np.asarray(d_yaw, dtype=np.float64).reshape(-1)
    n = len(d_x)
    assert len(d_y) == n and len(d_yaw) == n

    x = np.empty(n + 1, dtype=np.float64)
    y = np.empty(n + 1, dtype=np.float64)
    yaw = np.empty(n + 1, dtype=np.float64)
    x[0], y[0], yaw[0] = x0, y0, yaw0
    for k in range(n):
        c = np.cos(yaw[k])
        s = np.sin(yaw[k])
        x[k + 1] = x[k] + d_x[k] * c - d_y[k] * s
        y[k + 1] = y[k] + d_x[k] * s + d_y[k] * c
        yaw[k + 1] = _wrap_angle(yaw[k] + d_yaw[k])
    return x, y, yaw


# ---------------------------------------------------------------------------
# Optional: end-to-end helper
# ---------------------------------------------------------------------------
def prepare_all(
    gt_path: str,
    odom_path: str,
    ratios: Sequence[float] = (0.7, 0.15, 0.15),
    align_method: str = "nearest",
    align_tol: Optional[float] = None,
    stationary_kwargs: Optional[dict] = None,
    batch_size: int = 256,
    num_workers: int = 0,
) -> Dict:
    """Convenience pipeline: align -> split -> fit scaler -> build loaders."""
    df = load_and_align(gt_path, odom_path, tol=align_tol, method=align_method)
    df_tr, df_va, df_te = split_timeseries(df, ratios=ratios)
    scaler = fit_scaler(df_tr, [f"od_{c}" for c in INPUT_COLS])
    loaders = build_dataloaders(
        df_tr, df_va, df_te, scaler,
        batch_size=batch_size,
        num_workers=num_workers,
        stationary_kwargs=stationary_kwargs,
    )
    return {
        "df_aligned": df,
        "df_train": df_tr,
        "df_val": df_va,
        "df_test": df_te,
        "scaler": scaler,
        "loaders": loaders,
    }


# ---------------------------------------------------------------------------
# Self-test with synthetic data
# ---------------------------------------------------------------------------
def _selftest(tmpdir: str = ".") -> None:
    rng = np.random.default_rng(42)
    n = 500
    t = np.linspace(0.0, 10.0, n)  # 50 Hz
    # GT: circular motion
    d_x_gt = 0.1 * np.ones(n) * (np.abs(np.sin(t * 0.5)) > 0.1)  # include stops
    d_y_gt = np.zeros(n)
    d_yaw_gt = 0.02 * np.ones(n) * (np.abs(np.sin(t * 0.5)) > 0.1)
    gt = pd.DataFrame({
        "t": t,
        "d_x": d_x_gt,
        "d_y": d_y_gt,
        "d_z": np.zeros(n),
        "d_rolling": np.zeros(n),
        "d_pitch": np.zeros(n),
        "d_yaw": d_yaw_gt,
    })
    # Odom: GT + biased scale + noise; slightly different timestamps
    t_od = t + rng.normal(0, 0.002, size=n)
    t_od = np.sort(t_od)
    od = pd.DataFrame({
        "t": t_od,
        "d_x": d_x_gt * 1.1 + rng.normal(0, 0.005, size=n),
        "d_y": rng.normal(0, 0.001, size=n),
        "d_z": np.zeros(n),
        "d_rolling": np.zeros(n),
        "d_pitch": np.zeros(n),
        "d_yaw": d_yaw_gt * 0.95 + rng.normal(0, 0.001, size=n),
    })
    gt_p = os.path.join(tmpdir, "_synthetic_gt.csv")
    od_p = os.path.join(tmpdir, "_synthetic_odom.csv")
    gt.to_csv(gt_p, index=False)
    od.to_csv(od_p, index=False)

    pack = prepare_all(gt_p, od_p, batch_size=64)
    print("[selftest] aligned rows:", len(pack["df_aligned"]))
    print("[selftest] splits:",
          len(pack["df_train"]), len(pack["df_val"]), len(pack["df_test"]))
    if _HAS_TORCH:
        batch = next(iter(pack["loaders"]["train"]))
        print("[selftest] batch x:", tuple(batch["x"].shape),
              "y:", tuple(batch["y"].shape),
              "stat:", tuple(batch["stationary"].shape))
    x_acc, y_acc, yaw_acc = accumulate_trajectory(
        pack["df_aligned"]["gt_d_x"].to_numpy(),
        pack["df_aligned"]["gt_d_y"].to_numpy(),
        pack["df_aligned"]["gt_d_yaw"].to_numpy(),
    )
    print("[selftest] traj len:", len(x_acc),
          f"final=({x_acc[-1]:.3f},{y_acc[-1]:.3f},{yaw_acc[-1]:.3f})")
    # cleanup
    for p in (gt_p, od_p):
        try: os.remove(p)
        except OSError: pass
    print("[selftest] OK")


if __name__ == "__main__":
    _selftest()
