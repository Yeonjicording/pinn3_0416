"""
model_module.py — Inverse-PINN for Scout Odometry Correction

Implements:
    class InversePINN(nn.Module)
        MLP backbone (SiLU, configurable depth/width, dropout) predicting
        [d_x_corr, d_yaw_corr] in original units, plus learnable physical
        coefficients (baseline b, wheel-radius scale s_r, and combined
        left/right wheel asymmetry scale alpha_sum) registered as
        nn.Parameter and constrained to physically sensible ranges via
        softplus. Note: (alpha_L, alpha_R) are individually unidentifiable
        from body-frame deltas alone (only their average appears in the
        yaw-rate equation), so the model uses a single alpha_sum parameter
        interpretable as the mean left/right wheel scale.

    class PhysicsLoss(nn.Module)
        Weighted sum of:
            L_data        : weighted MSE on (d_x, d_yaw)
            L_stationary  : zero-motion penalty gated by stationary mask
            L_nonholonomic: soft anchor toward physics-scaled prediction
                            (MLP output vs. coefficient-scaled odom residual)
            L_coeff       : L2 prior on coefficients around physical init
            L_magnitude   : small L2 on output magnitude

    class IdentityBaseline(nn.Module)
        pred = [d_x_odom, d_yaw_odom] (no correction).

    class LinearBaseline(nn.Module)
        per-axis affine calibration of (d_x_odom, d_yaw_odom).

    build_model(cfg: dict) -> nn.Module
    build_loss(cfg: dict) -> PhysicsLoss

The module is consumed by training-manager and evaluation-analyst.
See ``_workspace/02_model_design.md`` for design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants / index helpers for x_raw ordering (matches data_module.INPUT_COLS)
# INPUT_COLS = ["d_x", "d_y", "d_z", "d_rolling", "d_pitch", "d_yaw"]
# ---------------------------------------------------------------------------
IDX_DX = 0
IDX_DY = 1
IDX_DZ = 2
IDX_DROLL = 3
IDX_DPITCH = 4
IDX_DYAW = 5

# Physical priors
# ---------------------------------------------------------------------------
# B0_INIT: wheel track (left-right wheel centre separation) initial value.
#
#   Official URDF values from agilexrobotics/ugv_gazebo_sim:
#     Scout 2.0  → track = 0.583 m   (scout_v2.xacro L18)
#       https://github.com/agilexrobotics/ugv_gazebo_sim/blob/master/scout/scout_description/urdf/scout_v2.xacro
#     Scout Mini → track = 0.456 m   (scout_mini.xacro L20)
#       https://github.com/agilexrobotics/ugv_gazebo_sim/blob/master/scout/scout_description/urdf/scout_mini.xacro
#
#   NOTE: The current value 0.49 m matches the Scout 2.0 *wheelbase*
#   (front-to-rear axle distance = 0.498 m), NOT the wheel track.
#   For differential-drive kinematics, b should be the left-right
#   wheel separation (track). The model learns the true value via
#   inverse inference, so 0.49 serves only as a starting prior.
#
#   *** Adjust B0_INIT for your robot: ***
#     Scout 2.0  → B0_INIT = 0.583
#     Scout Mini → B0_INIT = 0.456
# ---------------------------------------------------------------------------
B0_INIT = 0.49       # baseline prior (m) — approximate starting value; see note above
B0_MIN = 0.20        # soft floor for baseline
SR_INIT = 1.0        # wheel radius scale
ALPHA_SUM_INIT = 2.0 # combined left+right wheel scale (alpha_L + alpha_R), prior ≈ 2.0


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    in_dim: int = 6
    out_dim: int = 2
    hidden_dim: int = 128
    hidden_blocks: int = 3       # number of internal (H->H) blocks
    dropout: float = 0.1
    activation: str = "silu"     # 'silu' | 'tanh' | 'gelu'


@dataclass
class LossConfig:
    w_data: float = 1.0
    w_stationary: float = 1.0
    w_nonholonomic: float = 0.1
    w_coeff: float = 0.01
    w_magnitude: float = 1e-3
    # per-axis data-loss weights (inverse variance of GT training targets).
    # If None, defaults to (1.0, 1.0).
    data_weights: Optional[tuple] = None  # (w_x, w_yaw)


# ---------------------------------------------------------------------------
# MLP backbone
# ---------------------------------------------------------------------------
def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "silu" or name == "swish":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class _MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        layers = [nn.Linear(cfg.in_dim, cfg.hidden_dim), _activation(cfg.activation)]
        for _ in range(cfg.hidden_blocks):
            layers += [
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                _activation(cfg.activation),
                nn.Dropout(cfg.dropout),
            ]
        layers += [nn.Linear(cfg.hidden_dim, cfg.out_dim)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                # Xavier works well with SiLU/tanh/GELU on bounded inputs.
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Inverse-PINN: MLP + learnable physical coefficients
# ---------------------------------------------------------------------------
class InversePINN(nn.Module):
    """Option-(A) architecture.

    The MLP predicts ``[d_x_corr, d_yaw_corr]`` directly in original units
    (meters, radians). Separately, three scalar nn.Parameter values encode the
    system coefficients to be inverse-inferred. They are reparameterized via
    softplus so that:
        b         = B0_MIN + softplus(b_raw)      in (B0_MIN, +inf)
        s_r       = softplus(s_r_raw)             in (0, +inf)
        alpha_sum = softplus(asum_raw)            in (0, +inf)  (prior ≈ 2.0)

    ``alpha_sum`` represents (alpha_L + alpha_R), i.e. the combined
    left+right wheel scale. The individual asymmetry (alpha_R − alpha_L)
    is not identifiable from body-frame (d_x, d_yaw) observations alone
    because only the average enters the yaw-rate equation in this data
    regime (no raw per-wheel velocities). Report alpha_sum / 2 as the
    "mean left/right wheel scale".

    Raw parameters are initialized so that effective values match priors.
    """

    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.mlp = _MLP(self.cfg)

        # softplus inverse: given target value v, raw = log(exp(v) - 1)
        def inv_softplus(v: float) -> float:
            # clamp to avoid overflow for very small v
            import math
            v = float(v)
            return math.log(math.expm1(v)) if v > 1e-6 else math.log(1e-6)

        b_raw_init = inv_softplus(B0_INIT - B0_MIN)   # so that B0_MIN + softplus(raw) = B0_INIT
        s_raw_init = inv_softplus(SR_INIT)
        asum_raw_init = inv_softplus(ALPHA_SUM_INIT)

        self.b_raw = nn.Parameter(torch.tensor(float(b_raw_init)))
        self.s_r_raw = nn.Parameter(torch.tensor(float(s_raw_init)))
        self.asum_raw = nn.Parameter(torch.tensor(float(asum_raw_init)))

    # ---- parameter accessors (apply softplus constraints) -----------------
    @property
    def b(self) -> torch.Tensor:
        return B0_MIN + F.softplus(self.b_raw)

    @property
    def s_r(self) -> torch.Tensor:
        return F.softplus(self.s_r_raw)

    @property
    def alpha_sum(self) -> torch.Tensor:
        """Combined (alpha_L + alpha_R); only this average is identifiable."""
        return F.softplus(self.asum_raw)

    def coefficients(self) -> Dict[str, torch.Tensor]:
        return {
            "b": self.b,
            "s_r": self.s_r,
            "alpha_sum": self.alpha_sum,
        }

    def coefficient_values(self) -> Dict[str, float]:
        with torch.no_grad():
            return {k: float(v.detach().cpu()) for k, v in self.coefficients().items()}

    def mlp_parameters(self):
        return self.mlp.parameters()

    def coeff_parameters(self):
        return [self.b_raw, self.s_r_raw, self.asum_raw]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predictions of shape (B, 2) = [d_x_corr, d_yaw_corr]."""
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
class IdentityBaseline(nn.Module):
    """pred = [d_x_odom, d_yaw_odom]. No learnable params."""

    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        # keep a dummy param so optim can be built uniformly
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward_from_raw(self, x_raw: torch.Tensor) -> torch.Tensor:
        dx = x_raw[:, IDX_DX:IDX_DX + 1]
        dyaw = x_raw[:, IDX_DYAW:IDX_DYAW + 1]
        return torch.cat([dx, dyaw], dim=-1)

    def forward(self, x: torch.Tensor, x_raw: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x_raw is None:
            raise RuntimeError("IdentityBaseline requires x_raw via forward(x, x_raw=...)")
        return self.forward_from_raw(x_raw)


class LinearBaseline(nn.Module):
    """Per-axis affine calibration on (d_x_odom, d_yaw_odom).

        p_dx   = a1 * dx_od   + b1
        p_dyaw = a2 * dyaw_od + b2
    """

    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.a = nn.Parameter(torch.ones(2))
        self.b = nn.Parameter(torch.zeros(2))

    def forward(self, x: torch.Tensor, x_raw: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x_raw is None:
            raise RuntimeError("LinearBaseline requires x_raw via forward(x, x_raw=...)")
        dx = x_raw[:, IDX_DX]
        dyaw = x_raw[:, IDX_DYAW]
        p_dx = self.a[0] * dx + self.b[0]
        p_dyaw = self.a[1] * dyaw + self.b[1]
        return torch.stack([p_dx, p_dyaw], dim=-1)


# ---------------------------------------------------------------------------
# Physics-informed loss
# ---------------------------------------------------------------------------
class PhysicsLoss(nn.Module):
    """Aggregate physics-informed loss. See ``02_model_design.md`` section 'Loss'.

    forward(pred, target, x_raw, stationary, coeffs) -> (loss_total, parts_dict)
        pred       : (B, 2) [p_dx, p_dyaw]  (original units)
        target     : (B, 2) [gt_dx, gt_dyaw]
        x_raw      : (B, 6) raw odom (original units)
        stationary : (B,)   bool
        coeffs     : dict with keys {b, s_r, alpha_L, alpha_R} (tensors)
                     May be None for baselines -> physics/coeff terms become 0.
    """

    def __init__(self, cfg: Optional[LossConfig] = None):
        super().__init__()
        self.cfg = cfg or LossConfig()
        if self.cfg.data_weights is None:
            w_x, w_yaw = 1.0, 1.0
        else:
            w_x, w_yaw = self.cfg.data_weights
        self.register_buffer(
            "w_axis", torch.tensor([float(w_x), float(w_yaw)], dtype=torch.float32)
        )
        # store priors as buffers for coeff regularization
        self.register_buffer("b_prior", torch.tensor(B0_INIT, dtype=torch.float32))
        self.register_buffer("sr_prior", torch.tensor(SR_INIT, dtype=torch.float32))
        self.register_buffer("alpha_sum_prior", torch.tensor(ALPHA_SUM_INIT, dtype=torch.float32))

    # ------------------------------------------------------------------
    def _loss_data(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff2 = (pred - target) ** 2
        # broadcast axis weights
        w = self.w_axis.to(diff2.device).view(1, -1)
        return (diff2 * w).mean()

    def _loss_stationary(self, pred: torch.Tensor, stationary: torch.Tensor) -> torch.Tensor:
        if stationary is None:
            return pred.new_zeros(())
        S = stationary.float()
        denom = S.mean().clamp_min(1e-6)
        per = (pred ** 2).sum(dim=-1) * S
        return per.mean() / denom

    def _loss_nonholonomic(
        self,
        pred: torch.Tensor,
        x_raw: torch.Tensor,
        stationary: torch.Tensor,
        coeffs: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if coeffs is None:
            return pred.new_zeros(())
        dx_od = x_raw[:, IDX_DX]
        dyaw_od = x_raw[:, IDX_DYAW]

        b = coeffs["b"]
        s_r = coeffs["s_r"]
        alpha_sum = coeffs["alpha_sum"]  # (alpha_L + alpha_R)

        phys_dx = s_r * dx_od
        # simplified first-order model; baseline ratio (b_0 / b) inversely scales dyaw.
        # alpha_sum / 2 is the mean left/right wheel scale.
        phys_dyaw = s_r * dyaw_od * (alpha_sum * 0.5) * (B0_INIT / b.clamp_min(1e-3))

        p_dx = pred[:, 0]
        p_dyaw = pred[:, 1]
        r_x = p_dx - phys_dx
        r_yaw = p_dyaw - phys_dyaw

        # Physics-anchor only; lateral (r_lat = dy_od) term removed — it had
        # no gradient path to any learnable parameter (input-only tensor),
        # so it was ineffective as a loss term. Non-holonomic prior is
        # encoded instead by the model predicting only (d_x, d_yaw) with
        # d_y implicitly zero at trajectory accumulation time.
        l_anchor = (r_x ** 2 + r_yaw ** 2).mean()
        return l_anchor

    def _loss_coeff(self, coeffs: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        if coeffs is None:
            return self.b_prior.new_zeros(())
        return (
            (coeffs["b"] - self.b_prior) ** 2
            + (coeffs["s_r"] - self.sr_prior) ** 2
            + (coeffs["alpha_sum"] - self.alpha_sum_prior) ** 2
        )

    def _loss_magnitude(self, pred: torch.Tensor) -> torch.Tensor:
        return (pred ** 2).mean()

    # ------------------------------------------------------------------
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        x_raw: torch.Tensor,
        stationary: Optional[torch.Tensor] = None,
        coeffs: Optional[Dict[str, torch.Tensor]] = None,
    ):
        L_data = self._loss_data(pred, target)
        L_stat = self._loss_stationary(pred, stationary) if stationary is not None else pred.new_zeros(())
        L_nh = self._loss_nonholonomic(pred, x_raw, stationary, coeffs)
        L_coeff = self._loss_coeff(coeffs)
        L_mag = self._loss_magnitude(pred)

        c = self.cfg
        total = (
            c.w_data * L_data
            + c.w_stationary * L_stat
            + c.w_nonholonomic * L_nh
            + c.w_coeff * L_coeff
            + c.w_magnitude * L_mag
        )
        parts = {
            "total": total.detach(),
            "data": L_data.detach(),
            "stationary": L_stat.detach(),
            "nonholonomic": L_nh.detach(),
            "coeff": L_coeff.detach(),
            "magnitude": L_mag.detach(),
        }
        return total, parts


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def build_model(cfg: Optional[dict] = None) -> nn.Module:
    """Build a model from a config dict.

    cfg keys:
        name : 'pinn' (default) | 'identity' | 'linear'
        model: ModelConfig kwargs
    """
    cfg = cfg or {}
    name = cfg.get("name", "pinn").lower()
    mcfg = ModelConfig(**(cfg.get("model") or {}))
    if name == "pinn":
        return InversePINN(mcfg)
    if name == "identity":
        return IdentityBaseline(mcfg)
    if name == "linear":
        return LinearBaseline(mcfg)
    raise ValueError(f"Unknown model name: {name}")


def build_loss(cfg: Optional[dict] = None) -> PhysicsLoss:
    cfg = cfg or {}
    lcfg = LossConfig(**(cfg.get("loss") or {}))
    return PhysicsLoss(lcfg)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _smoke_test() -> None:
    torch.manual_seed(0)
    B = 32
    x = torch.randn(B, 6)
    x_raw = torch.randn(B, 6) * 0.05
    # make a few stationary
    x_raw[:5] = 0.0
    stationary = torch.zeros(B, dtype=torch.bool)
    stationary[:5] = True
    # synthetic GT targets (d_x ~ small positive, d_yaw ~ small)
    target = torch.stack([x_raw[:, IDX_DX] * 0.98, x_raw[:, IDX_DYAW] * 1.02], dim=-1)

    # ----- Inverse-PINN -----
    model = build_model({"name": "pinn", "model": {"hidden_dim": 64, "hidden_blocks": 2, "dropout": 0.1}})
    loss_fn = build_loss({"loss": {"data_weights": (1.0, 10.0)}})

    assert sum(p.numel() for p in model.parameters()) > 0
    n_coeffs = sum(p.numel() for p in model.coeff_parameters())
    n_mlp = sum(p.numel() for p in model.mlp_parameters())
    print(f"[smoke] InversePINN params: mlp={n_mlp}, coeffs={n_coeffs}")
    print(f"[smoke] init coeffs: {model.coefficient_values()}")

    pred = model(x)
    assert pred.shape == (B, 2), pred.shape

    total, parts = loss_fn(pred, target, x_raw, stationary, model.coefficients())
    print("[smoke] loss parts:", {k: float(v) for k, v in parts.items()})

    # backward works
    total.backward()
    grad_ok = all(p.grad is not None for p in model.coeff_parameters())
    assert grad_ok, "coeff parameters did not receive gradients"
    print("[smoke] backward OK; coeff grads present")

    # ----- IdentityBaseline -----
    id_model = build_model({"name": "identity"})
    id_pred = id_model(x, x_raw=x_raw)
    assert id_pred.shape == (B, 2)
    total_id, _ = loss_fn(id_pred, target, x_raw, stationary, coeffs=None)
    print(f"[smoke] IdentityBaseline loss={float(total_id):.6f}")

    # ----- LinearBaseline -----
    lin_model = build_model({"name": "linear"})
    lin_pred = lin_model(x, x_raw=x_raw)
    total_lin, _ = loss_fn(lin_pred, target, x_raw, stationary, coeffs=None)
    total_lin.backward()
    print(f"[smoke] LinearBaseline loss={float(total_lin):.6f}, a.grad={lin_model.a.grad.tolist()}")

    # ----- Softplus constraint sanity -----
    # baseline stays above B0_MIN
    assert float(model.b) > B0_MIN - 1e-6
    assert float(model.s_r) > 0
    assert float(model.alpha_sum) > 0
    print("[smoke] OK")


if __name__ == "__main__":
    _smoke_test()
