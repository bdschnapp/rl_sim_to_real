"""Learned dynamics model for the tractor-trailer rig — a drop-in
replacement for e2e_rl's `StateSpaceTractorTrailer`.

Two pieces live here:

1.  `DynamicsMLP` — small `nn.Module` that predicts a body-frame state
    delta from (current body-frame state, action history window).

2.  `NeuralDynamicsTractorTrailer` — wrapper class mirroring the
    `StateSpaceTractorTrailer` external API (`.loop(action)`,
    `.reset(...)`, plus all of the state attributes the bridge reads).
    Internally it maintains an action-history deque and the NN forward
    pass; world-frame pose is integrated outside the network so the
    network only has to learn the small, translation-invariant
    body-frame dynamics.

Why body-frame deltas:
  - Translation- and rotation-invariant — the model learns physics
    rather than memorising map coordinates.
  - Easier to train (target magnitudes are small and well-scaled).
  - Robust to OOD inputs — a confused network outputs near-zero, which
    means "no change" rather than "teleport".

Action history window:
  - The model sees the last N+1 commands (default N=10 → 1 s at 10 Hz).
  - Lets the network learn arbitrary input lag / actuator dynamics
    without needing internal delay buffers.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Dimensions / conventions
# ----------------------------------------------------------------------------

# Body-frame state (8-dim) — what the NN sees as input/output:
#   [0] xd          longitudinal velocity (body x), m/s
#   [1] yd          lateral velocity (body y), m/s
#   [2] pd          yaw rate of tractor, rad/s
#   [3] s           steering tire angle (state), rad
#   [4] dx_t        trailer axle x relative to truck, body frame, m
#   [5] dy_t        trailer axle y relative to truck, body frame, m
#   [6] beta        hitch angle (truck_yaw − trailer_yaw), rad
#   [7] t_yaw_rate  trailer yaw rate, rad/s
STATE_DIM = 8

# Action (2-dim) — what the bridge sends as a command each tick:
#   [0] steering_tire_angle_cmd, rad
#   [1] velocity_cmd, m/s
ACTION_DIM = 2


# ----------------------------------------------------------------------------
# DynamicsMLP
# ----------------------------------------------------------------------------


class DynamicsMLP(nn.Module):
    """Predicts the 8-dim body-frame state delta.

    Input  → output:
      (state[B,8], action_history[B, N+1, 2]) → delta[B, 8]

    The action history is flattened internally; we keep the (N+1, 2)
    shape externally so callers don't have to track the flatten ordering.
    """

    def __init__(self, action_window: int = 11, hidden: int = 128):
        super().__init__()
        self.action_window = int(action_window)
        self.hidden = int(hidden)
        in_dim = STATE_DIM + ACTION_DIM * self.action_window
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, STATE_DIM),
        )

    def forward(self, state: torch.Tensor, action_history: torch.Tensor) -> torch.Tensor:
        if action_history.dim() == 3:
            # (B, N+1, 2) → (B, (N+1)*2)
            action_history = action_history.flatten(1)
        x = torch.cat([state, action_history], dim=-1)
        return self.net(x)

    # ---- checkpoint I/O -----------------------------------------------------

    def save_checkpoint(self, path: str, meta: Optional[dict] = None):
        blob = {
            "state_dict": self.state_dict(),
            "meta": {
                "action_window": self.action_window,
                "hidden": self.hidden,
                **(meta or {}),
            },
        }
        torch.save(blob, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: Optional[str] = None) -> "DynamicsMLP":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(blob, dict) and "state_dict" in blob:
            meta = blob.get("meta", {})
            sd = blob["state_dict"]
        else:
            # bare state_dict — fall back to default kwargs
            meta = {}
            sd = blob
        model = cls(
            action_window=int(meta.get("action_window", 11)),
            hidden=int(meta.get("hidden", 128)),
        )
        model.load_state_dict(sd)
        model.eval()
        if device:
            model.to(device)
        return model


# ----------------------------------------------------------------------------
# Drop-in wrapper matching StateSpaceTractorTrailer
# ----------------------------------------------------------------------------


def _wrap_pi(angle: float) -> float:
    """Wrap an angle to (-π, π]."""
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


class _Trailer:
    """Simple attribute container matching `TrailerModel`'s external API.

    Stores trailer axle position, yaw, yaw rate, and the hitch-to-axle
    distance `L`. The NN wrapper class writes these on every `.loop()`
    call so downstream code that reads `vehicle.trailer.x` etc. keeps
    working unchanged.
    """

    def __init__(self, length: float):
        self.L = float(length)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.yaw_rate = 0.0


class NeuralDynamicsTractorTrailer:
    """Drop-in replacement for `StateSpaceTractorTrailer`.

    Same constructor signature (accepts `args=dict(...)` like the e2e_rl
    one does), same `.loop(action)` returning an 11-dim numpy array,
    same `.reset(xd, x=0, y=0, p=0)`, same instance attributes that the
    Smith Predictor and the e2e_rl envs read.

    Internally:
      * Maintains a `(action_window, 2)` history buffer, FIFO updated on
        every `.loop()` call.
      * Computes the body-frame state, runs `DynamicsMLP.forward()` once,
        decodes the predicted delta back into world-frame state.
      * Integrates world-frame pose using semi-implicit Euler with the
        post-step body-frame velocities — keeps `(x, y, p)` consistent
        with `(xd, yd, pd)` for any downstream consumer that compares
        them.
    """

    def __init__(
        self,
        dynamics_net: DynamicsMLP,
        args: Optional[dict] = None,
        trailer_length: float = 2.0,
        device: Optional[str] = None,
    ):
        # Accept the e2e_rl-style args dict, but fall back to sensible
        # defaults for the lab vehicle if not provided.
        args = dict(args or {})
        self.dt = float(args.get("dt", 0.1))
        self.lf = float(args.get("lf", 0.33))
        self.lr = float(args.get("lr", 0.32))
        # `trailer_length` is passed as a separate kwarg in e2e_rl too;
        # we accept either spelling.
        L = float(args.get("trailer_length_m", trailer_length))
        self.trailer = _Trailer(length=L)

        # Network
        self._model = dynamics_net
        self._model.eval()
        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model.to(self._device)

        self.action_window = int(dynamics_net.action_window)
        # Pre-allocate history buffer and torch tensors to avoid per-step
        # allocations on the hot path.
        self._u_history = np.zeros((self.action_window, ACTION_DIM), dtype=np.float32)
        self._state_buf = np.zeros(STATE_DIM, dtype=np.float32)

        # ---- e2e_rl-compatible attributes ---------------------------------
        # Tractor
        self.x = 0.0
        self.y = 0.0
        self.p = 0.0
        self.xd = 0.0
        self.yd = 0.0
        self.pd = 0.0
        self.s = 0.0
        # PID accumulators (e2e_rl maintains these; we just expose them
        # so the bridge's `_sync_shadow_from` can read/write them without
        # raising AttributeError).
        self.i = 0.0
        self.e_prev = 0.0
        # Delay-target accumulators (set by the v17_delay_aware patched
        # loop on the e2e_rl model; we expose them so the bridge's
        # state-sync code doesn't crash. Delay is now learned by the NN,
        # so these aren't actually used by the inner dynamics.)
        self._s_target = 0.0
        self._xd_target = 0.0
        # Time-constant attributes the patched loop branches on. Setting
        # them to 0 ensures any external code that calls the patched-loop
        # path takes the τ=0 short-circuit branch.
        self._steer_tau = 0.0
        self._velocity_tau = 0.0

    # ------------------------------------------------------------------- API

    def reset(self, xd: float = 0.0, x: float = 0.0, y: float = 0.0, p: float = 0.0):
        """Re-initialise the truck pose and clear all dynamic state.

        Matches `StateSpaceTractorTrailer.reset()`: tractor at `(x, y, p)`
        with longitudinal velocity `xd`, trailer aligned with tractor
        (β = 0), all yaw rates / accelerations zero.
        """
        self.x = float(x)
        self.y = float(y)
        self.p = float(p)
        self.xd = float(xd)
        self.yd = 0.0
        self.pd = 0.0
        self.s = 0.0
        self.i = 0.0
        self.e_prev = 0.0
        self._s_target = 0.0
        self._xd_target = float(xd)

        # Trailer: aligned with tractor (β = 0). Axle is L behind the
        # hitch in tractor-yaw direction; hitch is `lr` behind tractor
        # CG (matches e2e_rl's TrailerModel.reset convention).
        self.trailer.yaw = float(p)
        cos_p = math.cos(self.p)
        sin_p = math.sin(self.p)
        hitch_x = self.x - self.lr * cos_p
        hitch_y = self.y - self.lr * sin_p
        self.trailer.x = hitch_x - self.trailer.L * math.cos(self.trailer.yaw)
        self.trailer.y = hitch_y - self.trailer.L * math.sin(self.trailer.yaw)
        self.trailer.yaw_rate = 0.0

        # Clear action history.
        self._u_history.fill(0.0)

    def loop(self, action) -> np.ndarray:
        """One dynamics step. Returns the 11-dim concatenated state
        array matching `StateSpaceTractorTrailer.loop()`:

            [x, y, xd, yd, p, pd, s, tx, ty, t_yaw, t_yaw_rate]
        """
        # 1. Push new action onto the history (FIFO).
        u = np.asarray(action, dtype=np.float32).flatten()
        if u.size < ACTION_DIM:
            u = np.pad(u, (0, ACTION_DIM - u.size))
        else:
            u = u[:ACTION_DIM]
        self._u_history[:-1] = self._u_history[1:]
        self._u_history[-1] = u

        # 2. Build body-frame state vector.
        cos_p = math.cos(self.p)
        sin_p = math.sin(self.p)
        dx_w = self.trailer.x - self.x
        dy_w = self.trailer.y - self.y
        dx_t = cos_p * dx_w + sin_p * dy_w
        dy_t = -sin_p * dx_w + cos_p * dy_w
        beta = _wrap_pi(self.p - self.trailer.yaw)

        self._state_buf[:] = (
            self.xd,
            self.yd,
            self.pd,
            self.s,
            dx_t,
            dy_t,
            beta,
            self.trailer.yaw_rate,
        )

        # 3. Forward pass.
        with torch.no_grad():
            s_t = torch.from_numpy(self._state_buf).unsqueeze(0).to(self._device)
            u_t = torch.from_numpy(self._u_history).unsqueeze(0).to(self._device)
            delta = self._model(s_t, u_t).squeeze(0).cpu().numpy()

        # 4. Decode delta — apply to body-frame state.
        d_xd, d_yd, d_pd, d_s, d_dx_t, d_dy_t, d_beta, d_t_yaw_rate = delta

        new_xd = float(self._state_buf[0] + d_xd)
        new_yd = float(self._state_buf[1] + d_yd)
        new_pd = float(self._state_buf[2] + d_pd)
        new_s = float(np.clip(self._state_buf[3] + d_s, -math.pi / 4, math.pi / 4))
        new_dx_t = float(self._state_buf[4] + d_dx_t)
        new_dy_t = float(self._state_buf[5] + d_dy_t)
        new_beta = _wrap_pi(self._state_buf[6] + d_beta)
        new_t_yaw_rate = float(self._state_buf[7] + d_t_yaw_rate)

        # 5. Integrate world-frame pose with the new velocities
        # (semi-implicit Euler). Body-frame (xd, yd) maps to world via
        # the current truck yaw.
        self.x = float(self.x + self.dt * (new_xd * cos_p - new_yd * sin_p))
        self.y = float(self.y + self.dt * (new_xd * sin_p + new_yd * cos_p))
        self.p = _wrap_pi(self.p + self.dt * new_pd)

        # 6. Commit scalar state.
        self.xd = new_xd
        self.yd = new_yd
        self.pd = new_pd
        self.s = new_s

        # 7. Reconstruct trailer pose in world frame.
        cos_p_new = math.cos(self.p)
        sin_p_new = math.sin(self.p)
        self.trailer.x = float(self.x + cos_p_new * new_dx_t - sin_p_new * new_dy_t)
        self.trailer.y = float(self.y + sin_p_new * new_dx_t + cos_p_new * new_dy_t)
        self.trailer.yaw = _wrap_pi(self.p - new_beta)
        self.trailer.yaw_rate = new_t_yaw_rate

        return np.array(
            [
                self.x, self.y, self.xd, self.yd, self.p, self.pd, self.s,
                self.trailer.x, self.trailer.y, self.trailer.yaw, self.trailer.yaw_rate,
            ],
            dtype=np.float32,
        )

    # -------------------------------------------------------- factory helper

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        args: Optional[dict] = None,
        trailer_length: float = 2.0,
        device: Optional[str] = None,
    ) -> "NeuralDynamicsTractorTrailer":
        """Convenience constructor: load `DynamicsMLP` weights then
        return a drop-in wrapper around them.
        """
        model = DynamicsMLP.load_checkpoint(checkpoint_path, device=device)
        return cls(
            dynamics_net=model,
            args=args,
            trailer_length=trailer_length,
            device=device,
        )
