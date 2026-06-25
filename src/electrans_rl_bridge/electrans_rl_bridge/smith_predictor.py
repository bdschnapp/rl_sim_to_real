#!/usr/bin/env python3
"""Smith Predictor wrapper for a delay-free RL policy on a delayed env.

Predictor-feedback architecture: at each control tick, copy the real
env's measured state into a shadow kinematic model and roll forward `d`
ticks using the pending (already-issued, not-yet-fully-effective)
commands. The shadow's predicted future state is then converted back
into a 32-dim lidar+state observation and handed to a policy that was
trained at τ=0. The policy sees a delay-free virtual env; the predictor
absorbs the latency.

Matches the user-provided pseudocode `SmithPredictorPolicy.predict_future_state`
in `smith_predictor_rl_trailer.py`. The kinematic model is the same
`StateSpaceTractorTrailer` the env uses; the shadow is a separate
instance whose per-instance `_steer_tau` / `_velocity_tau` are 0, so the
existing `patch_vehicle_with_delay`-patched `loop()` short-circuits to
instant actuation. Re-syncing to the measured state each tick is the
implicit residual feedback for this single-shadow design — the shadow
is anchored to ground truth and only rolls forward `d` steps before
being reset.
"""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Optional

import numpy as np


class SmithPredictor:
    """Wraps a (delay-injected) env so its `step()` returns the predicted
    delay-free observation `d` ticks in the future.

    Assumptions:
    - `env` is a reverse lab env (e.g. `ReverseLidarStateObservationLineFollowingEnv`)
      with a 32-dim flat Box observation (8-dim state + 24-dim lidar).
    - `v17_delay_aware.patch_vehicle_with_delay` has been applied globally
      (so the env's vehicle has the lag-capable patched `loop()`). The
      shadow uses τ=0 to bypass that lag.
    - The env exposes `.vehicle` (a `StateSpaceTractorTrailer`),
      `.xx, .yy` (centerline arrays), `.occ_grid` (the occupancy grid
      used for lidar ray-casting), and `.get_errors(x, y, p, xx, yy)`.
    """

    def __init__(
        self,
        env,
        *,
        steer_tau: float,
        velocity_tau: float,
        dt: float = 0.1,
        lidar_beams: int = 24,
        delay_steps: Optional[int] = None,
    ):
        self.env = env
        self.dt = float(dt)
        self.steer_tau = float(steer_tau)
        self.velocity_tau = float(velocity_tau)
        self.lidar_beams = int(lidar_beams)

        # `d` ticks of forward rollout — at least 1 even if τ rounds to 0,
        # since the actuator dynamics typically still take ~one tick to fully
        # respond. Computed from the larger of the two τ values.
        if delay_steps is not None:
            self.delay_steps = int(delay_steps)
        else:
            tau_max = max(self.steer_tau, self.velocity_tau)
            self.delay_steps = max(1, int(round(tau_max / self.dt)))

        # Build the shadow vehicle. Lazy-imported so this file doesn't
        # depend on e2e_rl being on sys.path at import time.
        from VehicleModels.tractor_trailer import StateSpaceTractorTrailer
        from e2erl_utils import config as c

        self.shadow = StateSpaceTractorTrailer(
            args=dict(c.tesla_model_s_vehicle_params),
            trailer_length=c.trailer_length_m,
        )
        # Force instance-level zero lag so the patched loop short-circuits
        # (`if self._steer_tau > 1e-6: ... else: self.s = self._s_target`).
        # If patch_vehicle_with_delay has NOT been applied (loop is the
        # unpatched original), these attributes are simply unused — no harm.
        self.shadow._steer_tau = 0.0
        self.shadow._velocity_tau = 0.0
        # The patched loop integrates action[0] into self._s_target then
        # bypasses lag → self.s = self._s_target. Start with both set to 0.
        self.shadow._s_target = 0.0
        self.shadow._xd_target = 0.0

        action_dim = int(env.action_space.shape[0])
        self._zero_action = np.zeros(action_dim, dtype=np.float32)
        self.pending_commands: deque = deque(
            [self._zero_action.copy() for _ in range(self.delay_steps)],
            maxlen=self.delay_steps,
        )

    # Pass through env attrs that the policy or eval driver may want
    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()

    # ----- core API -----
    def reset(self, *, seed=None, options=None):
        # Gym 0.26+ reset signature returns (obs, info).
        reset_kwargs = {}
        if seed is not None:
            reset_kwargs["seed"] = seed
        if options is not None:
            reset_kwargs["options"] = options
        real_obs, info = self.env.reset(**reset_kwargs)
        # Re-fill queue with zero commands.
        self.pending_commands.clear()
        for _ in range(self.delay_steps):
            self.pending_commands.append(self._zero_action.copy())
        # At t=0 the shadow's predicted state, after rolling zero actions
        # forward `d` ticks from the freshly-reset env state, is the same
        # as the env state. Skip the rollout and return real_obs directly.
        return real_obs, info

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.float32).flatten()
        real_obs, reward, term, trunc, info = self.env.step(action_arr)

        # Append new command to the queue (maxlen pops the oldest).
        self.pending_commands.append(action_arr.copy())

        # Sync shadow to current measured state.
        self._sync_shadow_from(self.env.vehicle)

        # Forward-roll the shadow `d` ticks through the pending commands.
        # `pending_commands` already contains exactly d entries (deque maxlen).
        for u in list(self.pending_commands):
            self.shadow.loop(u)

        # Reconstruct the observation the policy would have seen at
        # the predicted future state.
        obs_hat = self._reconstruct_obs(self.shadow)
        return obs_hat, reward, term, trunc, info

    # ----- internals -----
    def _sync_shadow_from(self, real_vehicle):
        """Copy every relevant attribute from the real env's vehicle into
        the shadow, so the rollout starts from ground truth."""
        # Tractor pose & velocity
        self.shadow.x = float(real_vehicle.x)
        self.shadow.y = float(real_vehicle.y)
        self.shadow.xd = float(real_vehicle.xd)
        self.shadow.yd = float(real_vehicle.yd)
        self.shadow.p = float(real_vehicle.p)
        self.shadow.pd = float(real_vehicle.pd)
        self.shadow.s = float(real_vehicle.s)
        # PID integrator state (longitudinal velocity tracking) — without
        # syncing these, the shadow's velocity-tracking PID drifts away
        # from the real env's behaviour over the rollout.
        if hasattr(real_vehicle, "i"):
            self.shadow.i = float(real_vehicle.i)
        if hasattr(real_vehicle, "e_prev"):
            self.shadow.e_prev = float(real_vehicle.e_prev)
        # Lag-state accumulators from the patch. Shadow has τ=0 so its
        # _s_target value drives self.s directly. We sync from real's
        # current _s_target if present; otherwise from real.s as a fallback.
        if hasattr(real_vehicle, "_s_target"):
            self.shadow._s_target = float(real_vehicle._s_target)
        else:
            self.shadow._s_target = float(real_vehicle.s)
        if hasattr(real_vehicle, "_xd_target"):
            self.shadow._xd_target = float(real_vehicle._xd_target)
        else:
            self.shadow._xd_target = float(real_vehicle.xd)
        # Trailer
        self.shadow.trailer.x = float(real_vehicle.trailer.x)
        self.shadow.trailer.y = float(real_vehicle.trailer.y)
        self.shadow.trailer.yaw = float(real_vehicle.trailer.yaw)
        if hasattr(real_vehicle.trailer, "yaw_rate"):
            self.shadow.trailer.yaw_rate = float(real_vehicle.trailer.yaw_rate)

    def _reconstruct_obs(self, vehicle):
        """Build the 32-dim observation a `ReverseLidarStateObservationLineFollowingEnv`
        would emit if its vehicle were at the predicted future state."""
        # Lazy import to avoid e2e_rl import-time coupling.
        from Environments.LineFollowing import compute_curvature
        from Environments.ObstacleAvoidance import Pose, get_obstacle_distances
        from e2erl_utils import config as c

        # --- state vector (8-dim) ---
        # Mirrors `_get_state_vector_obs()` of the reverse env (LineFollowing.py:833).
        s = float(vehicle.s)
        gamma = float(vehicle.p - vehicle.trailer.yaw)

        # Tractor cross-track + heading errors against the env's published
        # path (re-using env's helper).
        error, error_theta = self.env.get_errors(
            float(vehicle.x), float(vehicle.y), float(vehicle.p),
            self.env.xx, self.env.yy,
        )
        error_t, error_theta_t = self.env.get_errors(
            float(vehicle.trailer.x), float(vehicle.trailer.y), float(vehicle.trailer.yaw),
            self.env.xx, self.env.yy,
        )

        # Curvature lookaheads. `compute_curvature` only accesses
        # env.xx, env.yy, env.vehicle.trailer.x, env.vehicle.trailer.y.
        curve_shim = SimpleNamespace(
            xx=self.env.xx,
            yy=self.env.yy,
            vehicle=SimpleNamespace(
                trailer=SimpleNamespace(
                    x=float(vehicle.trailer.x), y=float(vehicle.trailer.y),
                ),
            ),
        )
        k1 = compute_curvature(curve_shim, lookahead_steps=10,
                               max_curvature=c.curvature_observation)
        k2 = compute_curvature(curve_shim, lookahead_steps=20,
                               max_curvature=c.curvature_observation)

        state_vec = np.array(
            [s, gamma, error, error_theta, error_t, error_theta_t, k1, k2],
            dtype=np.float32,
        )

        # --- lidar (24-dim) ---
        # Mirrors `ReverseLidarStateObservationLineFollowingEnv._get_obs()`:
        # lidar mounted on trailer, facing trailer.yaw + π.
        lidar_pose = Pose(
            x=float(vehicle.trailer.x),
            y=float(vehicle.trailer.y),
            yaw=float(vehicle.trailer.yaw) + np.pi,
        )
        lidar = get_obstacle_distances(
            self.env.occ_grid,
            lidar_pose,
            num_sensors=self.lidar_beams,
        )

        return np.concatenate(
            [state_vec, lidar.astype(np.float32)],
        ).astype(np.float32)


# ---------------------------------------------------------- ROS-side variant
def _wrap_pi(a: float) -> float:
    """Wrap angle to [-π, π)."""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


class SmithPredictorState:
    """Full Smith Predictor with explicit residual feedback for the ROS
    bridge. Maintains TWO shadow `StateSpaceTractorTrailer` instances:

    - `shadow_free`:    instant-actuator (τ=0) — represents the delay-free
                        trajectory the policy was trained on.
    - `shadow_delayed`: same lag as the real env — represents what ROS
                        *should* report if the model matched the plant.

    Both step forward with every commanded action; neither is re-synced.
    The instantaneous **residual** = (real_state − shadow_delayed) captures
    whatever the model can't predict (different RK4 integration in the
    C++ sim, slightly different trailer kinematic equation, ground
    contact effects, etc.). Adding that residual back to shadow_free
    gives the delay-free state corrected for model mismatch:

        corrected = shadow_free + (real − shadow_delayed)

    If models are exact (pygame case): residual ≡ 0 → corrected ≡ shadow_free.
    If models drift (ROS case): residual absorbs the drift so the policy
    still sees a meaningful "as-if-delay-free" state.

    Usage in bridge (per control tick, in order):

        # 1. AFTER env.step has run and the new ROS state is in self._ego:
        x_p, y_p, yaw_p, steer_p, xd_p, hitch_p = sp.step_and_predict(
            previous_action, ros_x, ros_y, ros_yaw, ros_steer, ros_xd, ros_hitch,
        )
        # 2. Feed predicted state to adapter:
        adapter.set_ego_state(x_p, y_p, yaw_p, steer_p, xd_p)
        adapter.set_trailer_state_from_hitch(hitch_p)
        obs = adapter.get_observation()
        next_action = policy.predict(obs)
        # 3. Bridge publishes next_action; loops back to step 1 next tick
        #    with next_action as `previous_action`.
    """

    def __init__(
        self,
        *,
        steer_tau: float,
        velocity_tau: float,
        dt: float = 0.1,
        delay_steps: Optional[int] = None,  # kept for launch-arg compat; unused
        action_dim: int = 2,
    ):
        self.dt = float(dt)
        self.steer_tau = float(steer_tau)
        self.velocity_tau = float(velocity_tau)
        # `delay_steps` retained for launch-file backward compatibility.
        # Dual-shadow architecture doesn't use a pending-action rollout
        # horizon — shadows evolve continuously alongside the real env.
        if delay_steps is not None:
            self.delay_steps = int(delay_steps)
        else:
            tau_max = max(self.steer_tau, self.velocity_tau)
            self.delay_steps = max(1, int(round(tau_max / self.dt)))

        from VehicleModels.tractor_trailer import StateSpaceTractorTrailer
        from e2erl_utils import config as c

        # shadow_free: delay-free (τ=0 short-circuits the patched loop).
        self.shadow_free = StateSpaceTractorTrailer(
            args=dict(c.tesla_model_s_vehicle_params),
            trailer_length=c.trailer_length_m,
        )
        self.shadow_free._steer_tau = 0.0
        self.shadow_free._velocity_tau = 0.0
        self.shadow_free._s_target = 0.0
        self.shadow_free._xd_target = 0.0

        # shadow_delayed: same lag as the real env. Patched loop uses the
        # exact-exponential discretisation, stable for any τ.
        self.shadow_delayed = StateSpaceTractorTrailer(
            args=dict(c.tesla_model_s_vehicle_params),
            trailer_length=c.trailer_length_m,
        )
        self.shadow_delayed._steer_tau = self.steer_tau
        self.shadow_delayed._velocity_tau = self.velocity_tau
        self.shadow_delayed._s_target = 0.0
        self.shadow_delayed._xd_target = 0.0

        self.action_dim = int(action_dim)
        self._initialized = False
        # Cached residual for debug logging.
        self._last_residual = None

    def initialize(self, x: float, y: float, yaw: float,
                   steer: float, xd: float, hitch_angle: float) -> None:
        """Seed both shadows from the initial ROS state. Called lazily on
        first `step_and_predict`, or explicitly via `reset()` after a
        sim state jump (e.g., /initialpose republish)."""
        for shadow in (self.shadow_free, self.shadow_delayed):
            self._set_shadow_state(shadow, x, y, yaw, steer, xd, hitch_angle)
        self._initialized = True
        self._last_residual = None

    def step_and_predict(
        self, action,
        x: float, y: float, yaw: float,
        steer: float, xd: float, hitch_angle: float,
    ) -> tuple:
        """Step both shadows forward with `action`, compute residual
        against ROS state, return the corrected delay-free prediction.

        Args are the post-step ROS state. Returns
        (x, y, yaw, steer, xd, hitch_angle) of the corrected state to
        feed the (delay-free-trained) policy."""
        if not self._initialized:
            self.initialize(x, y, yaw, steer, xd, hitch_angle)
            return float(x), float(y), float(yaw), float(steer), float(xd), float(hitch_angle)

        a = np.asarray(action, dtype=np.float32).flatten()
        # Step both shadows with the same commanded action. The patched
        # `loop()` applies first-order lag based on each shadow's
        # _steer_tau / _velocity_tau attributes.
        self.shadow_free.loop(a)
        self.shadow_delayed.loop(a)

        hitch_free = float(self.shadow_free.p - self.shadow_free.trailer.yaw)
        hitch_delayed = float(self.shadow_delayed.p - self.shadow_delayed.trailer.yaw)

        # Residual = real - shadow_delayed (model-error + disturbance term).
        r_x = float(x) - float(self.shadow_delayed.x)
        r_y = float(y) - float(self.shadow_delayed.y)
        r_yaw = _wrap_pi(float(yaw) - float(self.shadow_delayed.p))
        r_steer = float(steer) - float(self.shadow_delayed.s)
        r_xd = float(xd) - float(self.shadow_delayed.xd)
        r_hitch = _wrap_pi(float(hitch_angle) - hitch_delayed)
        self._last_residual = (r_x, r_y, r_yaw, r_steer, r_xd, r_hitch)

        # Corrected = shadow_free + residual.
        cx = float(self.shadow_free.x) + r_x
        cy = float(self.shadow_free.y) + r_y
        cyaw = _wrap_pi(float(self.shadow_free.p) + r_yaw)
        cs = float(self.shadow_free.s) + r_steer
        cxd = float(self.shadow_free.xd) + r_xd
        ch = _wrap_pi(hitch_free + r_hitch)
        return cx, cy, cyaw, cs, cxd, ch

    def reset(self, x: Optional[float] = None, y: Optional[float] = None,
              yaw: Optional[float] = None, steer: Optional[float] = None,
              xd: Optional[float] = None, hitch_angle: Optional[float] = None) -> None:
        """Re-seed both shadows. If all args are None, just marks the
        predictor as uninitialised — it'll lazily re-init on the next
        `step_and_predict` call using whatever state is provided then.
        Otherwise seeds immediately from the given state."""
        self._initialized = False
        self._last_residual = None
        if x is not None:
            self.initialize(x, y, yaw, steer, xd, hitch_angle)

    # ---- legacy API kept so the older bridge code-paths don't break -----
    def sync(self, *args, **kwargs):
        """No-op in the dual-shadow architecture. Shadows evolve via
        `step_and_predict` rather than being re-synced each tick."""
        return None

    def predict(self, *args, **kwargs):
        """No-op shim. Bridge code should call `step_and_predict`."""
        return None

    def push_action(self, *args, **kwargs):
        """No-op. The action is consumed inside `step_and_predict`."""
        return None

    # ---- internals -------------------------------------------------------
    def _set_shadow_state(self, shadow, x, y, yaw, steer, xd, hitch_angle):
        shadow.x = float(x)
        shadow.y = float(y)
        shadow.p = float(yaw)
        shadow.s = float(steer)
        shadow.xd = float(xd)
        shadow.yd = 0.0
        shadow.pd = 0.0
        shadow.xdd = 0.0
        shadow.ydd = 0.0
        shadow.pdd = 0.0
        if hasattr(shadow, "i"):
            shadow.i = 0.0
        if hasattr(shadow, "e_prev"):
            shadow.e_prev = 0.0
        shadow._s_target = float(steer)
        shadow._xd_target = float(xd)

        trailer_yaw = float(yaw) - float(hitch_angle)
        L_trailer = float(getattr(shadow.trailer, "L", 2.0))
        lr = float(getattr(shadow, "lr", 0.32))
        hitch_x = float(x) - lr * np.cos(float(yaw))
        hitch_y = float(y) - lr * np.sin(float(yaw))
        shadow.trailer.x = hitch_x - L_trailer * np.cos(trailer_yaw)
        shadow.trailer.y = hitch_y - L_trailer * np.sin(trailer_yaw)
        shadow.trailer.yaw = trailer_yaw
        if hasattr(shadow.trailer, "yaw_rate"):
            shadow.trailer.yaw_rate = 0.0
