#!/usr/bin/env python3
"""Monkey-patch for a CONSTANT-SPEED + STOP-SIGNAL action mode.

This is an alternative to ``train_lab_model._patch_variable_speed_action``.
Where the variable-speed patch makes the policy choose its own velocity, this
patch keeps the vehicle at a CONSTANT speed (``fixed_speed=True`` is left
untouched) and adds a second action component that lets the policy *decide to
stop*.

Action = 2-D ``[steer_rate, stop_signal]`` with ``stop_signal ∈ [-1, 1]``:

  * ``stop_signal > threshold`` (default 0.0)  → the agent has chosen to STOP.
    We zero the vehicle speed, terminate the episode, and return a heavy
    negative reward. In these regular, no-obstacle training envs a stop is
    ALWAYS the wrong choice, so a stop is ALWAYS penalised — this teaches the
    policy to keep driving and only later (in obstacle envs) learn when a stop
    is justified.

  * otherwise → drive normally at constant speed, using ``action[0]`` as the
    steering rate.

Why this works WITHOUT touching e2e_rl
--------------------------------------
``LineFollowingEnv.format_action`` (LineFollowing.py ~L398), under
``fixed_speed=True``, returns ``[action[0], fixed_speed_command]`` and DROPS
``action[1]``. So if we keep ``fixed_speed=True`` and merely widen the
``action_space`` to 2-D, the vehicle already ignores ``action[1]`` and runs at
the constant speed. We therefore only need to:

  (a) set a 2-D ``action_space`` after the original ``__init__``, and
  (b) wrap ``step`` to read ``action[1]`` and handle the stop, otherwise
      delegate to the ORIGINAL step with the full action (whose
      ``format_action`` drops ``action[1]``).

Both base lidar classes are patched:
  * forward: ``Environments.ObstacleAvoidance.LidarStateObservationLineFollowingEnv``
  * reverse: ``Environments.LineFollowing.ReverseLidarStateObservationLineFollowingEnv``

The tractor-only envs subclass these bases and their ``TractorOnlyMixin.step``
calls ``super().step(action)`` passing the full action through, so the
stop-intercept on the base ``step`` fires for tractor-only too.

The patch is idempotent: it stashes the originals on the class the first time
it runs and re-wraps from those, so applying it twice does not double-wrap.
"""

from __future__ import annotations

from pathlib import Path
import sys


# Sentinel attribute names used to stash the unwrapped originals on each class
# so re-patching is idempotent.
_ORIG_INIT_ATTR = "_stop_signal_orig_init"
_ORIG_STEP_ATTR = "_stop_signal_orig_step"


def _config_steering_action_deg() -> float:
    """Max steering rate in DEGREES, from e2e_rl config. Mirrors
    ``train_lab_model._config_steering_action`` (falls back to 25.0)."""
    try:
        from e2erl_utils import config as c
        return float(c.steering_action)
    except Exception:
        return 25.0


def _wrap_class(env_cls, *, threshold: float, stop_penalty: float,
                drive_penalty: float) -> None:
    """Wrap ``__init__`` (widen action_space to 2-D) and ``step`` (stop
    intercept) on a single base lidar env class. Idempotent."""
    import numpy as np
    from gymnasium import spaces

    # Stash the unwrapped originals exactly once (idempotency). We bind them
    # to THIS class (not an inherited attr) so each base class keeps its own.
    if _ORIG_INIT_ATTR not in env_cls.__dict__:
        setattr(env_cls, _ORIG_INIT_ATTR, env_cls.__init__)
    if _ORIG_STEP_ATTR not in env_cls.__dict__:
        # step may be inherited (LineFollowingEnv.step); grab the resolved one.
        setattr(env_cls, _ORIG_STEP_ATTR, env_cls.step)

    orig_init = getattr(env_cls, _ORIG_INIT_ATTR)
    orig_step = getattr(env_cls, _ORIG_STEP_ATTR)

    def stop_signal_init(self, *args, **kwargs):
        # Call the original __init__ untouched — fixed_speed stays True
        # (forward base has no fixed_speed kwarg; reverse defaults to True and
        # we never override it), so format_action keeps using a constant speed.
        orig_init(self, *args, **kwargs)
        max_steer_rate = np.deg2rad(_config_steering_action_deg())
        self.action_space = spaces.Box(
            low=np.array([-max_steer_rate, -1.0], dtype=np.float32),
            high=np.array([max_steer_rate, +1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def stop_signal_step(self, action):
        a = np.asarray(action, dtype=np.float32).flatten()
        stop = float(a[1]) if a.size > 1 else -1.0
        if stop > threshold:
            # Agent chose to STOP. Zero the speed, build a terminal obs/info,
            # and return a heavy negative reward with terminated=True. In these
            # no-obstacle envs a stop is always wrong, so always penalise.
            self.vehicle.xd = 0.0
            obs = self._get_obs()
            info = self._get_info()
            return obs, -abs(stop_penalty), True, False, info
        # Drive normally: delegate to the ORIGINAL step with the full action.
        # The original format_action drops action[1] under fixed_speed=True and
        # uses the constant fixed_speed_command for velocity.
        obs, reward, term, trunc, info = orig_step(self, action)
        # Smooth downward gradient on the stop dim: with `threshold` high enough
        # that exploration noise rarely fires a hard stop (so episodes survive
        # and the policy can learn the value of DRIVING), the actor would
        # otherwise get no signal on the stop dim and its output could drift up
        # toward the threshold. A small per-step penalty on POSITIVE stop intent
        # gives a reliable gradient pushing the stop output <= 0 (i.e. "keep
        # driving"), without terminating the episode. Only bites for stop > 0,
        # so it never interferes with a confidently-driving (stop < 0) policy.
        if drive_penalty and stop > 0.0:
            reward = reward - drive_penalty * stop
        return obs, reward, term, trunc, info

    env_cls.__init__ = stop_signal_init
    env_cls.step = stop_signal_step


def patch_stop_signal_action(
    e2e_rl_path,
    threshold: float = 0.5,
    stop_penalty: float = 200.0,
    stop_noise: float = 0.1,
    drive_penalty: float = 0.5,
) -> None:
    """Apply the constant-speed + stop-signal action patch to BOTH base lidar
    env classes (forward + reverse).

    Parameters
    ----------
    e2e_rl_path : path-like
        Filesystem path to the e2e_rl checkout (added to ``sys.path``).
    threshold : float
        ``stop_signal`` above this value triggers a hard stop (default 0.5).
        Set well above 0 so that, with a zero-centred tanh stop output and the
        small ``stop_noise`` below, exploration noise almost never crosses it —
        episodes then survive to full length and the policy can learn the value
        of DRIVING. (With the previous threshold=0.0, ~50% of exploration steps
        crossed it, terminating episodes in ~2 steps, so the policy never
        experienced a full route and collapsed into a "drive a few steps then
        stop" local optimum.) The DEPLOY bridge must use the SAME threshold.
    stop_penalty : float
        Magnitude of the negative reward returned on a hard stop (default 200).
        The returned reward is ``-abs(stop_penalty)``.
    drive_penalty : float
        Per-step penalty coefficient on POSITIVE stop intent while driving
        (reward -= drive_penalty * max(0, stop)). Gives the actor a reliable
        downward gradient on the stop dim so its output stays <= 0 even though
        hard stops are rare at the raised threshold. Default 0.5.
    """
    e2e_rl_path = Path(e2e_rl_path)
    if str(e2e_rl_path) not in sys.path:
        sys.path.insert(0, str(e2e_rl_path))

    import Environments.ObstacleAvoidance as oa
    import Environments.LineFollowing as lf

    _wrap_class(
        oa.LidarStateObservationLineFollowingEnv,
        threshold=threshold,
        stop_penalty=stop_penalty,
        drive_penalty=drive_penalty,
    )
    _wrap_class(
        lf.ReverseLidarStateObservationLineFollowingEnv,
        threshold=threshold,
        stop_penalty=stop_penalty,
        drive_penalty=drive_penalty,
    )

    # Reduce exploration noise on the stop dimension. make_action_noise_sigma
    # defaults sigma[1:]=0.5 (sized for variable-speed velocity), which is far
    # too much for a [-1,1] stop_signal at threshold 0 — exploration would cross
    # it constantly → near-constant random stops/terminations that destabilise
    # training. Patch the module function so the stop dim uses `stop_noise`; the
    # steer dim keeps 0.05. Both trainers call make_action_noise_sigma, so this
    # covers trailer + tractor-only.
    import numpy as np
    import train as e2e_train
    if not hasattr(e2e_train, "_stop_signal_orig_noise"):
        e2e_train._stop_signal_orig_noise = e2e_train.make_action_noise_sigma

    def _stop_signal_noise_sigma(n_actions):
        sigma = np.full(n_actions, 0.05, dtype=np.float32)
        if n_actions > 1:
            sigma[1:] = stop_noise
        return sigma

    e2e_train.make_action_noise_sigma = _stop_signal_noise_sigma
