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


def _wrap_class(env_cls, *, threshold: float, stop_penalty: float) -> None:
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
        return orig_step(self, action)

    env_cls.__init__ = stop_signal_init
    env_cls.step = stop_signal_step


def patch_stop_signal_action(
    e2e_rl_path,
    threshold: float = 0.0,
    stop_penalty: float = 200.0,
) -> None:
    """Apply the constant-speed + stop-signal action patch to BOTH base lidar
    env classes (forward + reverse).

    Parameters
    ----------
    e2e_rl_path : path-like
        Filesystem path to the e2e_rl checkout (added to ``sys.path``).
    threshold : float
        ``stop_signal`` above this value triggers a stop (default 0.0).
    stop_penalty : float
        Magnitude of the negative reward returned on a stop (default 200.0).
        The returned reward is ``-abs(stop_penalty)``.
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
    )
    _wrap_class(
        lf.ReverseLidarStateObservationLineFollowingEnv,
        threshold=threshold,
        stop_penalty=stop_penalty,
    )
