#!/usr/bin/env python3
"""Delay-aware reverse-driving TD3 trainer (v17).

Implements the DA-MDP framework described in notes2.txt:
  - Augments the observation with an action-history buffer (last K steering
    actions) to recover the Markov property under actuator delay.
  - GRU encodes the action history into a latent z_t; z_t is concatenated
    with the instantaneous state s_t before the actor/critic heads.
  - Progressive curriculum: Phase 1 (τ=0) → Phase 2 (constant τ) →
    Phase 3 (randomized τ ~ U[0, τ_max]).

Reuses lab-scale config overrides, env-vehicle patches, path-mode mixture,
and variable-speed action-space patches from train_lab_model.py.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_E2E_RL = Path("/home/ben/Ben/Thesis/e2e_rl")
DEFAULT_OUT = REPO_ROOT / "lab_models_v17"
SCRIPTS_DIR = REPO_ROOT / "src/electrans_rl_bridge/scripts"

sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------- delay patch
def patch_vehicle_with_delay(
    e2e_rl_path: Path,
    *,
    steer_tau_fn,
    velocity_tau_fn,
):
    """Patch StateSpaceVehicleModel with configurable per-episode actuator lag.

    steer_tau_fn / velocity_tau_fn are callables returning a τ value
    (called once per reset). This lets Phase 3 randomize τ per episode.
    """
    if str(e2e_rl_path) not in sys.path:
        sys.path.insert(0, str(e2e_rl_path))

    import numpy as np
    from VehicleModels.vehicle_model import StateSpaceVehicleModel

    orig_loop = StateSpaceVehicleModel.loop
    orig_reset = StateSpaceVehicleModel.reset

    def patched_reset(self, xd, x=0, y=0, p=0):
        orig_reset(self, xd, x, y, p)
        self._s_target = 0.0
        self._xd_target = float(xd)
        # Resample τ for this episode
        self._steer_tau = float(steer_tau_fn())
        self._velocity_tau = float(velocity_tau_fn())

    def patched_loop(self, action):
        if not hasattr(self, "_s_target"):
            self._s_target = float(self.s)
        if not hasattr(self, "_xd_target"):
            self._xd_target = float(self.xd)
        if not hasattr(self, "_steer_tau"):
            self._steer_tau = float(steer_tau_fn())
        if not hasattr(self, "_velocity_tau"):
            self._velocity_tau = float(velocity_tau_fn())

        # 1. Update steering target by policy's commanded rate, clamp ±π/4
        self._s_target = float(np.clip(
            self._s_target + action[0] * self.dt, -np.pi / 4, np.pi / 4
        ))
        # 2. Actual tire angle lags target with first-order dynamics.
        # Use the exact-exponential discretisation rather than forward-
        # Euler so the integration is unconditionally stable for any τ:
        #     ds/dt = (s_target - s) / τ   →   s_next = s_target + (s - s_target) * exp(-dt/τ)
        # Forward-Euler s += (s_target - s) * (dt/τ) is unstable for dt/τ ≥ 2
        # (the case for τ < ~0.05 s with dt = 0.1 s) and overshoots the
        # target by ~(dt/τ)×, producing chaotic chattering. The exact form
        # uses α = 1 - exp(-dt/τ) ∈ [0, 1], always interpolating monotonically
        # between current and target with no overshoot.
        import math
        if self._steer_tau > 1e-6:
            alpha_s = 1.0 - math.exp(-self.dt / self._steer_tau)
            self.s = self.s + (self._s_target - self.s) * alpha_s
        else:
            self.s = self._s_target

        # 3. Velocity target lag (same exact-exponential form).
        if self._velocity_tau > 1e-6:
            alpha_v = 1.0 - math.exp(-self.dt / self._velocity_tau)
            self._xd_target = self._xd_target + (
                action[1] - self._xd_target
            ) * alpha_v
        else:
            self._xd_target = float(action[1])

        # 4. Original loop with already-applied steer and lagged velocity
        modified_action = np.array([0.0, self._xd_target], dtype=float)
        return orig_loop(self, modified_action)

    StateSpaceVehicleModel.loop = patched_loop
    StateSpaceVehicleModel.reset = patched_reset


# ---------------------------------------------------------------- env wrapper
# NOTE: Module-level (not in a factory function) so it pickles by class
# reference. Older actions (most-distant past) at index 0; newest at K-1.
import gymnasium as gym
from gymnasium import spaces


class ActionHistoryWrapper(gym.ObservationWrapper):
    """Augments observation with a sliding window of last K actions.

    Implements the DA-MDP state augmentation s̃_t = [s_t, a_{t-1},
    a_{t-2}, ..., a_{t-K}]. The GRU features extractor consumes the
    action_history portion to encode delay-aware context.

    v19+: flat Box obs (state ++ flattened action_history) instead of Dict,
    so SB3's NStepReplayBuffer works (it doesn't yet support Dict obs).
    The GRU extractor slices the Box back into state + (K, action_dim)
    history internally — see GRUDelayAwareFeatureExtractor.forward.
    """

    def __init__(self, env, K: int = 5):
        super().__init__(env)
        self.K = int(K)
        self.action_dim = int(env.action_space.shape[0])
        inner = env.observation_space
        self.state_dim = int(inner.shape[0])
        # Concatenate state + flattened action_history. History bounds are
        # ±inf since it carries normalized actions; state bounds come from
        # the inner env.
        low = np.concatenate([
            inner.low.astype(np.float32),
            -np.inf * np.ones((self.K * self.action_dim,), dtype=np.float32),
        ])
        high = np.concatenate([
            inner.high.astype(np.float32),
            np.inf * np.ones((self.K * self.action_dim,), dtype=np.float32),
        ])
        self.observation_space = spaces.Box(
            low=low, high=high,
            shape=(self.state_dim + self.K * self.action_dim,),
            dtype=np.float32,
        )
        self._history = np.zeros((self.K, self.action_dim), dtype=np.float32)

    def reset(self, **kwargs):
        self._history[:] = 0.0
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        # Record this action BEFORE stepping so subsequent obs reflects
        # the action that's about to enter the actuator pipeline.
        self._history = np.roll(self._history, -1, axis=0)
        self._history[-1] = np.asarray(action, dtype=np.float32)
        obs, reward, term, trunc, info = self.env.step(action)
        return self.observation(obs), reward, term, trunc, info

    def observation(self, obs):
        state = np.asarray(obs, dtype=np.float32)
        return np.concatenate([state, self._history.flatten()]).astype(np.float32)


def make_action_history_wrapper(K: int):
    """Backward-compat shim — returns the module-level class with K set."""
    def _factory(env):
        return ActionHistoryWrapper(env, K=K)
    return ActionHistoryWrapper  # for class references in older callsites


# ---------------------------------------------------------------- GRU encoder
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GRUDelayAwareFeatureExtractor(BaseFeaturesExtractor):
    """Encodes Dict observation {state, action_history} into a single
    feature vector by passing action_history through a GRU and
    concatenating its final hidden state with the raw state vector.

    The instantaneous state already contains tire angle s, hitch γ,
    and cross-track terms — the GRU's job is to encode the *pending*
    actions in the actuator pipeline, recovering the true current
    delay-shifted state per DA-MDP."""

    def __init__(self, observation_space, gru_hidden: int = 64,
                 gru_weight: float = 1.0,
                 state_dim: int = 32, K: int = 5, action_dim: int = 2):
        # v19+: observation_space is a flat Box (state_dim + K*action_dim,)
        # rather than a Dict. We slice internally. The caller must pass
        # state_dim, K, action_dim explicitly via features_extractor_kwargs
        # so we know where to slice — these come from the env wrapper.
        features_dim = state_dim + gru_hidden
        super().__init__(observation_space, features_dim=features_dim)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.K = int(K)
        self.gru_hidden = gru_hidden
        self.gru = nn.GRU(
            input_size=self.action_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )
        # Scalar weight applied to the GRU output before concatenation.
        # Stored as a buffer so it persists in the state_dict across
        # train/save/load and can be updated by the curriculum runner
        # between phases.
        self.register_buffer(
            "gru_weight", th.tensor(float(gru_weight), dtype=th.float32)
        )

    def forward(self, observations):
        # observations: (B, state_dim + K*action_dim) flat Box
        state = observations[:, :self.state_dim]
        history_flat = observations[:, self.state_dim:]
        history = history_flat.view(-1, self.K, self.action_dim)
        _, h_n = self.gru(history)                     # (1, B, gru_hidden)
        z = h_n.squeeze(0)                             # (B, gru_hidden)
        return th.cat([state, self.gru_weight * z], dim=-1)


def set_gru_weight(model, value: float) -> None:
    """Update gru_weight buffer on all four feature extractors (actor,
    actor_target, critic, critic_target) of a loaded TD3 model.
    Called between curriculum phases."""
    for net_attr in ("actor", "actor_target", "critic", "critic_target"):
        net = getattr(model.policy, net_attr, None)
        if net is None:
            continue
        fe = getattr(net, "features_extractor", None)
        if fe is None or not hasattr(fe, "gru_weight"):
            continue
        fe.gru_weight.fill_(float(value))


def make_gru_features_extractor(gru_hidden: int = 64):
    """Backward-compat shim — returns the module-level class."""
    return GRUDelayAwareFeatureExtractor


# ---------------------------------------------- v19: ASAP smoothing + TD3 subclass
import torch.nn.functional as F
from stable_baselines3 import TD3 as _SB3TD3
from stable_baselines3.common.utils import polyak_update


class TD3WithASAP(_SB3TD3):
    """TD3 + ASAP second-order action smoothing loss in the actor objective.

    Per the document: `L_smooth = β · E[||a_t − 2·a_{t-1} + a_{t-2}||²]`
    suppresses high-frequency chattering by penalising steering acceleration
    (the second-order temporal derivative of action). Without this, TD3's
    actor finds a local optimum that bang-bangs the steering to survive
    longer under delay — exactly the failure mode we observed in v18.

    We exploit the DA-MDP observation: action_history already contains the
    last K actions. The 'previous action' a_{t-1} is the newest entry, and
    a_{t-2} the one before. No replay-buffer modification required.
    """

    def __init__(self, *args, asap_beta: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.asap_beta = float(asap_beta)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Re-implement TD3.train() with ASAP term in the actor loss.
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses, asap_losses = [], [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            discounts = (
                replay_data.discounts if replay_data.discounts is not None else self.gamma
            )

            with th.no_grad():
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = sum(F.mse_loss(q, target_q_values) for q in current_q_values)
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if self._n_updates % self.policy_delay == 0:
                actions_t = self.actor(replay_data.observations)
                actor_q_loss = -self.critic.q1_forward(
                    replay_data.observations, actions_t
                ).mean()

                # --- ASAP second-order action smoothing penalty ---
                # Observations are flat Box (state ++ flattened action_history).
                # We pull the slicing dims from the actor's features_extractor,
                # which knows state_dim, K, action_dim from features_extractor_kwargs.
                fe = self.actor.features_extractor
                if (
                    self.asap_beta > 0.0
                    and hasattr(fe, "state_dim")
                    and hasattr(fe, "K")
                    and hasattr(fe, "action_dim")
                    and fe.K >= 2
                ):
                    obs = replay_data.observations
                    history_flat = obs[:, fe.state_dim:]
                    history = history_flat.view(-1, fe.K, fe.action_dim)
                    a_tm1 = history[:, -1, :]
                    a_tm2 = history[:, -2, :]
                    # Second-order difference: a_t - 2·a_{t-1} + a_{t-2}
                    second_diff = actions_t - 2.0 * a_tm1 + a_tm2
                    asap_loss = (second_diff ** 2).sum(dim=-1).mean()
                    actor_loss = actor_q_loss + self.asap_beta * asap_loss
                    asap_losses.append(asap_loss.item())
                else:
                    actor_loss = actor_q_loss

                actor_losses.append(actor_loss.item())
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if asap_losses:
            self.logger.record("train/asap_loss", np.mean(asap_losses))


# ---------------------------------------------- v19: anti-jackknife safety override
def patch_env_with_safety_override(
    e2e_rl_path: Path,
    *,
    hitch_threshold_rad: float = 0.7,  # ≈ 40° per the document
):
    """Patch the reverse env's step() to override the policy's steering
    action when |hitch_angle| exceeds threshold, applying max counter-steer.

    Per the document:
    > 'A Sliding Mode Controller (SMC) combined with a jackknife prevention
    >  override can enforce strict mathematical boundaries on the steering
    >  rate, ensuring safe reverse driving... If the joint angle exceeds a
    >  critical boundary (e.g., |θ_a| ≥ 40°), the supervisor overrides the
    >  policy's commanded steering input and applies maximum counter-steering'

    Effect: the policy CANNOT learn that bang-bang chattering is survivable,
    because at the moment it pushes the trailer into a near-jackknife state
    (|hitch| > 0.7 rad), the env replaces its steering with a counter-steer
    that drives the hitch back toward 0. This forces the policy to find
    actually-stable control strategies — chattering no longer pays off
    because the supervisor cleans up after every dangerous excursion.
    """
    if str(e2e_rl_path) not in sys.path:
        sys.path.insert(0, str(e2e_rl_path))

    import numpy as np
    import Environments.LineFollowing as lf

    orig_step = lf.ReverseStateObservationLineFollowingEnv.step

    def patched_step(self, action):
        # Hitch = vehicle.p - trailer.yaw (instantaneous, pre-step state)
        hitch = float(self.vehicle.p - self.vehicle.trailer.yaw)
        # Normalise into [-π, π] so wrap-around doesn't break the threshold
        hitch_wrapped = (hitch + np.pi) % (2 * np.pi) - np.pi
        if abs(hitch_wrapped) > hitch_threshold_rad:
            # Force max counter-steer (action[0] is steering rate; flip its
            # sign opposite to hitch_wrapped to drive hitch back toward 0).
            # Max rate magnitude = action_space high[0].
            max_rate = float(self.action_space.high[0])
            counter_rate = -np.sign(hitch_wrapped) * max_rate
            # Keep velocity command unchanged
            action = np.array([counter_rate, float(action[1])], dtype=action.dtype)
        return orig_step(self, action)

    lf.ReverseStateObservationLineFollowingEnv.step = patched_step


# --------------------------------------------------------- reward/step eval CB
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class RewardPerStepEvalCallback(BaseCallback):
    """Eval callback that saves the model with the **highest reward-per-step**
    instead of the highest total reward.

    Critical fix for environments where the standard EvalCallback would
    save fail-fast policies: in our reverse-driving env, a 13-step crash
    accumulates -500 terminal + ~+10 step reward → total ≈ -490, whereas
    an 800-step long-survival policy accumulates -500 terminal + 800 × -1
    step reward → total ≈ -1300. SB3's default `best_mean_reward`
    comparison picks the *crashing* policy as "best" because -490 > -1300.

    Reward/step inverts this correctly:
      - Fail-fast: r/step ≈ -490/13 ≈ -38   (bad)
      - Long-failing: r/step ≈ -1300/800 ≈ -1.6   (better)
      - Successful: r/step dominated by +200 terminal over ~100 steps ≈ +5+  (best)
    """

    def __init__(
        self, eval_env, best_model_save_path: str, log_path: str,
        eval_freq: int, n_eval_episodes: int = 10, deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.deterministic = deterministic
        self.best_rps = -float("inf")
        self._history = []  # list of (timestep, rewards_array, lengths_array)
        os.makedirs(best_model_save_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if self.n_calls % self.eval_freq != 0:
            return True

        ep_rewards, ep_lengths = evaluate_policy(
            self.model, self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            return_episode_rewards=True,
        )
        ep_rewards = np.asarray(ep_rewards, dtype=np.float64)
        ep_lengths = np.asarray(ep_lengths, dtype=np.float64)
        mean_r = float(ep_rewards.mean())
        mean_L = float(ep_lengths.mean())
        # Use mean total reward / mean length so a few short-episode outliers
        # don't dominate the per-episode r/step distribution.
        rps = mean_r / max(mean_L, 1.0)

        self._history.append((int(self.num_timesteps), ep_rewards.copy(), ep_lengths.copy()))

        is_new_best = rps > self.best_rps
        if is_new_best:
            self.best_rps = rps
            best_zip = os.path.join(self.best_model_save_path, "best_model.zip")
            self.model.save(best_zip)
            tag = "** NEW BEST **"
        else:
            tag = ""

        if self.verbose:
            print(
                f"[RPS-eval] ts={self.num_timesteps:>8}  rps={rps:>7.3f}  "
                f"reward={mean_r:>8.1f}  ep_len={mean_L:>7.1f}  {tag}",
                flush=True,
            )

        # Save eval log compatible with the existing analysis tooling
        # (timesteps, results, ep_lengths arrays — same shape as SB3's npz).
        timesteps = np.array([h[0] for h in self._history], dtype=np.int64)
        results = np.stack([h[1] for h in self._history], axis=0)
        ep_lens = np.stack([h[2] for h in self._history], axis=0)
        np.savez(
            os.path.join(self.log_path, "evaluations.npz"),
            timesteps=timesteps,
            results=results,
            ep_lengths=ep_lens,
        )
        return True


# ---------------------------------------------------------------- env factory
def make_lab_env(
    e2e_rl_path: Path,
    *,
    K: int,
    lidar_beams: int,
    reward_mode: str,
    max_episode_steps: int = 1000,
):
    """Build the lab-scale reverse env with action-history wrapper."""
    import Environments.LineFollowing as lf
    HistWrapper = make_action_history_wrapper(K)
    env = lf.ReverseLidarStateObservationLineFollowingEnv(
        render_mode=None,
        max_episode_steps=max_episode_steps,
        lidar_beams=lidar_beams,
        reward_mode=reward_mode,
        fixed_speed=False,  # variable-speed
    )
    return HistWrapper(env)


# ---------------------------------------------------------------- main runner
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--prev-phase-zip", type=str, default=None,
                        help="Path to previous phase's best_model.zip to warm-start from.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--lidar-beams", type=int, default=24)
    parser.add_argument("--reward", default="multiplicative")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--K", type=int, default=5,
                        help="Action-history window length")
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--v-min", type=float, default=0.5)
    parser.add_argument("--v-max", type=float, default=3.0)
    parser.add_argument("--steer-tau", type=float, default=0.0,
                        help="Steering actuator lag in seconds (Phase 1: 0.0, Phase 2: 0.05)")
    parser.add_argument("--velocity-tau", type=float, default=0.0,
                        help="Velocity actuator lag in seconds (Phase 1: 0.0, Phase 2: 0.10)")
    parser.add_argument("--randomize-delay", action="store_true",
                        help="Phase 3: sample τ ~ U[0, steer_tau] / U[0, velocity_tau] per episode")
    parser.add_argument("--gru-weight", type=float, default=1.0,
                        help="Scalar multiplier on GRU output (curriculum knob; "
                             "Phase 1=0.0 → state-only baseline, ramp to 1.0 across Phase 2)")
    parser.add_argument("--asap-beta", type=float, default=0.0,
                        help="ASAP second-order action smoothing loss weight β. "
                             "0 disables. Doc recommends 0.05–0.10. Critical for "
                             "preventing the bang-bang chattering local optimum in "
                             "delayed envs.")
    parser.add_argument("--safety-override", action="store_true",
                        help="Apply anti-jackknife env safety override: when "
                             "|hitch| > 0.7 rad (≈40°), force max counter-steer "
                             "regardless of policy action. Prevents policy from "
                             "learning that chattering-near-jackknife is survivable.")
    parser.add_argument("--safety-threshold", type=float, default=0.7,
                        help="Hitch angle (rad) above which safety override kicks in.")
    parser.add_argument("--n-steps", type=int, default=3,
                        help="N-step bootstrapping length for the Bellman target. "
                             "Doc recommends N=2-5 to match the actuator delay. "
                             "Critical for correct credit assignment under delay: "
                             "single-step TD assigns reward to the wrong action when "
                             "the action's effect hasn't propagated yet. Activates "
                             "SB3's NStepReplayBuffer automatically when >1.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--e2e-rl-path", default=str(DEFAULT_E2E_RL))
    args = parser.parse_args()

    e2e_rl_path = Path(args.e2e_rl_path).resolve()
    out_dir = Path(args.out_dir).resolve()

    # ---- apply lab config + path-generator + vehicle-params patches ----
    sys.path.insert(0, str(SCRIPTS_DIR))
    sys.path.insert(0, str(e2e_rl_path))
    import train_lab_model as tlm

    tlm._apply_lab_config_overrides(e2e_rl_path)
    tlm._patch_env_vehicle_params(e2e_rl_path)
    tlm._patch_path_generator(e2e_rl_path)
    tlm._patch_variable_speed_action(e2e_rl_path, v_min=args.v_min, v_max=args.v_max)
    tlm._patch_velocity_randomisation(e2e_rl_path)

    # ---- apply delay patch (per-phase) ----
    rng = np.random.default_rng()
    if args.randomize_delay:
        # Phase 3: per-episode random τ ∈ [0, steer_tau] / [0, velocity_tau]
        steer_max = float(args.steer_tau)
        vel_max = float(args.velocity_tau)
        def steer_tau_fn():
            return float(rng.uniform(0.0, steer_max))
        def velocity_tau_fn():
            return float(rng.uniform(0.0, vel_max))
    else:
        # Phase 1 (τ=0) or Phase 2 (constant τ)
        steer_const = float(args.steer_tau)
        vel_const = float(args.velocity_tau)
        def steer_tau_fn():
            return steer_const
        def velocity_tau_fn():
            return vel_const
    patch_vehicle_with_delay(
        e2e_rl_path,
        steer_tau_fn=steer_tau_fn,
        velocity_tau_fn=velocity_tau_fn,
    )

    # v19: Apply env-side safety override (anti-jackknife counter-steer)
    if args.safety_override:
        patch_env_with_safety_override(
            e2e_rl_path,
            hitch_threshold_rad=args.safety_threshold,
        )
        print(f"[v17 phase={args.phase}] safety override ENABLED at |hitch|>{args.safety_threshold} rad")

    print(f"[v17 phase={args.phase}] steer_tau={args.steer_tau}s "
          f"velocity_tau={args.velocity_tau}s randomize={args.randomize_delay}")

    # ---- build vec envs ----
    from stable_baselines3 import TD3
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    def make_env_fn():
        def _f():
            env = make_lab_env(e2e_rl_path, K=args.K,
                               lidar_beams=args.lidar_beams,
                               reward_mode=args.reward)
            return Monitor(env)
        return _f

    if args.n_envs > 1:
        # Need to apply patches inside subprocess too
        import multiprocessing as mp
        mp.set_start_method("fork", force=True)
        train_env = SubprocVecEnv([make_env_fn() for _ in range(args.n_envs)])
    else:
        train_env = DummyVecEnv([make_env_fn()])

    eval_env = DummyVecEnv([make_env_fn()])

    # ---- features extractor ----
    GRUExtractor = make_gru_features_extractor(gru_hidden=args.gru_hidden)

    # v19+: env wrapper returns flat Box; tell the extractor where to slice.
    # action_dim from the inner env's action_space (variable_speed = 2).
    state_dim = int(train_env.observation_space.shape[0]) - args.K * 2
    policy_kwargs = dict(
        features_extractor_class=GRUExtractor,
        features_extractor_kwargs=dict(
            gru_hidden=args.gru_hidden,
            gru_weight=args.gru_weight,
            state_dim=state_dim,
            K=args.K,
            action_dim=2,
        ),
        net_arch=[256, 256],
        share_features_extractor=False,
    )

    save_dir = out_dir / f"phase_{args.phase}" / "models/reverse/lidar_24/multiplicative"
    save_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = save_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # ---- build / load model ----
    if args.prev_phase_zip and Path(args.prev_phase_zip).exists():
        print(f"[v17 phase={args.phase}] warm-starting from {args.prev_phase_zip}")
        model = TD3WithASAP.load(
            args.prev_phase_zip, env=train_env, device=args.device
        )
        model.asap_beta = float(args.asap_beta)
        from v17_delay_aware import set_gru_weight
        set_gru_weight(model, args.gru_weight)
        print(f"[v17 phase={args.phase}] gru_weight={args.gru_weight} asap_beta={args.asap_beta}")
    else:
        print(f"[v17 phase={args.phase}] training from scratch  asap_beta={args.asap_beta}  n_steps={args.n_steps}")
        model = TD3WithASAP(
            policy="MlpPolicy",
            env=train_env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            buffer_size=200_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            policy_delay=2,
            target_noise_clip=0.5,
            target_policy_noise=0.2,
            n_steps=args.n_steps,
            device=args.device,
            verbose=1,
            asap_beta=args.asap_beta,
        )

    eval_callback = RewardPerStepEvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(logs_dir),
        eval_freq=max(args.eval_freq // max(1, args.n_envs), 1),
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    print(f"[v17 phase={args.phase}] training {args.timesteps} steps  →  {save_dir}")
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=eval_callback, progress_bar=False)
    model.save(str(save_dir / "final.zip"))
    print(f"[v17 phase={args.phase}] done in {(time.time()-t0)/60:.1f} min")
    print(f"[v17 phase={args.phase}] best_model.zip at {save_dir}/best_model.zip")


if __name__ == "__main__":
    main()
