"""Classical pure-pursuit steering policies — drop-in at the TD3 predict() seam.

Selected with controller:=pure_pursuit (rl_bridge_node param). The object
mimics the SB3 API (`predict(obs, deterministic=True) -> (action, None)`) and
is swapped in for `self.model` / `self.reverse_model` AFTER the normal env +
model construction, so every part of the bridge pipeline — the adapter's
centerline rotation/Y-flip, reverse_native_obs / forward_native_obs surgery,
the post-predict steer-rate sign flips, integration, clamps and publishing —
is byte-identical to the TD3 path.

Why no extra transformations are needed here (Ben asked for "the same
transformations as the RL policy"): both control laws below are ODD functions
of the observation vector (atan is odd, everything else is linear), so they
are mirror-equivariant. The bridge's forward mirror convention (obs[0] = -steer,
Y-flipped world, -action[0] on the way out) and the reverse native-obs
convention (obs[0] left mirrored, reverse_steer_rate_sign = -1, so the
policy-frame steering integrates the policy's own action) therefore cancel for
pure pursuit exactly the way they cancel for the trained policies. The
controller just reads the obs the TD3 would have read.

Laws + default gains are the e2e_rl-tuned ones (controllers/tuned_params.json),
identical numbers to tractor_trailer_rl_cupy/batched/pure_pursuit.py FWD_GAINS /
REV_GAINS_E2ERL — tuned in the same e2e_rl env family this bridge replicates:

  forward:  delta_des = -(k_ff*atan(L*k1) + k_y*e_y + k_theta*e_psi)
  reverse:  delta_des =  k_hitch*hitch + k_ff*k1 + k_y*e_y_t + k_theta*e_psi_t
            (k_hitch/k_y are curvature-scheduled — see _curve_blend)
  then      steer_rate = clip(k_steer * (delta_des - s) / dt, +/-max_rate)

Tractor-only (no trailer) uses the same two laws with the trailer terms removed:
  forward:  identical to above — e_y/e_psi are already the tractor's own errors,
            only the obs indices shift (see predict).
  reverse:  delta_des =  k_ff*k1 + k_y*e_y + k_theta*e_psi
            The k_hitch term DROPS OUT (a single rigid body has no hitch and
            cannot jackknife), and the path terms fall back to the tractor's own
            errors — the only ones a tractor-only obs carries. k_y stays
            curvature-scheduled; k_hitch_straight/k_hitch_curve are simply
            unused in this mode.

NOTE (cupy pure_pursuit.py header): these reverse gains destabilised the
much larger SHUNT truck and had to be re-tuned there — the hitch-stabiliser
sign is geometry-dependent. For THIS lab geometry (0.65 m wheelbase, 2.8 m
trailer) the e2e_rl set is the tuned one. Verify in sim before a live run.

Obs layouts at the seam (raw physical units, see ros_env_adapter):
  trailer (32-D):      [s, hitch, e_y, e_psi, e_y_t, e_psi_t, k1, k2, lidar*24]
  tractor-only (29-D): [s, e_y, e_psi, k1, k2, lidar*24]

The tractor-only laws were added 2026-08-03. No tuned tractor-only reverse law
existed to inherit — every reverse PP reference in the codebase is hitch-based
(e2e_rl controllers/pure_pursuit.ReverseHitchPurePursuitController, the cupy
REV_GAINS, pid.py REV_GAINS) — so it was written by dropping the trailer's hitch
term and falling back to the tractor's own errors. RViz-validated same day:
the term SIGNS are correct as written (native e_y/e_psi, no negation needed),
and only k_y was wrong — trailer-tuned values oscillated on straights and had to
come down to 0.1 in both directions. See FWD_GAINS_TRACTOR_ONLY /
REV_GAINS_TRACTOR_ONLY.
"""

import numpy as np

FWD_GAINS = dict(
    k_steer=0.2,
    k_ff=0.7023064120291692,
    k_y=0.17023881008095487,
    k_theta=2.778463838892346,
)
# TRACTOR-ONLY (no trailer) forward. Same law, but k_y cut to 0.1 — Ben,
# 2026-08-03: the trailer-tuned k_y oscillated the truck about the path on
# straights. See REV_GAINS_TRACTOR_ONLY for why a single body needs a much
# slower lateral loop than the tuned trailer value.
FWD_GAINS_TRACTOR_ONLY = dict(
    k_steer=0.2,
    k_ff=0.7023064120291692,
    k_y=0.1,
    k_theta=2.778463838892346,
)
# Reverse gains hand-tuned live in the planning sim on the 3D map
# (Ben, 2026-07-30): the e2e_rl-tuned set (k_hitch=0.798, k_y=1.246) slalomed
# the tractor and the hitch oscillation GREW (8->24 deg); more hitch damping
# and less trailer lateral feedback stabilised it.
# 2026-08-01: no single (k_hitch, k_y) pair works everywhere — straights want
# hitch damping, corners want more trailer lateral feedback. So k_hitch/k_y are
# curvature-scheduled between two independent pairs (see _curve_blend below);
# every one of the four is live-tunable as pp_rev_<key>, because rl_bridge_node
# declares a param per key in this dict.
# First cut was a straight SWAP of one pair (straight 3.0/1.0 -> curve 1.0/3.0).
# That helped but k_y = 3.0 in the corner was too hot — Ben saw the trailer
# oscillating about the path — hence the separate, gentler curve pair.
# (k_theta/k_ff unchanged from the e2e_rl set, and NOT scheduled.)
REV_GAINS = dict(
    k_steer=0.2,
    k_hitch_straight=3.0,
    k_y_straight=0.8,
    k_hitch_curve=1.5,
    k_y_curve=1.5,
    k_theta=1.846962929234278,
    k_ff=-1.995183948132352,
)

# TRACTOR-ONLY (no trailer) reverse. No k_hitch_* keys at all: a single rigid
# body has no hitch, so those gains are meaningless here and omitting them keeps
# rl_bridge_node from declaring dead pp_rev_k_hitch_* params in this mode.
#
# k_y_straight cut 0.8 -> 0.1 (Ben, live test 2026-08-03: the trailer value
# oscillated the steering ±8 deg on straights; 0.1 fixed it). The idealised
# second-order lateral loop said 0.8 was already overdamped (zeta = k_theta /
# (2*sqrt(k_y*L)) = 1.28), which is why this needed a live test to settle: that
# model has no term for the one-tick-deadbeat steering command feeding a
# rate-limited axis, and it's that lag which actually sets the stability limit.
# The fix works by timescale separation — w_n = V*sqrt(k_y/L) drops 0.44 -> 0.16
# rad/s, pulling the lateral loop clear of the steering dynamics. zeta is
# speed-independent, so this survives changes to bridge_velocity_max_reverse.
#
# k_y_curve stays 1.5: the corner tracked well and was never the problem (there
# k_ff*k1 supplies most of the steering, so the lateral term is secondary).
REV_GAINS_TRACTOR_ONLY = dict(
    k_steer=0.2,
    k_y_straight=0.1,
    k_y_curve=1.5,
    k_theta=1.846962929234278,
    k_ff=-1.995183948132352,
)

# --- reverse curvature scheduling (k_hitch <-> k_y swap) -------------------
# obs[6] (k1) is the path curvature 5 m AHEAD OF THE TRAILER AXLE in the
# direction of travel: compute_curvature() takes the centerline sample nearest
# the trailer axle and steps +10 samples at centerline_resolution_m = 0.5 m.
# (k2 = obs[7] is the same thing at +20 samples = 10 m; unused here.)
#
# Band: both lanelet maps are strongly bimodal. Profiled 2026-08-01 at the
# 0.5 m centerline resolution the node actually publishes —
#   3d_map (ll 255, 60.5 m): 78% of samples |k| < 0.02, corner clips at 0.3
#   lab_map (ll 77, 18.7 m): median |k| 0.0003, p90 0.22, corner clips at 0.3
# The only things in the middle are isolated SINGLE-sample lanelet-vertex
# spikes (3d_map: |k| = 0.096 at arc 16 m, 0.108 at arc 23 m), so the low edge
# sits above those to keep a vertex artefact from flipping the gains on a
# straight.
CURVE_K_LO = 0.12      # |k1| at/below -> the *_straight pair (w = 0)
CURVE_K_HI = 0.25      # |k1| at/above -> the *_curve pair    (w = 1)
# k1 is a PREVIEW, so it decays back to ~0 while the vehicle is still driving
# the corner it saw 5 m ago — on 3d_map k1 peaks with the trailer at arc
# 31-34 m but the corner itself is arc 35-39 m, so a memoryless k1 -> gain map
# would apply curve gains ONLY BEFORE the corner and straight gains inside it.
# (Same on lab_map: k1 peaks at arc 3-5 m, corner is arc 6-9 m.) Attack is instant
# — transitioning early is the point — but release is rate-limited to the time
# needed to cover the lookahead: 5 m / 0.4 m/s (bridge_velocity_max_reverse).
CURVE_RELEASE_S = 12.5


class PurePursuitPolicy:
    def __init__(self, reverse: bool, tractor_only: bool, wheelbase_m: float,
                 dt: float, action_space, gains: dict | None = None,
                 aux_action_value: float = -1.0):
        """aux_action_value fills action[1] when the action space is 2-D.
        For stop_signal that means "never request a stop" (-1.0 << threshold);
        arrival is handled by lane_reference_node's goal gate."""
        self.reverse = reverse
        self.tractor_only = tractor_only
        self.L = float(wheelbase_m)
        self.dt = float(dt)
        self.low = np.asarray(action_space.low, dtype=np.float64)
        self.high = np.asarray(action_space.high, dtype=np.float64)
        self.n_action = int(self.low.shape[0])
        self.aux = float(aux_action_value)
        if gains:
            self.g = dict(gains)
        elif reverse:
            self.g = dict(REV_GAINS_TRACTOR_ONLY if tractor_only else REV_GAINS)
        else:
            self.g = dict(FWD_GAINS_TRACTOR_ONLY if tractor_only else FWD_GAINS)
        self._curve_w = 0.0   # reverse only; 0 = *_straight pair, 1 = *_curve pair

    def _curve_weight(self, k1):
        """Advance and return the curve blend weight w in [0, 1] from the
        lookahead curvature. MUTATES self._curve_w — call exactly once per tick.
        """
        w = (abs(float(k1)) - CURVE_K_LO) / (CURVE_K_HI - CURVE_K_LO)
        w = min(1.0, max(0.0, w))
        # Instant attack, rate-limited release (see CURVE_RELEASE_S).
        self._curve_w = max(w, self._curve_w - self.dt / CURVE_RELEASE_S)
        return self._curve_w

    def _blend(self, w, key):
        """Interpolate <key>_straight -> <key>_curve on w. w = 0 gives the
        straight value verbatim, w = 1 the curve value, so both ends stay
        directly tunable via pp_rev_*.
        """
        return (1.0 - w) * self.g[key + "_straight"] + w * self.g[key + "_curve"]

    def predict(self, obs, deterministic=True):
        o = np.asarray(obs, dtype=np.float64).ravel()
        s = o[0]
        if self.reverse:
            if self.tractor_only:
                # No trailer: the k_hitch stabiliser has nothing to act on, so
                # it drops out and the path terms use the TRACTOR's own errors.
                # Same shape and same seam handling as the trailer law below
                # (no leading minus, i.e. reverse == forward with the steering
                # sense inverted); only the hitch term and obs indices differ.
                #
                # FRAME: e_y/e_psi arrive NATIVE here — ros_env_adapter negates
                # obs[1:5] and, unlike the 8-dim trailer layout, does NOT
                # re-mirror the path errors afterwards (that re-mirror exists
                # only to keep the trailer errors coherent with the HITCH's
                # control direction; with no hitch the rationale is gone). Native
                # is also the frame the reverse tractor-only TD3 was proven in
                # (adapter note, 2026-06-28). So these gains act on the
                # OPPOSITE-signed laterals from the trailer law — do NOT assume a
                # trailer gain transfers here. Confirmed correct as written in
                # RViz 2026-08-03: no negation of o[1]/o[2] needed; the observed
                # straight-line oscillation was k_y magnitude, not sign.
                e_y, e_psi, k1 = o[1], o[2], o[3]
                k_y = self._blend(self._curve_weight(k1), "k_y")
                delta_des = (self.g["k_ff"] * k1 + k_y * e_y
                             + self.g["k_theta"] * e_psi)
            else:
                hitch, e_y_t, e_psi_t, k1 = o[1], o[4], o[5], o[6]
                w = self._curve_weight(k1)
                k_hitch, k_y = self._blend(w, "k_hitch"), self._blend(w, "k_y")
                delta_des = (k_hitch * hitch + self.g["k_ff"] * k1
                             + k_y * e_y_t + self.g["k_theta"] * e_psi_t)
            # Reverse seam frame: laterals arrive native, but obs[0] is the
            # MIRRORED steering and the bridge negates action[0] on the way out
            # (reverse_steer_rate_sign=-1). Unlike forward (a consistent global
            # mirror, odd law -> cancels), this mixed frame requires mirroring
            # the desired steering: rate_true = -(a') with a' below gives
            # rate_true = (delta_des - s_true)/dt, the native law.
            delta_des = -delta_des
        else:
            if self.tractor_only:
                e_y, e_psi, k1 = o[1], o[2], o[3]
            else:
                e_y, e_psi, k1 = o[2], o[3], o[6]
            delta_des = -(self.g["k_ff"] * np.arctan(self.L * k1)
                          + self.g["k_y"] * e_y + self.g["k_theta"] * e_psi)
        # k_steer relaxes the classic one-tick deadbeat. k_steer=1.0 is the
        # textbook form (reach delta_des within a single dt) and is what the sim
        # tuning assumed; the default is now 0.2 (see below). But dt=0.1 makes that an effective loop gain of 10/s,
        # which saturates the ±deg2rad(25)=0.436 rad/s action bound for any
        # steering error over 2.5 deg — i.e. PP runs bang-bang. In sim that is
        # harmless because obs[0] (the bridge's VIRTUAL _target_steering) tracks
        # the vehicle's real tire angle closely. On the real robot the tire angle
        # lags (servo + CAN + STEER_SCALE), so a controller that assumes its
        # command took effect instantly limit-cycles.
        #
        # Default 0.2 (Ben, 2026-08-03) => effective gain 2/s, steering inner-loop
        # time constant tau = dt/k_steer = 0.5 s. Picked from the design rule
        # "command the inner loop SLOWER than the actuator can move", against
        # Ben's (UNVALIDATED) observation that the Hunter takes ~0.5 s to go from
        # centred to full lock. Measure it properly by stepping the command and
        # timing /vehicle/status/steering_status to 63% — if the true tau is
        # nearer 0.2 s, k_steer can come back up toward 0.5.
        # NOTE this changes SIM behaviour too: every PP gain tuned before this
        # date was tuned at k_steer=1.0 (deadbeat) with the action bound
        # saturated, so those gains may want revisiting now that it is relaxed.
        # Set pp_{fwd,rev}_k_steer 1.0 to reproduce the old deadbeat exactly.
        steer_rate = self.g["k_steer"] * (delta_des - s) / self.dt
        action = np.full(self.n_action, self.aux, dtype=np.float64)
        action[0] = steer_rate
        action = np.clip(action, self.low, self.high)
        return action.astype(np.float32), None
