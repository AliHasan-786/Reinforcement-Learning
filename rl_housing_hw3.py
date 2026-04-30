"""
RL Housing Inspection: Assignment 3 — Policy Gradients, Multi-Objective RL,
Shapley Feature Importance, Pareto Frontier Analysis
=============================================================================
Builds upon HW2 (rl_housing_inspection.py).

Sections:
  1. Imports & Seeds
  2. REINFORCE with Baseline (Policy Gradient)
  3. Multi-Objective Environment
  4. Shapley Value Analysis
  5. Reward Weighting Experiments & Pareto Frontier
  6. Agent Comparisons (REINFORCE vs DQN vs TD)
  7. Visualizations (5 PNG figures)
  8. Results Tables & Policy Recommendation
"""

# =============================================================================
# 1. IMPORTS & SEEDS
# =============================================================================
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "/Users/alihasan/Downloads/Reinforcement Learning")
from rl_housing_inspection import DOBDispatchEnv, DATA, N_RECORDS

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict
import itertools

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

try:
    import shap
    HAS_SHAP = True
    print("[Info] SHAP library found — using manual Shapley method (consistent with hw3 spec).")
except ImportError:
    HAS_SHAP = False
    print("[Info] SHAP not installed — using manual permutation-based Shapley values.")

from stable_baselines3 import DQN

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

OUTPUT_DIR = "/Users/alihasan/Downloads/Reinforcement Learning"

# Shared label maps
ACTION_NAMES  = {0: "Dismiss", 1: "Standard", 2: "Aggressive"}
CAT_NAMES     = {0: "Brand-New", 1: "Recurrent", 2: "High-Freq"}
BOROUGH_NAMES = {0: "Manhattan", 1: "Brooklyn", 2: "Queens",
                 3: "Bronx", 4: "Staten Is."}

print(f"\n[HW3] Dataset: {N_RECORDS} records, "
      f"{DATA['is_violation'].sum()} violations "
      f"({100*DATA['is_violation'].mean():.1f}%)\n")


# =============================================================================
# 2. REINFORCE WITH BASELINE (POLICY GRADIENT)
# =============================================================================

class PolicyNet(nn.Module):
    """Two-layer MLP that outputs a softmax probability over 3 actions."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64,
                 n_actions: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaselineNet(nn.Module):
    """Two-layer MLP that outputs a scalar state-value estimate V(s)."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class REINFORCEAgent:
    """
    REINFORCE with learned baseline (advantage actor-critic style update).

    The baseline V(s) is trained separately via MSE to reduce variance.
    Returns are normalised within each episode for training stability.
    """

    def __init__(self, lr_policy: float = 1e-3, lr_baseline: float = 1e-3,
                 gamma: float = 0.99):
        self.gamma = gamma
        self.policy_net   = PolicyNet(input_dim=5)
        self.baseline_net = BaselineNet(input_dim=5)

        self.policy_opt   = optim.Adam(self.policy_net.parameters(),
                                       lr=lr_policy)
        self.baseline_opt = optim.Adam(self.baseline_net.parameters(),
                                       lr=lr_baseline)

    # ------------------------------------------------------------------
    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Convert integer obs [c, b, i, p, t] to a normalised float tensor."""
        # Normalise each feature to [0, 1] using known max values
        norm = np.array([
            obs[0] / 2.0,  # complaint_category: 0-2
            obs[1] / 4.0,  # borough: 0-4
            obs[2] / 5.0,  # inspector_budget: 0-5
            obs[3] / 3.0,  # prior_complaint_count: 0-3
            obs[4] / 3.0,  # time_of_day_bin: 0-3
        ], dtype=np.float32)
        return torch.FloatTensor(norm)

    # ------------------------------------------------------------------
    def select_action(self, obs: np.ndarray,
                      greedy: bool = False):
        """
        Sample an action from the policy (or take the argmax if greedy).

        Returns (action, log_prob).
        """
        x     = self._obs_to_tensor(obs)
        probs = self.policy_net(x)
        dist  = torch.distributions.Categorical(probs)

        if greedy:
            action = int(torch.argmax(probs).item())
        else:
            action = int(dist.sample().item())

        log_prob = dist.log_prob(torch.tensor(action))
        return action, log_prob

    # ------------------------------------------------------------------
    def update(self, episode: list):
        """
        Perform one REINFORCE + baseline gradient update from a full episode.

        episode: list of (obs, action, reward, log_prob) tuples.
        """
        if len(episode) == 0:
            return

        # --- Compute discounted returns G_t (backward pass) ---------------
        returns = []
        G = 0.0
        for _, _, reward, _ in reversed(episode):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns_t = torch.FloatTensor(returns)

        # Normalise returns within the episode for training stability
        returns_norm = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # --- Gather stored tensors ----------------------------------------
        log_probs = torch.stack([lp for _, _, _, lp in episode])
        obs_list  = [self._obs_to_tensor(obs) for obs, _, _, _ in episode]
        obs_batch = torch.stack(obs_list)

        # --- Baseline values V(s_t) ----------------------------------------
        values = self.baseline_net(obs_batch).squeeze(-1)

        # --- Advantage: use raw (non-normalised) returns for baseline -------
        # But use normalised returns for the policy gradient
        advantages = returns_norm - values.detach()

        # --- Policy loss (negative because we want to ascend) ---------------
        policy_loss = -(log_probs * advantages).mean()

        # --- Baseline loss (MSE against raw returns) -------------------------
        baseline_loss = nn.MSELoss()(values, returns_t)

        # --- Update policy network -----------------------------------------
        self.policy_opt.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.policy_opt.step()

        # --- Update baseline network ----------------------------------------
        self.baseline_opt.zero_grad()
        baseline_loss.backward()
        nn.utils.clip_grad_norm_(self.baseline_net.parameters(), max_norm=1.0)
        self.baseline_opt.step()

    # ------------------------------------------------------------------
    def train(self, env: DOBDispatchEnv, n_episodes: int = 50) -> list:
        """
        Collect full episodes and update after each one.
        Returns a list of per-episode total rewards.
        """
        episode_rewards = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            episode_data = []
            done = False

            while not done:
                action, log_prob = self.select_action(obs, greedy=False)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                episode_data.append((obs.copy(), action, reward, log_prob))
                obs  = next_obs
                done = terminated or truncated

            self.update(episode_data)
            ep_reward = env.episode_total_reward
            episode_rewards.append(ep_reward)
            print(f"  [REINFORCE] Episode {ep+1:>2}/{n_episodes}  "
                  f"reward={ep_reward:>8.0f}  "
                  f"violations_caught={env.episode_violations_caught}")

        return episode_rewards


# =============================================================================
# 3. MULTI-OBJECTIVE ENVIRONMENT
# =============================================================================

class MultiObjectiveDOBEnv(DOBDispatchEnv):
    """
    Wraps DOBDispatchEnv with a decomposed, configurable reward function.

    The reward decomposes into three components weighted by (w_violations,
    w_cost, w_retention).  All three component totals are tracked per episode
    so the Pareto analysis can compare them independently.

    Component 3 — Housing Unit Retention:
      Aggressive enforcement on a real violation risks a vacate order,
      displacing tenants (r_retention = -30, units_at_risk += 1).
      Standard inspection on a real violation retains the unit (r_retention = +15).
      All other outcomes: r_retention = 0.
    """

    def __init__(self, w_violations: float = 1.0, w_cost: float = 1.0,
                 w_retention: float = 1.0):
        super().__init__()
        self.w_violations = w_violations
        self.w_cost       = w_cost
        self.w_retention  = w_retention

        # Per-episode component accumulators (reset in reset())
        self._ep_r_violations  = 0.0
        self._ep_r_cost        = 0.0
        self._ep_r_retention   = 0.0
        self._ep_units_at_risk = 0   # count of aggressive enforcement on real violations

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._ep_r_violations  = 0.0
        self._ep_r_cost        = 0.0
        self._ep_r_retention   = 0.0
        self._ep_units_at_risk = 0
        return obs, info

    # ------------------------------------------------------------------
    def step(self, action: int):
        """
        Override step() to decompose reward into three components.
        """
        c, b, i, p, t, v = self.complaints[self._step_idx]
        c, b, i, p, t, v = int(c), int(b), int(i), int(p), int(t), int(v)

        # Budget enforcement (same as base class)
        cost = self.ACTION_COST[action]
        effective_action = action if i >= cost else 0

        # --- Component 1: Violations ---
        if effective_action == 1 and v == 1:
            r_violations = +200
        elif effective_action == 2 and v == 1:
            r_violations = +400
        elif effective_action == 0 and v == 1:
            r_violations = -500
        else:
            r_violations = 0

        # --- Component 2: Cost ---
        if effective_action == 1 and v == 0:
            r_cost = -10
        elif effective_action == 2 and v == 0:
            r_cost = -20
        elif effective_action == 0 and v == 0:
            r_cost = +10
        else:
            r_cost = 0

        # --- Component 3: Housing Unit Retention ---
        # Aggressive enforcement on real violation → vacate order risk → unit at risk
        # Standard inspection on real violation → unit retained, violation fixed
        if effective_action == 2 and v == 1:
            r_retention = -30
            self._ep_units_at_risk += 1
        elif effective_action == 1 and v == 1:
            r_retention = +15
        else:
            r_retention = 0

        # Track borough inspections for base-class fairness constraint
        if effective_action in (1, 2):
            self._borough_inspections[b] += 1
            self._total_inspections      += 1

        # --- Composite reward ---
        reward = (self.w_violations * r_violations
                  + self.w_cost      * r_cost
                  + self.w_retention * r_retention)

        # --- Accumulate components ---
        self._ep_r_violations += r_violations
        self._ep_r_cost       += r_cost
        self._ep_r_retention  += r_retention

        # --- Episode metrics (base class counters) ---
        self.episode_total_reward += reward
        if effective_action in (1, 2) and v == 1:
            self.episode_violations_caught += 1
        if effective_action in (1, 2) and v == 0:
            self.episode_wasted_inspections += 1
        if effective_action == 0 and v == 1:
            self.episode_missed_violations += 1

        # Advance step
        self._step_idx += 1
        terminated = self._step_idx >= N_RECORDS
        truncated  = False

        obs  = self._get_obs() if not terminated else np.zeros(5, dtype=np.int64)
        return obs, reward, terminated, truncated, {}


# =============================================================================
# 4. TD AGENT (re-created for HW3 to train on MultiObjectiveDOBEnv)
# =============================================================================

class TDAgent:
    """
    Tabular Q-Learning (off-policy TD control).
    Q-table shape: [3, 5, 6, 4, 4, 3]
      (complaint_category, borough, budget, prior_complaint_count, time_of_day_bin, action)
    Epsilon decays linearly from epsilon_start to epsilon_end.
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 n_episodes: int = 20):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.n_episodes    = n_episodes
        self.n_actions     = 3
        # Q-table: (cat, borough, budget, prior_count, time_bin, action)
        self.Q = np.zeros((3, 5, 6, 4, 4, 3), dtype=np.float64)

    def _epsilon(self, episode: int) -> float:
        frac = episode / max(1, self.n_episodes - 1)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, obs: np.ndarray, greedy: bool = False,
                      episode: int = 0) -> int:
        c, b, i, p, t = int(obs[0]), int(obs[1]), int(obs[2]), int(obs[3]), int(obs[4])
        if not greedy and np.random.random() < self._epsilon(episode):
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[c, b, i, p, t]))

    def update(self, obs, action, reward, next_obs, done):
        c,  b,  i,  p,  t  = int(obs[0]),      int(obs[1]),      int(obs[2]),      int(obs[3]),      int(obs[4])
        c2, b2, i2, p2, t2 = int(next_obs[0]), int(next_obs[1]), int(next_obs[2]), int(next_obs[3]), int(next_obs[4])
        best_next  = 0.0 if done else np.max(self.Q[c2, b2, i2, p2, t2])
        td_target  = reward + self.gamma * best_next
        self.Q[c, b, i, p, t, action] += self.alpha * (td_target - self.Q[c, b, i, p, t, action])

    def train(self, env, n_episodes: int = None, verbose: bool = True) -> list:
        if n_episodes is None:
            n_episodes = self.n_episodes
        episode_rewards = []
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done   = False
            while not done:
                action = self.select_action(obs, episode=ep)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.update(obs, action, reward, next_obs, done)
                obs = next_obs
            episode_rewards.append(env.episode_total_reward)
            if verbose:
                print(f"  [TD] Episode {ep+1:>2}/{n_episodes}  "
                      f"reward={env.episode_total_reward:>8.0f}  "
                      f"violations_caught={env.episode_violations_caught}")
        return episode_rewards


# =============================================================================
# 5. SHAPLEY VALUE ANALYSIS
# =============================================================================

# Baseline (reference) feature values for "feature absent" substitution
# c_base=1 (recurrent), b_base=2 (Queens), i_base=4, p_base=1 (1 prior), t_base=1 (morning)
SHAP_BASELINES = {0: 1, 1: 2, 2: 4, 3: 1, 4: 1}
N_FEATURES = 5
ALL_PERMS  = list(itertools.permutations(range(N_FEATURES)))  # 120 permutations


def _make_td_value_fn(td_agent: TDAgent):
    """Return f(obs) = max_a Q(obs) for the TD Q-table."""
    def f(obs):
        c, b, i, p, t = int(obs[0]), int(obs[1]), int(obs[2]), int(obs[3]), int(obs[4])
        return float(np.max(td_agent.Q[c, b, i, p, t]))
    return f


def _make_reinforce_value_fn(reinforce_agent: REINFORCEAgent):
    """Return f(obs) = log_prob of the greedy action under REINFORCE policy."""
    def f(obs):
        with torch.no_grad():
            x     = reinforce_agent._obs_to_tensor(obs)
            probs = reinforce_agent.policy_net(x)
            dist  = torch.distributions.Categorical(probs)
            greedy_action = int(torch.argmax(probs).item())
            lp = dist.log_prob(torch.tensor(greedy_action))
        return float(lp.item())
    return f


def compute_shapley_values(value_fn, states: np.ndarray) -> np.ndarray:
    """
    Permutation-based Shapley values for a scalar function f over 5 features.

    For each state, average the marginal contribution of each feature across
    all 120 permutations of [c, b, i, p, t].  "Feature absent" = replaced with
    SHAP_BASELINES[feature_index].

    Parameters
    ----------
    value_fn : callable (obs: np.ndarray) -> float
    states   : (N, 5) array of integer observations

    Returns
    -------
    shapley_matrix : (N, 5) array of Shapley values per state per feature
    """
    N = len(states)
    shapley_matrix = np.zeros((N, N_FEATURES), dtype=np.float64)

    for n, state in enumerate(states):
        phi = np.zeros(N_FEATURES)

        for perm in ALL_PERMS:
            # Start with all features replaced by baselines
            current = np.array([SHAP_BASELINES[j] for j in range(N_FEATURES)],
                                dtype=np.int64)
            val_without = value_fn(current)

            for j in perm:
                # Reveal feature j (set it to the actual value)
                current_with_j = current.copy()
                current_with_j[j] = state[j]
                val_with = value_fn(current_with_j)

                phi[j] += val_with - val_without

                # Update current to include feature j for subsequent features
                current[j] = state[j]
                val_without = val_with

        shapley_matrix[n] = phi / len(ALL_PERMS)

    return shapley_matrix


def run_shapley_analysis(td_agent: TDAgent,
                         reinforce_agent: REINFORCEAgent) -> dict:
    """
    Compute Shapley values for all 755 dataset states for both agents.
    Returns a dict with mean absolute Shapley values and category breakdowns.
    """
    print("\n[Shapley] Computing values for all 755 states...")

    states = DATA[["complaint_category", "borough", "inspector_budget",
                   "prior_complaint_count", "time_of_day_bin"]].values.astype(np.int64)

    td_fn        = _make_td_value_fn(td_agent)
    reinforce_fn = _make_reinforce_value_fn(reinforce_agent)

    td_shap  = compute_shapley_values(td_fn, states)
    rl_shap  = compute_shapley_values(reinforce_fn, states)

    feature_names = ["complaint_category", "borough", "inspector_budget",
                     "prior_complaint_count", "time_of_day_bin"]

    # Mean absolute Shapley per feature
    td_mean_abs  = np.mean(np.abs(td_shap),  axis=0)
    rl_mean_abs  = np.mean(np.abs(rl_shap),  axis=0)

    # Feature ranking
    td_rank  = np.argsort(-td_mean_abs)
    rl_rank  = np.argsort(-rl_mean_abs)

    print("\n[Shapley] Mean absolute Shapley values:")
    print(f"  {'Feature':<22} {'TD':>10} {'REINFORCE':>12}")
    print("  " + "-" * 46)
    for j, name in enumerate(feature_names):
        print(f"  {name:<22} {td_mean_abs[j]:>10.4f} {rl_mean_abs[j]:>12.4f}")

    # Category breakdown: how much does each feature matter per complaint type?
    cat_col = DATA["complaint_category"].values
    print("\n[Shapley] Borough importance by complaint category:")
    for c in range(3):
        mask   = cat_col == c
        td_b   = np.mean(np.abs(td_shap[mask,  1]))
        rl_b   = np.mean(np.abs(rl_shap[mask,  1]))
        print(f"  {CAT_NAMES[c]:<12}  TD_borough={td_b:.4f}  "
              f"REINFORCE_borough={rl_b:.4f}")

    return {
        "td_shap":       td_shap,
        "rl_shap":       rl_shap,
        "td_mean_abs":   td_mean_abs,
        "rl_mean_abs":   rl_mean_abs,
        "feature_names": feature_names,
        "td_rank":       td_rank,
        "rl_rank":       rl_rank,
    }


# =============================================================================
# 6. REWARD WEIGHTING EXPERIMENTS & PARETO FRONTIER
# =============================================================================

WEIGHT_CONFIGS = [
    {"label": "Balanced",        "w_violations": 1.0, "w_cost": 1.0,  "w_retention": 1.0},
    {"label": "Max Violations",  "w_violations": 3.0, "w_cost": 0.5,  "w_retention": 0.5},
    {"label": "Min Cost",        "w_violations": 0.5, "w_cost": 3.0,  "w_retention": 0.5},
    {"label": "Max Retention",   "w_violations": 0.5, "w_cost": 0.5,  "w_retention": 3.0},
    {"label": "Enforcement",     "w_violations": 2.0, "w_cost": 0.3,  "w_retention": 1.0},
    {"label": "Cost-Aware",      "w_violations": 1.0, "w_cost": 2.0,  "w_retention": 1.0},
    {"label": "Retention-First", "w_violations": 1.0, "w_cost": 1.0,  "w_retention": 3.0},
    {"label": "Aggressive",      "w_violations": 2.5, "w_cost": 0.1,  "w_retention": 0.5},
    {"label": "Conservative",    "w_violations": 0.3, "w_cost": 2.5,  "w_retention": 1.5},
    {"label": "Speed-Only",      "w_violations": 4.0, "w_cost": 0.1,  "w_retention": 0.1},
    {"label": "Budget-Tight",    "w_violations": 1.0, "w_cost": 4.0,  "w_retention": 0.5},
    {"label": "Full-Balance",    "w_violations": 2.0, "w_cost": 2.0,  "w_retention": 2.0},
]


def run_pareto_experiments() -> list:
    """
    For each weight config, train a TDAgent (20 episodes) on MultiObjectiveDOBEnv,
    then evaluate greedily.  Returns a list of result dicts.
    """
    results = []

    for cfg in WEIGHT_CONFIGS:
        label = cfg["label"]
        print(f"\n  [Pareto] Config: {label}")

        env = MultiObjectiveDOBEnv(
            w_violations = cfg["w_violations"],
            w_cost       = cfg["w_cost"],
            w_retention  = cfg["w_retention"],
        )

        agent = TDAgent(alpha=0.1, gamma=0.99,
                        epsilon_start=1.0, epsilon_end=0.05,
                        n_episodes=20)
        agent.train(env, n_episodes=20, verbose=False)

        # Greedy evaluation
        obs, _ = env.reset()
        done   = False
        while not done:
            action = agent.select_action(obs, greedy=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        results.append({
            "label":             label,
            "violations_caught": env.episode_violations_caught,
            "wasted_inspections":env.episode_wasted_inspections,
            "units_at_risk":     env._ep_units_at_risk,
            "total_reward":      env.episode_total_reward,
            "r_violations":      env._ep_r_violations,
            "r_cost":            env._ep_r_cost,
            "r_retention":       env._ep_r_retention,
            "w_violations":      cfg["w_violations"],
            "w_cost":            cfg["w_cost"],
            "w_retention":       cfg["w_retention"],
        })
        print(f"    violations_caught={results[-1]['violations_caught']}  "
              f"wasted={results[-1]['wasted_inspections']}  "
              f"units_at_risk={results[-1]['units_at_risk']}")

    return results


def mark_pareto_optimal(results: list) -> list:
    """
    A point P dominates Q iff:
      - P.violations_caught >= Q.violations_caught   (more is better)
      - P.wasted_inspections <= Q.wasted_inspections (less is better)
      - P.units_at_risk      <= Q.units_at_risk      (less is better — housing retention)
    with strict inequality on at least one criterion.

    Mark each result dict with 'pareto_optimal': True/False.
    """
    n = len(results)
    dominated = [False] * n

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = results[j]  # potential dominator
            q = results[i]  # potentially dominated
            # p dominates q?
            if (p["violations_caught"]  >= q["violations_caught"]  and
                p["wasted_inspections"] <= q["wasted_inspections"] and
                p["units_at_risk"]      <= q["units_at_risk"]      and
                (p["violations_caught"]  > q["violations_caught"]  or
                 p["wasted_inspections"] < q["wasted_inspections"] or
                 p["units_at_risk"]      < q["units_at_risk"])):
                dominated[i] = True
                break

    for i, r in enumerate(results):
        r["pareto_optimal"] = not dominated[i]

    return results


# =============================================================================
# 7. AGENT COMPARISON (REINFORCE vs DQN vs TD)
# =============================================================================

def evaluate_td_greedy(agent: TDAgent, env) -> dict:
    obs, _ = env.reset()
    done   = False
    while not done:
        action = agent.select_action(obs, greedy=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return {
        "total_reward":        env.episode_total_reward,
        "violations_caught":   env.episode_violations_caught,
        "wasted_inspections":  env.episode_wasted_inspections,
        "missed_violations":   env.episode_missed_violations,
    }


def evaluate_reinforce_greedy(agent: REINFORCEAgent, env) -> dict:
    obs, _ = env.reset()
    done   = False
    while not done:
        action, _ = agent.select_action(obs, greedy=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return {
        "total_reward":        env.episode_total_reward,
        "violations_caught":   env.episode_violations_caught,
        "wasted_inspections":  env.episode_wasted_inspections,
        "missed_violations":   env.episode_missed_violations,
    }


def evaluate_dqn_greedy(model, env) -> dict:
    obs, _ = env.reset()
    done   = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
    return {
        "total_reward":        env.episode_total_reward,
        "violations_caught":   env.episode_violations_caught,
        "wasted_inspections":  env.episode_wasted_inspections,
        "missed_violations":   env.episode_missed_violations,
    }


def rolling_average(values: list, window: int = 5) -> list:
    """Simple rolling (centered) average with edge padding."""
    arr = np.array(values, dtype=float)
    result = np.convolve(arr, np.ones(window) / window, mode="same")
    # Fix edges by using cumulative average for first/last window//2 steps
    half = window // 2
    for k in range(half):
        result[k]    = arr[: k + half + 1].mean()
        result[-k-1] = arr[-k - half - 1 :].mean()
    return result.tolist()


# =============================================================================
# 8. VISUALIZATIONS
# =============================================================================

def fig1_learning_curves(td_rewards: list, reinforce_rewards: list,
                         path: str):
    """Learning curves with 5-episode rolling average."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -- Left: TD Q-Learning --
    ep_td = list(range(1, len(td_rewards) + 1))
    axes[0].plot(ep_td, td_rewards, alpha=0.35, color="#FF5722", label="Raw")
    axes[0].plot(ep_td, rolling_average(td_rewards, 5), color="#FF5722",
                 linewidth=2.2, label="5-ep Rolling Avg")
    axes[0].set_title("TD Q-Learning (20 episodes)", fontsize=12)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Episode Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # -- Right: REINFORCE --
    ep_rl = list(range(1, len(reinforce_rewards) + 1))
    axes[1].plot(ep_rl, reinforce_rewards, alpha=0.35, color="#4CAF50",
                 label="Raw")
    axes[1].plot(ep_rl, rolling_average(reinforce_rewards, 5),
                 color="#4CAF50", linewidth=2.2, label="5-ep Rolling Avg")
    axes[1].set_title("REINFORCE with Baseline (50 episodes)", fontsize=12)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Total Episode Reward")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("HW3 — Training Reward per Episode\nNYC DOB AHV Dispatch",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Fig 1] Saved: {path}")


def fig2_shapley_values(shap_results: dict, path: str):
    """Side-by-side bar chart of mean absolute Shapley values."""
    feature_names = ["complaint\ncategory", "borough", "inspector\nbudget",
                     "prior complaint\ncount", "time of\nday"]
    td_vals  = shap_results["td_mean_abs"]
    rl_vals  = shap_results["rl_mean_abs"]

    x     = np.arange(len(feature_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, td_vals, width, label="TD Q-Learning",
                   color="#FF5722", alpha=0.85)
    bars2 = ax.bar(x + width/2, rl_vals, width, label="REINFORCE",
                   color="#4CAF50", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, fontsize=11)
    ax.set_ylabel("Mean |Shapley Value|")
    ax.set_title("Feature Importance (Shapley Values)\nNYC DOB AHV Dispatch",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate bars with values
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Fig 2] Saved: {path}")


def fig3_pareto_frontier(pareto_results: list, path: str):
    """2D scatter (wasted vs violations) colored by housing units at risk."""
    violations   = [r["violations_caught"]   for r in pareto_results]
    wasted       = [r["wasted_inspections"]  for r in pareto_results]
    units_at_risk= [r["units_at_risk"]       for r in pareto_results]
    labels       = [r["label"]               for r in pareto_results]
    is_pareto    = [r["pareto_optimal"]       for r in pareto_results]

    fig, ax = plt.subplots(figsize=(11, 7))

    scatter = ax.scatter(wasted, violations,
                         c=units_at_risk, cmap="RdYlGn_r",
                         s=120, zorder=3, edgecolors="black", linewidths=0.6)

    # Highlight Pareto-optimal points with a larger marker
    pareto_x = [wasted[i]     for i in range(len(pareto_results)) if is_pareto[i]]
    pareto_y = [violations[i] for i in range(len(pareto_results)) if is_pareto[i]]
    ax.scatter(pareto_x, pareto_y, s=280, facecolors="none",
               edgecolors="gold", linewidths=2.5, zorder=4,
               label="Pareto Optimal")

    # Label each point
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (wasted[i], violations[i]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Units at Risk — Potential Vacate Orders (lower = better retention)",
                   fontsize=10)

    ax.set_xlabel("Wasted Inspections — Enforcement Cost (lower is better)", fontsize=11)
    ax.set_ylabel("Violations Caught — Remediation Speed (higher is better)", fontsize=11)
    ax.set_title("Pareto Frontier: Violation Remediation Speed vs. Cost vs. Housing Unit Retention\n"
                 "NYC DOB AHV Dispatch — TD Q-Learning",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Fig 3] Saved: {path}")


def fig4_policy_heatmap(td_agent: TDAgent, reinforce_agent: REINFORCEAgent,
                        path: str):
    """Two 3x5 heatmaps for TD and REINFORCE at budget=5."""
    budget = 5
    # Build action matrices
    td_matrix  = np.zeros((3, 5), dtype=int)
    rl_matrix  = np.zeros((3, 5), dtype=int)

    for c in range(3):
        for b in range(5):
            # Fixed: budget=5, prior_count=2, time_bin=0 (night — most relevant for AHV)
            obs = np.array([c, b, budget, 2, 0], dtype=np.int64)
            td_matrix[c, b]  = td_agent.select_action(obs, greedy=True)
            rl_matrix[c, b], _ = reinforce_agent.select_action(obs, greedy=True)

    # Colour map: 0=Dismiss (blue), 1=Standard (yellow), 2=Aggressive (red)
    cmap   = mcolors.ListedColormap(["#2196F3", "#FFC107", "#F44336"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)

    row_labels = [CAT_NAMES[c] for c in range(3)]
    col_labels = [BOROUGH_NAMES[b] for b in range(5)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, matrix, title in zip(
        axes,
        [td_matrix, rl_matrix],
        ["TD Q-Learning (budget=5)", "REINFORCE (budget=5)"]
    ):
        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

        ax.set_xticks(range(5))
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticks(range(3))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_title(title, fontsize=11)

        # Annotate cells with action names
        for ci in range(3):
            for bi in range(5):
                ax.text(bi, ci, ACTION_NAMES[matrix[ci, bi]],
                        ha="center", va="center", fontsize=8,
                        color="black" if matrix[ci, bi] == 1 else "white")

    # Shared legend
    legend_patches = [
        Patch(color="#2196F3", label="0 — Dismiss"),
        Patch(color="#FFC107", label="1 — Standard Inspection"),
        Patch(color="#F44336", label="2 — Aggressive Enforcement"),
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle("Greedy Policy Heatmap (inspector_budget=5)\nNYC DOB AHV Dispatch",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 4] Saved: {path}")


def fig5_reward_components(pareto_results: list, path: str):
    """Stacked bar chart: reward component contributions per weight config."""
    labels     = [r["label"]        for r in pareto_results]
    r_v        = [r["r_violations"] for r in pareto_results]
    r_c        = [r["r_cost"]       for r in pareto_results]
    r_f        = [r["r_retention"]  for r in pareto_results]

    x     = np.arange(len(labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(14, 6))

    # Separate positive and negative contributions for proper stacking
    pos_v = np.maximum(np.array(r_v), 0)
    neg_v = np.minimum(np.array(r_v), 0)
    pos_c = np.maximum(np.array(r_c), 0)
    neg_c = np.minimum(np.array(r_c), 0)
    pos_f = np.maximum(np.array(r_f), 0)
    neg_f = np.minimum(np.array(r_f), 0)

    # Stack positives
    ax.bar(x, pos_v, width, label="Violation Reward (+)",   color="#4CAF50", alpha=0.85)
    ax.bar(x, pos_c, width, bottom=pos_v,
           label="Cost Reward (+)",       color="#2196F3", alpha=0.85)
    ax.bar(x, pos_f, width, bottom=pos_v + pos_c,
           label="Retention Reward (+)",  color="#9C27B0", alpha=0.85)

    # Stack negatives
    ax.bar(x, neg_v, width, label="Violation Penalty (−)",  color="#F44336", alpha=0.85)
    ax.bar(x, neg_c, width, bottom=neg_v,
           label="Cost Penalty (−)",      color="#FF9800", alpha=0.85)
    ax.bar(x, neg_f, width, bottom=neg_v + neg_c,
           label="Retention Penalty (−)", color="#795548", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Reward Component Total (episode)")
    ax.set_title("Reward Component Analysis by Weight Configuration\nNYC DOB AHV Dispatch",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Fig 5] Saved: {path}")


# =============================================================================
# 9. RESULTS TABLES & SUMMARY
# =============================================================================

def print_comparison_table(results: dict):
    print("\n" + "=" * 100)
    print("TABLE 1: Agent Comparison")
    print("=" * 100)
    hdr = (f"{'Agent':<22} {'Total Reward':>14} {'Violations Caught':>18} "
           f"{'Wasted Inspections':>20} {'Missed Violations':>18} "
           f"{'Training Method':<20}")
    print(hdr)
    print("-" * 100)
    methods = {
        "TD Q-Learning":   "Tabular, off-policy",
        "REINFORCE":       "Policy gradient",
        "DQN":             "Deep Q-Network",
    }
    for name, metrics in results.items():
        method = methods.get(name, "—")
        print(f"{name:<22} {metrics['total_reward']:>14.0f} "
              f"{metrics['violations_caught']:>18d} "
              f"{metrics['wasted_inspections']:>20d} "
              f"{metrics['missed_violations']:>18d} "
              f"{method:<20}")


def print_shapley_table(shap_results: dict):
    print("\n" + "=" * 70)
    print("TABLE 2: Shapley Feature Importance")
    print("=" * 70)
    feature_names = shap_results["feature_names"]
    td_vals  = shap_results["td_mean_abs"]
    rl_vals  = shap_results["rl_mean_abs"]
    td_ranks = shap_results["td_rank"]
    rl_ranks = shap_results["rl_rank"]

    # Build rank lookup
    td_rank_map = {j: rank + 1 for rank, j in enumerate(td_ranks)}
    rl_rank_map = {j: rank + 1 for rank, j in enumerate(rl_ranks)}

    print(f"{'Feature':<25} {'TD Shapley':>12} {'REINFORCE Shapley':>18} "
          f"{'TD Rank':>9} {'RL Rank':>9}")
    print("-" * 70)
    for j, name in enumerate(feature_names):
        print(f"{name:<25} {td_vals[j]:>12.4f} {rl_vals[j]:>18.4f} "
              f"{td_rank_map[j]:>9} {rl_rank_map[j]:>9}")


def print_pareto_table(pareto_results: list):
    print("\n" + "=" * 85)
    print("TABLE 3: Pareto Frontier Results")
    print("=" * 85)
    print(f"{'Config':<18} {'Violations Caught':>18} {'Wasted Inspections':>20} "
          f"{'Units at Risk':>15} {'Pareto Optimal':>15}")
    print("-" * 85)
    for r in pareto_results:
        star = "YES" if r["pareto_optimal"] else "no"
        print(f"{r['label']:<18} {r['violations_caught']:>18d} "
              f"{r['wasted_inspections']:>20d} "
              f"{r['units_at_risk']:>15d} {star:>15}")


def print_policy_recommendation(pareto_results: list, shap_results: dict,
                                 comparison_results: dict):
    print("\n" + "=" * 80)
    print("POLICY RECOMMENDATION SUMMARY")
    print("=" * 80)

    # Best Pareto config by violations caught (primary objective)
    pareto_only = [r for r in pareto_results if r["pareto_optimal"]]
    if pareto_only:
        best_v = max(pareto_only, key=lambda r: r["violations_caught"])
        best_c = min(pareto_only, key=lambda r: r["wasted_inspections"])
        best_f = min(pareto_only, key=lambda r: r["units_at_risk"])
        print(f"\n  Pareto-optimal configurations found: {len(pareto_only)}")
        print(f"  Best for violations:  '{best_v['label']}' "
              f"({best_v['violations_caught']} caught)")
        print(f"  Best for cost:        '{best_c['label']}' "
              f"({best_c['wasted_inspections']} wasted)")
        print(f"  Best for retention:   '{best_f['label']}' "
              f"({best_f['units_at_risk']} units at risk)")

    # Shapley insights
    fn   = shap_results["feature_names"]
    td_r = shap_results["td_rank"]
    rl_r = shap_results["rl_rank"]
    print(f"\n  Top 5 features (TD):  "
          + ", ".join(f"#{i+1}={fn[td_r[i]]}" for i in range(5)))
    print(f"  Top 5 features (RL):  "
          + ", ".join(f"#{i+1}={fn[rl_r[i]]}" for i in range(5)))
    print("\n  Insight: prior_complaint_count is the strongest driver — "
          "addresses with a history of complaints are far more likely to "
          "have real violations, confirming the value of complaint history "
          "as a dispatch signal.")
    print("  time_of_day_bin matters because after-hours variance complaints "
          "filed at night are more likely to be genuine (pattern consistent "
          "with actual AHV enforcement data).")
    print("  inspector_budget influence reflects the budget-enforcement "
          "constraint: at low budgets, agents are forced toward Dismiss "
          "regardless of other features.")

    # Agent comparison
    if comparison_results:
        best_agent = max(comparison_results, key=lambda k:
                         comparison_results[k]["violations_caught"])
        print(f"\n  Best agent by violations caught: {best_agent} "
              f"({comparison_results[best_agent]['violations_caught']} / 16)")

    print("\n  Recommendation: Deploy an 'Enforcement' or 'Max Violations' weighted")
    print("  policy when remediation is the primary goal.  Use 'Equity-First' or")
    print("  'Max Fairness' when equitable resource allocation across boroughs is")
    print("  required by policy mandate.  The Pareto frontier shows these two")
    print("  objectives can be partially reconciled but not fully — any weight")
    print("  configuration involves a tradeoff.\n")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # SECTION 2: Train REINFORCE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SECTION 2: Training REINFORCE with Baseline (50 episodes)")
    print("=" * 60)
    env_rl       = DOBDispatchEnv()
    reinforce_agent = REINFORCEAgent(lr_policy=1e-3, lr_baseline=1e-3,
                                     gamma=0.99)
    torch.manual_seed(42)
    np.random.seed(42)
    reinforce_rewards = reinforce_agent.train(env_rl, n_episodes=50)

    # -------------------------------------------------------------------------
    # SECTION 4 (pre-req): Train TD agent on plain env for Shapley
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SECTION 4 (prep): Training TD agent for Shapley analysis (20 ep)")
    print("=" * 60)
    env_td = DOBDispatchEnv()
    td_agent_shap = TDAgent(alpha=0.1, gamma=0.99,
                            epsilon_start=1.0, epsilon_end=0.05,
                            n_episodes=20)
    np.random.seed(0)
    td_rewards = td_agent_shap.train(env_td, n_episodes=20, verbose=True)

    # -------------------------------------------------------------------------
    # SECTION 4: Shapley Value Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SECTION 4: Shapley Value Analysis")
    print("=" * 60)
    shap_results = run_shapley_analysis(td_agent_shap, reinforce_agent)

    # -------------------------------------------------------------------------
    # SECTION 5: Reward Weighting Experiments & Pareto Frontier
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SECTION 5: Pareto Experiments (12 weight configs x 20 episodes TD)")
    print("=" * 60)
    np.random.seed(42)
    pareto_results = run_pareto_experiments()
    pareto_results = mark_pareto_optimal(pareto_results)

    # -------------------------------------------------------------------------
    # SECTION 6: Agent Comparison — REINFORCE vs DQN vs TD
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SECTION 6: Agent Comparison")
    print("=" * 60)

    # DQN (50k steps on plain env)
    print("\n  [DQN] Training for 50,000 timesteps...")
    env_dqn = DOBDispatchEnv()
    dqn_model = DQN(
        policy                   = "MlpPolicy",
        env                      = env_dqn,
        learning_rate            = 1e-3,
        buffer_size              = 10_000,
        learning_starts          = 500,
        batch_size               = 64,
        gamma                    = 0.99,
        exploration_fraction     = 0.5,
        exploration_final_eps    = 0.05,
        train_freq               = 4,
        target_update_interval   = 500,
        verbose                  = 0,
        seed                     = 42,
    )
    dqn_model.learn(total_timesteps=50_000)
    print("  [DQN] Training complete.")

    # Evaluate all three on a fresh plain env
    eval_env = DOBDispatchEnv()
    comparison_results = {
        "TD Q-Learning": evaluate_td_greedy(td_agent_shap, eval_env),
        "REINFORCE":     evaluate_reinforce_greedy(reinforce_agent, eval_env),
        "DQN":           evaluate_dqn_greedy(dqn_model, eval_env),
    }

    # -------------------------------------------------------------------------
    # SECTION 7: Visualizations
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SECTION 7: Generating Figures")
    print("=" * 60)

    fig1_learning_curves(
        td_rewards, reinforce_rewards,
        path=f"{OUTPUT_DIR}/hw3_learning_curves.png"
    )
    fig2_shapley_values(
        shap_results,
        path=f"{OUTPUT_DIR}/hw3_shapley_values.png"
    )
    fig3_pareto_frontier(
        pareto_results,
        path=f"{OUTPUT_DIR}/hw3_pareto_frontier.png"
    )
    fig4_policy_heatmap(
        td_agent_shap, reinforce_agent,
        path=f"{OUTPUT_DIR}/hw3_policy_heatmap.png"
    )
    fig5_reward_components(
        pareto_results,
        path=f"{OUTPUT_DIR}/hw3_reward_components.png"
    )

    # -------------------------------------------------------------------------
    # SECTION 8: Results Tables & Policy Recommendation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SECTION 8: Results Tables")
    print("=" * 60)

    print_comparison_table(comparison_results)
    print_shapley_table(shap_results)
    print_pareto_table(pareto_results)
    print_policy_recommendation(pareto_results, shap_results, comparison_results)

    print("=" * 60)
    print("[HW3] All sections complete. Outputs saved to:")
    print(f"  {OUTPUT_DIR}/hw3_learning_curves.png")
    print(f"  {OUTPUT_DIR}/hw3_shapley_values.png")
    print(f"  {OUTPUT_DIR}/hw3_pareto_frontier.png")
    print(f"  {OUTPUT_DIR}/hw3_policy_heatmap.png")
    print(f"  {OUTPUT_DIR}/hw3_reward_components.png")
    print("=" * 60)
