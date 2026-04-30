"""
RL Housing Inspection: NYC DOB After-Hours Variance (AHV) Complaint Response
=============================================================================
An RL agent acting as an optimal NYC DOB manager responding to AHV 311 complaints.

Sections:
  1. Imports
  2. Data Generation
  3. Environment
  4. MC Agent (First-Visit Monte Carlo)
  5. TD Agent (Tabular Q-Learning)
  6. DQN Agent (stable-baselines3)
  7. Baselines (Random, Always-Inspect)
  8. Training
  9. Evaluation
 10. Results Table
 11. Learning Curve Plot
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend so it works headlessly
import matplotlib.pyplot as plt
from collections import defaultdict
from stable_baselines3 import DQN
import warnings
warnings.filterwarnings("ignore")

# Output paths
OUTPUT_DIR = "/Users/alihasan/Downloads/Reinforcement Learning"
PLOT_PATH  = f"{OUTPUT_DIR}/rl_learning_curve.png"

# =============================================================================
# 2. DATA GENERATION
# =============================================================================
def generate_synthetic_data(n_records: int = 755, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic AHV complaint records matching the report's description:
      - 755 hourly records from NYC Open Data (AHV complaint type)
      - Exactly 16 true violations (~2.1%), concentrated in recurrent & high-freq types
      - Fields: complaint_category, borough, hour_of_day, inspector_budget,
                prior_complaint_count, time_of_day_bin, is_violation
        * complaint_category: 0=brand-new, 1=recurrent, 2=high-frequency
        * borough: 0-4 (Manhattan, Brooklyn, Queens, Bronx, Staten Island)
        * inspector_budget: per-step availability drawn from data (mostly 4-5)
        * prior_complaint_count: 0=none, 1=one prior, 2=two-three priors, 3=four+ priors
        * time_of_day_bin: 0=night(10pm-6am), 1=morning(6am-12pm),
                           2=afternoon(12pm-6pm), 3=evening(6pm-10pm)
    """
    rng = np.random.default_rng(seed)

    # Complaint category — roughly equal thirds
    complaint_category = rng.integers(0, 3, size=n_records)

    # Borough — roughly uniform across 5 boroughs
    borough = rng.integers(0, 5, size=n_records)

    # Hour of day — uniform
    hour_of_day = rng.integers(0, 24, size=n_records)

    # Inspector budget per step: mostly 4-5 (high availability), rarely 0-1
    inspector_budget = rng.choice([3, 4, 5, 5, 5], size=n_records)

    # Prior complaint count: correlated with complaint category
    # Brand-new → mostly 0-1 priors; recurrent → 1-2; high-freq → 2-3
    prior_complaint_count = np.zeros(n_records, dtype=int)
    for cat, pool in [(0, [0, 0, 1, 1, 2]),
                      (1, [1, 1, 2, 2, 3]),
                      (2, [2, 2, 3, 3, 3])]:
        idx = np.where(complaint_category == cat)[0]
        prior_complaint_count[idx] = rng.choice(pool, size=len(idx))

    # Time of day bin derived from hour_of_day
    # 0=night(22-5), 1=morning(6-11), 2=afternoon(12-17), 3=evening(18-21)
    time_of_day_bin = np.where(
        (hour_of_day >= 22) | (hour_of_day <= 5), 0,
        np.where((hour_of_day >= 6)  & (hour_of_day <= 11), 1,
        np.where((hour_of_day >= 12) & (hour_of_day <= 17), 2, 3))
    ).astype(int)

    # Place exactly 16 violations, concentrated in recurrent (1) and high-freq (2)
    # 2 in brand-new, 6 in recurrent, 8 in high-frequency (matching ~2% rate)
    is_violation = np.zeros(n_records, dtype=int)

    recurrent_idx  = np.where(complaint_category == 1)[0]
    highfreq_idx   = np.where(complaint_category == 2)[0]
    brandnew_idx   = np.where(complaint_category == 0)[0]

    chosen = np.concatenate([
        rng.choice(brandnew_idx,  2, replace=False),
        rng.choice(recurrent_idx, 6, replace=False),
        rng.choice(highfreq_idx,  8, replace=False),
    ])
    is_violation[chosen] = 1

    df = pd.DataFrame({
        "complaint_category":   complaint_category,
        "borough":              borough,
        "hour_of_day":          hour_of_day,
        "inspector_budget":     inspector_budget,
        "prior_complaint_count":prior_complaint_count,
        "time_of_day_bin":      time_of_day_bin,
        "is_violation":         is_violation,
    })
    return df


# Generate data once; all agents share the same fixed sequence
DATA = generate_synthetic_data(n_records=755, seed=42)
N_RECORDS = len(DATA)

print(f"[Data] {N_RECORDS} records | "
      f"{DATA['is_violation'].sum()} violations "
      f"({100*DATA['is_violation'].mean():.1f}%)")
print(DATA["complaint_category"].value_counts().sort_index()
      .rename({0:"brand-new",1:"recurrent",2:"high-freq"}).to_string())
print()


# =============================================================================
# 3. ENVIRONMENT
# =============================================================================
class DOBDispatchEnv(gym.Env):
    """
    Custom Gymnasium environment for NYC DOB AHV complaint dispatch.

    State  : MultiDiscrete([3, 5, 6, 4, 4])
               s[0] = complaint_category    c ∈ {0,1,2}
               s[1] = borough               b ∈ {0,1,2,3,4}
               s[2] = inspector_budget      i ∈ {0,1,2,3,4,5}
               s[3] = prior_complaint_count p ∈ {0,1,2,3}
               s[4] = time_of_day_bin       t ∈ {0,1,2,3}

    Action : Discrete(3)
               0 = Dismiss / Deprioritize  (cost 0)
               1 = Standard Inspection     (cost 1)
               2 = Aggressive Enforcement  (cost 2)

    Reward : defined by (action, is_violation) pair with fairness and
             budget-enforcement constraints.
    """

    metadata = {"render_modes": []}

    # Action costs
    ACTION_COST = {0: 0, 1: 1, 2: 2}

    # Base rewards
    REWARD_TABLE = {
        (0, 0): +10,   # correct dismissal
        (0, 1): -500,  # critical failure — missed real violation
        (1, 0): -10,   # wasted standard inspection
        (1, 1): +200,  # standard inspection catches violation
        (2, 0): -20,   # wasted aggressive enforcement
        (2, 1): +400,  # aggressive enforcement catches violation
    }

    # Fairness: >40% of all inspections in one borough → penalty per trigger
    FAIRNESS_THRESHOLD = 0.40
    FAIRNESS_PENALTY   = -20

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.MultiDiscrete([3, 5, 6, 4, 4])
        self.action_space      = spaces.Discrete(3)

        # Fixed complaint sequence (shared by all agents)
        self.complaints = DATA[["complaint_category", "borough",
                                "inspector_budget", "prior_complaint_count",
                                "time_of_day_bin", "is_violation"]].values

        # Episode state
        self._reset_episode()

    # ------------------------------------------------------------------
    def _reset_episode(self):
        self._step_idx            = 0
        self._borough_inspections = np.zeros(5, dtype=int)  # for fairness check
        self._total_inspections   = 0

        # Per-episode tracking
        self.episode_total_reward       = 0
        self.episode_violations_caught  = 0
        self.episode_wasted_inspections = 0
        self.episode_missed_violations  = 0

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_episode()
        obs = self._get_obs()
        return obs, {}

    # ------------------------------------------------------------------
    def _get_obs(self):
        c, b, i, p, t, _ = self.complaints[self._step_idx]
        return np.array([int(c), int(b), int(i), int(p), int(t)], dtype=np.int64)

    # ------------------------------------------------------------------
    def step(self, action: int):
        c, b, i, p, t, v = self.complaints[self._step_idx]
        c, b, i, p, t, v = int(c), int(b), int(i), int(p), int(t), int(v)

        cost = self.ACTION_COST[action]

        # Budget enforcement: if current step's inspector supply can't cover the
        # action cost, force a dismiss. With per-step budgets mostly 4-5, this is
        # rarely triggered (only when action=2 and budget=1 or budget=0 for action=1).
        effective_action = action
        if i < cost:
            effective_action = 0

        # Base reward
        reward = self.REWARD_TABLE[(effective_action, v)]

        # Track borough inspections for fairness constraint
        if effective_action in (1, 2):
            self._borough_inspections[b] += 1
            self._total_inspections      += 1

        # Fairness penalty check
        if self._total_inspections > 0:
            max_share = self._borough_inspections.max() / self._total_inspections
            if max_share > self.FAIRNESS_THRESHOLD:
                reward += self.FAIRNESS_PENALTY

        # Episode metrics
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

        obs      = self._get_obs() if not terminated else np.zeros(5, dtype=np.int64)
        info     = {}
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self):
        pass  # Not implemented


# =============================================================================
# 4. MC AGENT (First-Visit Monte Carlo)
# =============================================================================
class MCAgent:
    """
    First-Visit Monte Carlo control with epsilon-greedy policy.

    Q(s, a) is updated as a running average over first-visit returns G_t.
    """

    def __init__(self, gamma: float = 0.99, epsilon: float = 0.1):
        self.gamma   = gamma
        self.epsilon = epsilon
        self.n_actions = 3

        # Q-values and visit counts indexed by (c, b, i, a)
        self.Q       = defaultdict(float)
        self.returns_count = defaultdict(int)

    # ------------------------------------------------------------------
    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        s = tuple(obs)
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        # Greedy w.r.t. current Q
        q_vals = [self.Q[(s, a)] for a in range(self.n_actions)]
        return int(np.argmax(q_vals))

    # ------------------------------------------------------------------
    def update(self, episode: list):
        """
        episode: list of (obs, action, reward) tuples for one full episode.
        Perform first-visit MC update.
        """
        # Compute discounted returns from the end
        G = 0.0
        visited = set()

        for t in reversed(range(len(episode))):
            obs, action, reward = episode[t]
            s  = tuple(obs)
            sa = (s, action)
            G  = self.gamma * G + reward

            # First-visit check
            if sa not in visited:
                visited.add(sa)
                self.returns_count[sa] += 1
                # Incremental mean update
                n = self.returns_count[sa]
                self.Q[sa] += (G - self.Q[sa]) / n

    # ------------------------------------------------------------------
    def train(self, env: DOBDispatchEnv, n_episodes: int = 20) -> list:
        """Train for n_episodes; return list of per-episode total rewards."""
        episode_rewards = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            episode_data = []
            done = False

            while not done:
                action = self.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                episode_data.append((obs.copy(), action, reward))
                obs  = next_obs
                done = terminated or truncated

            self.update(episode_data)
            episode_rewards.append(env.episode_total_reward)
            print(f"  [MC] Episode {ep+1:>2}/{n_episodes}  "
                  f"reward={env.episode_total_reward:>8.0f}  "
                  f"violations_caught={env.episode_violations_caught}")

        return episode_rewards


# =============================================================================
# 5. TD AGENT (Tabular Q-Learning)
# =============================================================================
class TDAgent:
    """
    Tabular Q-Learning (off-policy TD control).

    Q-table shape: [3, 5, 6, 3]  (c, b, i, a)
    Epsilon decays linearly from epsilon_start to epsilon_end over all episodes.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float   = 0.05,
        n_episodes: int      = 20,
    ):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.n_episodes    = n_episodes
        self.n_actions     = 3

        # Q-table: (complaint_cat, borough, budget, prior_count, time_bin, action)
        self.Q = np.zeros((3, 5, 6, 4, 4, 3), dtype=np.float64)

    # ------------------------------------------------------------------
    def _epsilon(self, episode: int) -> float:
        """Linear epsilon decay."""
        frac = episode / max(1, self.n_episodes - 1)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    # ------------------------------------------------------------------
    def select_action(self, obs: np.ndarray, greedy: bool = False,
                      episode: int = 0) -> int:
        c, b, i, p, t = int(obs[0]), int(obs[1]), int(obs[2]), int(obs[3]), int(obs[4])
        if not greedy and np.random.random() < self._epsilon(episode):
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[c, b, i, p, t]))

    # ------------------------------------------------------------------
    def update(self, obs, action, reward, next_obs, done):
        c,  b,  i,  p,  t  = int(obs[0]),      int(obs[1]),      int(obs[2]),      int(obs[3]),      int(obs[4])
        c2, b2, i2, p2, t2 = int(next_obs[0]), int(next_obs[1]), int(next_obs[2]), int(next_obs[3]), int(next_obs[4])

        best_next = 0.0 if done else np.max(self.Q[c2, b2, i2, p2, t2])
        td_target = reward + self.gamma * best_next
        td_error  = td_target - self.Q[c, b, i, p, t, action]
        self.Q[c, b, i, p, t, action] += self.alpha * td_error

    # ------------------------------------------------------------------
    def train(self, env: DOBDispatchEnv, n_episodes: int = None) -> list:
        """Train for n_episodes; return list of per-episode total rewards."""
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
            print(f"  [TD] Episode {ep+1:>2}/{n_episodes}  "
                  f"reward={env.episode_total_reward:>8.0f}  "
                  f"violations_caught={env.episode_violations_caught}")

        return episode_rewards


# =============================================================================
# 6. DQN AGENT (stable-baselines3)
# =============================================================================
def train_dqn(env: DOBDispatchEnv, total_timesteps: int = 50_000) -> DQN:
    """
    Train a DQN agent using stable-baselines3.

    - MlpPolicy
    - epsilon: 1.0 → 0.05 over first 50% of training (exploration_fraction=0.5)
    """
    # SB3 requires the env to be wrapped so reset() returns a flat array.
    # Our env already returns np.array of shape (3,) so it's compatible.
    model = DQN(
        policy              = "MlpPolicy",
        env                 = env,
        learning_rate       = 1e-3,
        buffer_size         = 10_000,
        learning_starts     = 500,
        batch_size          = 64,
        gamma               = 0.99,
        exploration_fraction      = 0.5,
        exploration_final_eps     = 0.05,
        train_freq          = 4,
        target_update_interval   = 500,
        verbose             = 0,
        seed                = 42,
    )
    print(f"  [DQN] Training for {total_timesteps:,} timesteps …")
    model.learn(total_timesteps=total_timesteps)
    print("  [DQN] Training complete.")
    return model


# =============================================================================
# 7. BASELINES
# =============================================================================
def run_random_baseline(env: DOBDispatchEnv) -> dict:
    """Random action each step (evaluation only)."""
    obs, _ = env.reset()
    done   = False
    while not done:
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return _collect_metrics(env)


def run_always_inspect_baseline(env: DOBDispatchEnv) -> dict:
    """Always take action 1 (Standard Inspection)."""
    obs, _ = env.reset()
    done   = False
    while not done:
        obs, _, terminated, truncated, _ = env.step(1)
        done = terminated or truncated
    return _collect_metrics(env)


# =============================================================================
# 8. TRAINING
# =============================================================================
def _collect_metrics(env: DOBDispatchEnv) -> dict:
    return {
        "total_reward":        env.episode_total_reward,
        "violations_caught":   env.episode_violations_caught,
        "wasted_inspections":  env.episode_wasted_inspections,
        "missed_violations":   env.episode_missed_violations,
    }


def evaluate_agent(agent, env: DOBDispatchEnv, agent_type: str = "tabular") -> dict:
    """
    Run one full greedy episode through all 755 records.

    agent_type: "mc" | "td" | "dqn"
    """
    obs, _ = env.reset()
    done   = False
    while not done:
        if agent_type == "mc":
            action = agent.select_action(obs, greedy=True)
        elif agent_type == "td":
            action = agent.select_action(obs, greedy=True)
        elif agent_type == "dqn":
            action, _ = agent.predict(obs, deterministic=True)
            action    = int(action)
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}")

        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    return _collect_metrics(env)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    # Shared environment instance (reset between runs)
    env = DOBDispatchEnv()

    # --- Train MC ---
    print("\n=== Training: First-Visit Monte Carlo ===")
    mc_agent   = MCAgent(gamma=0.99, epsilon=0.1)
    np.random.seed(0)                          # reproducibility for exploration
    mc_rewards = mc_agent.train(env, n_episodes=20)

    # --- Train TD ---
    print("\n=== Training: Tabular Q-Learning (TD) ===")
    td_agent   = TDAgent(alpha=0.1, gamma=0.99,
                         epsilon_start=1.0, epsilon_end=0.05, n_episodes=20)
    np.random.seed(1)
    td_rewards = td_agent.train(env, n_episodes=20)

    # --- Train DQN ---
    print("\n=== Training: DQN (stable-baselines3) ===")
    dqn_env   = DOBDispatchEnv()               # fresh env for SB3
    dqn_model = train_dqn(dqn_env, total_timesteps=50_000)

    # ==========================================================================
    # 9. EVALUATION
    # ==========================================================================
    print("\n=== Evaluation (greedy / deterministic) ===")

    np.random.seed(99)
    results = {}

    results["Random Baseline"] = run_random_baseline(env)
    results["Always-Inspect"]  = run_always_inspect_baseline(env)
    results["MC (First-Visit)"]= evaluate_agent(mc_agent,  env, agent_type="mc")
    results["TD (Q-Learning)"] = evaluate_agent(td_agent,  env, agent_type="td")
    results["DQN (MLP Policy)"]= evaluate_agent(dqn_model, env, agent_type="dqn")

    # ==========================================================================
    # 10. RESULTS TABLE
    # ==========================================================================
    print("\n=== Results Table ===")
    header = (
        f"{'Agent':<22}"
        f"{'Total Reward':>14}"
        f"{'Violations Caught':>20}"
        f"{'Wasted Inspections':>20}"
        f"{'Missed Violations':>20}"
    )
    print(header)
    print("-" * len(header))
    for agent_name, metrics in results.items():
        print(
            f"{agent_name:<22}"
            f"{metrics['total_reward']:>14.0f}"
            f"{metrics['violations_caught']:>20d}"
            f"{metrics['wasted_inspections']:>20d}"
            f"{metrics['missed_violations']:>20d}"
        )

    # ==========================================================================
    # 11. LEARNING CURVE PLOT
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(9, 5))
    episodes = range(1, len(mc_rewards) + 1)

    ax.plot(episodes, mc_rewards, marker="o", label="MC (First-Visit)", color="#2196F3")
    ax.plot(episodes, td_rewards, marker="s", label="TD (Q-Learning)",  color="#FF5722")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    ax.set_title("Training Reward per Episode\nNYC DOB AHV Dispatch RL Agents",
                 fontsize=13)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Total Episode Reward", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"\n[Plot] Learning curve saved to: {PLOT_PATH}")

    # ==========================================================================
    # POLICY ANALYSIS
    # ==========================================================================
    ACTION_NAMES = {0: "Dismiss", 1: "Standard Insp.", 2: "Aggressive Enf."}
    CAT_NAMES    = {0: "brand-new", 1: "recurrent", 2: "high-freq"}
    BOROUGH_NAMES= {0: "Manhattan", 1: "Brooklyn", 2: "Queens",
                    3: "Bronx", 4: "Staten Is."}

    def mc_greedy_action(obs):
        s = tuple(obs)
        q_vals = [mc_agent.Q[(s, a)] for a in range(3)]
        return int(np.argmax(q_vals))

    print("\n=== Policy Analysis (budget=5, greedy) ===")
    print(f"{'':12}", end="")
    for b in range(5):
        print(f"{BOROUGH_NAMES[b]:>14}", end="")
    print()

    for agent_label, agent_fn in [
        ("MC",  lambda obs: mc_agent.select_action(obs, greedy=True)),
        ("TD",  lambda obs: td_agent.select_action(obs, greedy=True)),
    ]:
        print(f"\n--- {agent_label} Policy ---")
        for c in range(3):
            print(f"  {CAT_NAMES[c]:<10}", end="")
            for b in range(5):
                # Fixed: budget=5, prior_count=2, time_bin=0 (night)
                obs    = np.array([c, b, 5, 2, 0], dtype=np.int64)
                action = agent_fn(obs)
                print(f"{ACTION_NAMES[action]:>14}", end="")
            print()

    print("\n--- DQN Policy ---")
    for c in range(3):
        print(f"  {CAT_NAMES[c]:<10}", end="")
        for b in range(5):
            obs    = np.array([c, b, 5, 2, 0], dtype=np.int64)
            action, _ = dqn_model.predict(obs, deterministic=True)
            print(f"{ACTION_NAMES[int(action)]:>14}", end="")
        print()

    print("\nDone.")
