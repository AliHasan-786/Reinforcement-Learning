# Project Explainer: NYC Housing Inspection with Reinforcement Learning
## Everything You Need to Know — From Zero

---

## Part 1: What Is This Project Actually About?

### The Real-World Problem

New York City gets thousands of **311 complaints** every year. People call 311 when they think a landlord is doing illegal construction — working after hours, not having permits, etc. These complaints are called **AHV complaints** (After-Hours Variance).

The NYC Department of Buildings (DOB) has inspectors. When a complaint comes in, they have to decide: **send an inspector, or ignore it?**

The problem: only about **2.1% of complaints are real violations**. The other 97.9% are either noise, mistakes, or non-issues. If you send an inspector every time someone calls 311, you:
- Waste enormous resources (inspector time, money)
- Over-police certain neighborhoods (unfair)
- Still miss some real violations

So the question is: **can a computer learn to make smarter dispatch decisions?**

### What Is Reinforcement Learning (RL)?

Imagine training a dog. You don't explain the rules to it in words. Instead:
- The dog does something
- You give it a treat (reward) or say "no" (punishment)
- Over thousands of tries, the dog learns what behaviors get treats

Reinforcement Learning works exactly the same way, but for a computer program called an **agent**:
- The agent looks at the current **state** (what's happening right now)
- It takes an **action** (does something)
- The environment gives it a **reward** (a number — positive or negative)
- The agent learns which actions lead to more reward over time

The agent doesn't know the rules. It just tries things repeatedly and figures out patterns.

---

## Part 2: The Dataset (311 Service Requests)

### Where the data comes from

The file `311_Service_Requests_from_2020_to_Present_20260304.csv` is real NYC 311 data. But because the actual violation outcomes weren't labeled in that data, the code generates **synthetic data calibrated to match real statistics**:

- **755 records** — representing 755 individual AHV complaints
- **16 real violations** — only 2.1% are actual violations (matching real-world rates)
- Each complaint has 5 features (see below)

### The 5 Features (State Variables)

Each complaint is described by 5 pieces of information:

| Feature | Values | What it means |
|---|---|---|
| `complaint_category` | 0, 1, 2 | 0 = brand new complaint, 1 = recurrent (same building complained before), 2 = high-frequency (many complaints from this neighborhood) |
| `borough` | 0–4 | Which NYC borough: Manhattan, Brooklyn, Queens, Bronx, Staten Island |
| `inspector_budget` | 0–5 | How many inspectors are available right now (like staffing levels) |
| `prior_complaint_count` | 0–3 | How many times this address type has been complained about before (0=none, 1=once, 2=2-3 times, 3=four or more) |
| `time_of_day_bin` | 0–3 | When the complaint was filed: 0=night (10pm–6am), 1=morning (6am–12pm), 2=afternoon (12pm–6pm), 3=evening (6pm–10pm) |

**Why these 2 new features?**
- `prior_complaint_count`: Repeat offenders. If a building has been complained about many times before, it's more likely to have real violations. This is domain-relevant knowledge.
- `time_of_day_bin`: "After-Hours Variance" complaints are most meaningful when filed at night — that's when illegal construction is most suspicious. Night complaints are more likely to be real.

### Why are there only 16 violations?

Because that's reality. Most 311 calls are not real violations. The RL agent has to learn to find those 16 needles in a haystack of 739 non-violations.

---

## Part 3: The RL Setup (The "Game" the Agent Plays)

### MDP — The formal structure

This is set up as an **MDP (Markov Decision Process)**. Don't be intimidated — it just means:

- **State (S)**: What the agent sees at each step → the 5 features above
- **Action (A)**: What the agent can do
- **Reward (R)**: Score the agent gets for each action
- **Policy (π)**: The agent's strategy — given this state, what action do I take?

### The 3 Actions

| Action | What it means |
|---|---|
| 0 = Dismiss | Don't send an inspector, ignore the complaint |
| 1 = Standard Inspection | Send an inspector for a regular check |
| 2 = Aggressive Enforcement | Send inspector with authority to issue heavy penalties |

### The Reward Function (How the agent is scored)

This is the most important design decision in the whole project. Every reward is a number:

| Situation | Reward | Why |
|---|---|---|
| Correctly dismissed a non-violation | **+10** | Good — saved resources |
| Standard inspection caught a real violation | **+200** | Good — found the bad guy |
| Aggressive enforcement caught a real violation | **+400** | Very good — stopped it hard |
| Sent inspector but it wasn't a violation | **−10 or −20** | Bad — wasted resources |
| Dismissed a real violation (missed it) | **−500** | Very bad — critical failure |
| Aggressive enforcement on real violation | **−30** | Vacate order risk — tenant displacement |
| Standard inspection on real violation | **+15** | Unit retained and repaired |

The **−500 penalty** for missing a real violation is huge. This is intentional — you really don't want to miss real violations.

---

## Part 4: The Literature Review (Background Reading)

Before building anything, the class reviewed existing academic work. Here's what each paper was about and why it matters:

### Paper 1: Agent-Based Model for Informal Housing (Algeria)
- **What it is**: Researchers built a simulation of informal housing in an Algerian city using "agents" (not RL, but similar idea)
- **Why it matters**: Shows that computational models can help urban planners understand how housing violations spread
- **Connection to your project**: You're also using agent-based simulation, but applied to NYC enforcement

### Paper 2: Statistical Approaches for Disparate Impact in Fair Housing
- **What it is**: Legal/statistical framework for proving that a policy discriminates against protected groups
- **Why it matters**: Your fairness penalty (−20 for over-policing one borough) is directly motivated by this — you don't want your AI to accidentally target Black or Hispanic neighborhoods
- **Connection to your project**: The housing unit retention dimension in your Pareto frontier is motivated by this — aggressive enforcement can displace the very tenants you're trying to protect

### Paper 3: Integrated City Data + ML for Housing Public Health
- **What it is**: Using multiple city datasets (not just 311) to predict which buildings have health/housing violations before anyone complains
- **Why it matters**: Shows that ML can be proactive, not just reactive
- **Connection to your project**: Your agent is reactive (it responds to complaints). This paper argues for predictive systems — a future direction

### Paper 4: What Information Matters in Tenant Screening
- **What it is**: Study of how landlords use credit reports, eviction history, etc. to screen tenants
- **Why it matters**: Shows the information asymmetry problem — landlords have info advantage over tenants and regulators
- **Connection to your project**: Your complaint_category feature partially captures this (recurrent complaints = pattern of bad behavior by landlord)

---

## Part 5: Assignment 2 — What Was Built

### The Three Agents (HW2)

HW2 implemented three different algorithms that all play the same game:

---

#### Agent 1: Monte Carlo (MC)

**Concept**: Play the game many times, record what happened, average the results.

**Analogy**: Imagine you're learning to play poker. You play 1,000 hands, write down every decision and whether it won or lost, then go back and calculate: "On average, did raising in this situation make money?"

**How it works technically**:
1. Play a full episode (all 755 complaints) using random actions
2. At the end, look back at every state-action pair you encountered
3. Calculate the total reward you got after each state (the "return")
4. Update your estimates: "State X + Action Y → average return of Z"
5. Repeat thousands of times

**Weakness**: Very slow to learn. Needs many full episodes. Also didn't work great here because violations are so rare.

---

#### Agent 2: Tabular Q-Learning (TD) — THE WINNER

**Concept**: Learn as you go, step by step.

**Analogy**: Imagine you're driving to work and learning which routes are fast. After each turn, you adjust your estimate of how good that turn was — you don't wait until you reach work to update your knowledge.

**The Q-Table**: A giant lookup table. For every possible (state, action) combination, it stores a number: "How good is it to take action A in state S?"

**How it works technically**:
1. Look at current state → pick action (sometimes randomly to explore)
2. Get reward
3. Update Q-table: `Q(s,a) ← Q(s,a) + α × (reward + γ × max Q(next_state) - Q(s,a))`
   - α (alpha) = learning rate (how fast to update, like 0.1)
   - γ (gamma) = discount factor (how much to value future rewards, like 0.99)
4. The "Bellman equation" above is the core of Q-learning

**Why it won**: With only 5 discrete features and small value ranges, you can build a complete lookup table. The agent stores **exactly** what it learned about each specific (category, borough, budget, prior count, time) combination.

**Results**: Found **16/16 violations**, +4,710 total reward.

---

#### Agent 3: DQN (Deep Q-Network)

**Concept**: Instead of a lookup table, use a neural network to predict Q-values.

**Why needed**: When states are continuous (infinite possibilities), you can't have a lookup table for every possible state. A neural network can generalize.

**Analogy**: Instead of memorizing "Manhattan + recurrent + budget 5 → inspect", a neural network learns "if borough is X and budget is high, generally inspect."

**The "Deep" part**: Uses multiple layers of artificial neurons (like a simplified brain).

**Why it FAILED here**:
- The problem is too **imbalanced** (97.9% non-violations)
- The neural network just learned "always dismiss" — it gets +10 reward for every correct dismissal and rarely sees the +200 violation reward
- It's like trying to learn to recognize unicorns when you've only ever seen horses
- **Result**: 0/16 violations caught, −610 total reward

---

### The Baselines (for comparison)

- **Random**: Pick actions randomly → caught 12/16 violations but wasted tons of inspections
- **Always-Inspect**: Send inspector to everyone → caught all 16 but at enormous cost (739 wasted inspections, −4,230 reward)

---

## Part 6: Assignment 3 — What Was Built

### New Algorithm: REINFORCE with Baseline

**Concept**: Instead of learning Q-values (how good is each action?), directly learn a **policy** (probability of taking each action given state).

**Analogy**: TD Q-Learning learns "action A has value 200 in state S." REINFORCE instead learns "in state S, there's an 80% chance I should inspect and 20% dismiss." It learns probabilities directly.

**The Two Networks**:

1. **Policy Network (π)**: Input: state (5 numbers) → Output: probability of each action (3 probabilities that sum to 1)
   - Architecture: 5 inputs → 64 neurons → 64 neurons → 3 outputs
   - Uses ReLU activation (basically: "if negative, set to zero; if positive, keep it")
   - Uses Softmax at end (converts raw numbers to probabilities)

2. **Baseline Network (V)**: Input: state → Output: single number (estimated value of being in this state)
   - Same architecture but outputs 1 number instead of 3

**Why two networks?** The baseline reduces variance in training. Without it, the policy gets inconsistent gradient signals. It's like: instead of saying "that action was good" you say "that action was better than average."

**The Math (simplified)**:
- After each episode, calculate the total discounted reward from each step: G_t
- Subtract the baseline: Advantage = G_t - V(s_t)
- Update policy to make high-advantage actions more probable
- Formula: `θ ← θ + α × Advantage × gradient`

**Why it ALSO FAILED**: Same problem as DQN — class imbalance. The neural policy just learned to always dismiss.

---

### Shapley Values — Feature Importance

**What are Shapley values?**

Shapley values come from **cooperative game theory** (Nobel Prize economics concept). Imagine 5 workers (the 5 features) are splitting credit for a team's success. How much credit does each worker deserve?

Shapley values give each feature a fair share of credit based on: **how much does the prediction change when I include vs. exclude this feature?**

**The method used**:
1. For a given prediction, try every possible ordering of the 5 features
2. For each ordering, see how much the prediction changes when you add feature j
3. Average across all orderings

With 5 features, there are 5! = 120 possible orderings — the code enumerates all 120 for every state.

**"Masking" a feature**: When you "remove" a feature, you replace it with its baseline value. For complaint_category, baseline is 1 (recurrent). For borough, it's 2 (Queens). For budget, it's 4. For prior_complaint_count, it's 1. For time_of_day_bin, it's 1 (morning).

**Actual Results (from the code)**:

| Rank | Feature | Shapley Value (TD) | Meaning |
|---|---|---|---|
| 1 | inspector_budget | 15.55 | Dominant — budget constraints force actions regardless of complaint context |
| 2 | prior_complaint_count | 10.11 | Strong — history of complaints = more likely to be real |
| 3 | time_of_day_bin | 8.10 | Night complaints are more suspicious for AHV violations |
| 4 | complaint_category | 7.51 | The complaint type itself |
| 5 | borough | 6.62 | Where in NYC |

**What this means practically**: Budget is the #1 factor because when the inspector budget is 0 or 1, the agent is *forced* to dismiss regardless of everything else. Among the predictive features, prior complaint count is most informative — repeat offenders are more likely to have real violations.

---

### Multi-Objective Optimization and the Pareto Frontier

**The problem**: Your reward function has competing goals:
1. **Catch violations fast** (violation remediation speed)
2. **Save resources** (minimize wasted inspections = cost)
3. **Preserve housing** (housing unit retention — don't displace tenants)

These goals conflict. Catching violations aggressively might displace tenants via vacate orders. Saving money might mean missing violations. The three objectives can't all be maximized simultaneously.

**Housing unit retention explained**: When an inspector uses Aggressive Enforcement (action 2) on a real violation, they might issue a **vacate order** — forcing tenants to leave while repairs are made. This is necessary for safety but displaces people. Standard inspection (action 1) is less aggressive, fixes the violation, and lets tenants stay. So: more aggressive enforcement = faster remediation but more displacement. This is the housing unit retention tradeoff.

**What the Pareto frontier is**:

Imagine a graph where x-axis is "cost" (wasted inspections) and y-axis is "violations caught." A **Pareto optimal** solution is one where you can't improve one objective without making another worse.

Example:
- Config A: 16 violations, 257 wasted inspections
- Config B: 16 violations, 200 wasted inspections
- Config A is NOT Pareto optimal because Config B is strictly better (same violations, fewer wasted inspections)

The set of all non-dominated solutions forms the "Pareto frontier."

**The 12 configurations tested**: Different weight vectors (w_v, w_c, w_r) for violation, cost, and retention components. Training a fresh Q-Learning agent under each.

**Actual Results**:
- Only **3 out of 12** configs are Pareto-optimal
- **Min Cost** config achieves 16/16 violations with only 41 wasted inspections and 6 units at risk
- **Max Retention** config achieves 16/16 violations with only 2 units at risk — the best for tenant protection
- **Conservative** config: lowest cost (23 wasted) but misses 3 violations (13/16)
- Cost and housing unit retention are **irreconcilable**: to minimize cost you use less aggressive enforcement, but this simultaneously keeps tenants housed (good retention) — so actually they push in the same direction for some configs
- The real tradeoff is: **remediation speed vs. housing unit retention** — catching more violations faster requires more aggressive enforcement, which increases displacement risk

---

## Part 7: The Code Structure

### File 1: `rl_housing_inspection.py` (HW2)

```
Section 1: Imports (libraries needed)
Section 2: Data Generation (creates the 755 synthetic complaints)
Section 3: DOBDispatchEnv (the Gymnasium environment — the "game")
Section 4: Monte Carlo Agent
Section 5: TD Q-Learning Agent
Section 6: DQN Agent (uses stable-baselines3 library)
Section 7: Baselines (Random, Always-Inspect)
Section 8: Training Loop
Section 9: Evaluation
Section 10: Results Table
Section 11: Learning Curve Plot
```

### File 2: `rl_housing_hw3.py` (HW3)

```
Section 1: Imports + seeds
Section 2: REINFORCE Agent (PolicyNet + BaselineNet classes)
Section 3: Multi-Objective Environment (wrapper around HW2 environment)
Section 4: Shapley Value Analysis (manual permutation calculation)
Section 5: Reward Weighting Experiments + Pareto Frontier
Section 6: Agent Comparisons
Section 7: Visualizations (5 PNG figures)
Section 8: Results Tables
```

### File 3: `generate_hw3_report.py` (HW3)

This file generates the PDF. It uses **reportlab** (a Python library) to programmatically create a formatted academic report. It's not an algorithm — it's just layout/styling code that puts text, tables, and images into a PDF.

---

## Part 8: Key Libraries Used

| Library | What it does |
|---|---|
| `numpy` | Fast math with arrays (used everywhere for numerical computation) |
| `pandas` | Spreadsheet-like data manipulation (loading and filtering the dataset) |
| `gymnasium` | Framework for creating RL environments (defines the "game rules") |
| `torch` (PyTorch) | Neural network library (used for REINFORCE policy/baseline networks) |
| `stable_baselines3` | Pre-built RL algorithms (used for DQN) |
| `matplotlib` | Plotting (all the figures) |
| `reportlab` | PDF generation (creates the final report) |

---

## Part 9: The 5 Figures Generated

| Figure | File | What it shows |
|---|---|---|
| Learning Curves | `hw3_learning_curves.png` | Training reward per episode for TD vs REINFORCE — TD goes up, REINFORCE stays flat |
| Shapley Values | `hw3_shapley_values.png` | Bar chart of feature importance for each agent |
| Pareto Frontier | `hw3_pareto_frontier.png` | Scatter plot of cost vs. violations with housing units at risk as color — shows the tradeoff space |
| Reward Components | `hw3_reward_components.png` | Stacked bar chart of how each reward component contributes across configs |
| Policy Heatmap | `hw3_policy_heatmap.png` | Grid showing what action each agent takes in each state — TD is colorful, REINFORCE is all blue (always dismiss) |

---

## Part 10: HW3 Requirements Check (FINAL STATUS)

### What the assignment asks for vs. what was done:

| Requirement | Status | Notes |
|---|---|---|
| Use data from HW2 | ✅ Done | Same 755-record synthetic dataset, expanded to 5 features |
| REINFORCE with baseline as two-layer neural network | ✅ Done | PolicyNet(input_dim=5) + BaselineNet(input_dim=5), both 2-layer MLPs |
| Compare policy gradient vs. DQN | ✅ Done | Table 1 in report |
| Calculate Shapley values for **all features** | ✅ Done | All 5 features computed via all 120 permutations (5!) |
| **Select top 5** | ✅ Fixed | Added prior_complaint_count and time_of_day_bin as new features. Now exactly 5 features → top 5 = all 5 |
| Experiment with different reward weightings | ✅ Done | 12 configurations tested |
| Plot Pareto frontier (speed, cost, **housing unit retention**) | ✅ Fixed | Third axis is now housing units at risk from aggressive enforcement vacate orders |
| 2-3 pages methodology and results | ✅ Fixed | Policy recommendations section removed, intro condensed, report substantially trimmed |

**Submitted March 26, 2026.**

---

## Part 11: Glossary of Technical Terms

| Term | Plain English Definition |
|---|---|
| MDP (Markov Decision Process) | A formal mathematical model of decision-making: state → action → reward → next state |
| Episode | One full run through all 755 complaints |
| Q-value | A number representing "how good is action A in state S?" |
| Q-table | A lookup table storing Q-values for every (state, action) pair |
| Policy | The agent's strategy — given a state, what action to take |
| Reward | A score given after each action (positive = good, negative = bad) |
| Discount factor (γ) | How much future rewards are worth vs. immediate rewards (0.99 means value the future almost as much as now) |
| Learning rate (α) | How fast to update estimates (0.1 = small updates, more stable) |
| Exploration vs. Exploitation | Should I try new actions (explore) or stick with what I know works (exploit)? |
| Neural network | A mathematical model loosely inspired by the brain — layers of numbers connected by weights |
| Gradient | Direction of steepest increase in a function — used to update neural network weights |
| Backpropagation | Algorithm for computing gradients efficiently in neural networks |
| Softmax | A function that converts a list of numbers into probabilities that sum to 1 |
| ReLU | Activation function: max(0, x) — introduces nonlinearity into neural networks |
| Shapley value | Fair credit attribution for each feature's contribution to a model's prediction |
| Pareto frontier | Set of solutions where no objective can be improved without worsening another |
| Class imbalance | When one outcome (violation) is much rarer than another (non-violation) |
| Gymnasium | Python library for creating standardized RL environments (like a physics simulator for RL) |
| On-policy | Learning from actions taken by the current policy (REINFORCE) |
| Off-policy | Learning from any experience, regardless of policy (Q-Learning, DQN) |
| Monte Carlo | Method that estimates values by averaging over many random episodes |
| Temporal Difference (TD) | Method that updates estimates after each step, not after full episodes |
