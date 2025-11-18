# main.py
# Train and compare Q-Learning vs SARSA on the custom intersection environment

import os
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt

from environment import IntersectionEnv
from q_learning import QLearningAgent
from sarsa import SARSAAgent


# =====================================================
# CONFIGURATION
# =====================================================

CONFIG = {
    "env": {
        "max_steps": 300,
        "initial_phase": 0,
        "arrival_probs": {"N": 0.35, "S": 0.35, "E": 0.30, "W": 0.30},
        "depart_rate": 2,
        "queue_cap": 50,
        "seed": 7,
        "alpha_wait": 1.0,
        "alpha_queue": 0.2,
        "beta_throughput": 0.7,
        "switch_penalty": 0.0,
        "animation": False,
    },
    "training": {
        "episodes": 100,
        "eval_interval": 10,
    },
    "q_learning": {
        "alpha": 0.15,
        "gamma": 0.95,
        "epsilon": 0.3,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.02,
        "seed": 42,
        "discretize_cap": 20,
    },
    "sarsa": {
        "alpha": 0.15,
        "gamma": 0.95,
        "epsilon": 0.3,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.02,
        "seed": 123,
        "discretize_cap": 20,
    },
    "results_dir": "results",
}


# =====================================================
# UTILITY
# =====================================================

def ensure_results_dir(path: str):
    """Create results directory if missing."""
    os.makedirs(path, exist_ok=True)


# =====================================================
# RUN EPISODE (FOR BOTH ALGORITHMS)
# =====================================================

def run_episode(env: IntersectionEnv, agent, algorithm: str) -> Dict[str, float]:
    """Runs a single episode for Q-learning OR SARSA."""
    
    state, _ = env.reset()
    done = False
    
    if algorithm == "sarsa":
        action = agent.select_action(state)

    while not done:
        if algorithm == "q_learning":
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

        elif algorithm == "sarsa":
            next_state, reward, done, info = env.step(action)
            next_action = agent.select_action(next_state)
            agent.update(state, action, reward, next_state, next_action, done)
            state, action = next_state, next_action

        else:
            raise ValueError("Unknown RL algorithm.")

    agent.decay_epsilon()

    return env.episode_metrics()


# =====================================================
# TRAINING FUNCTION (USED BY STREAMLIT)
# =====================================================

def train_and_collect(config: Dict):
    """Train both Q-learning and SARSA agents and collect metrics."""
    
    env_q = IntersectionEnv(**config["env"])
    env_s = IntersectionEnv(**config["env"])

    q_agent = QLearningAgent(**config["q_learning"])
    s_agent = SARSAAgent(**config["sarsa"])

    episodes = config["training"]["episodes"]

    q_avg_wait, s_avg_wait = [], []
    q_avg_queue, s_avg_queue = [], []
    q_throughput, s_throughput = [], []
    q_total_reward, s_total_reward = [], []

    for ep in range(1, episodes + 1):
        q_metrics = run_episode(env_q, q_agent, algorithm="q_learning")
        s_metrics = run_episode(env_s, s_agent, algorithm="sarsa")

        q_avg_wait.append(q_metrics["average_waiting_time"])
        s_avg_wait.append(s_metrics["average_waiting_time"])

        q_avg_queue.append(q_metrics["average_queue_length"])
        s_avg_queue.append(s_metrics["average_queue_length"])

        q_throughput.append(q_metrics["throughput"])
        s_throughput.append(s_metrics["throughput"])

        q_total_reward.append(q_metrics["total_reward"])
        s_total_reward.append(s_metrics["total_reward"])

        if ep % config["training"]["eval_interval"] == 0:
            print(
                f"[Episode {ep}] "
                f"Q-Learning → wait={q_avg_wait[-1]:.2f}, queue={q_avg_queue[-1]:.2f}, thr={q_throughput[-1]:.0f}, R={q_total_reward[-1]:.2f} | "
                f"SARSA → wait={s_avg_wait[-1]:.2f}, queue={s_avg_queue[-1]:.2f}, thr={s_throughput[-1]:.0f}, R={s_total_reward[-1]:.2f}"
            )

    return (
        {"q_learning": q_avg_wait, "sarsa": s_avg_wait},
        {"q_learning": q_avg_queue, "sarsa": s_avg_queue},
        {"q_learning": q_throughput, "sarsa": s_throughput},
        {"q_learning": q_total_reward, "sarsa": s_total_reward},
    )


# =====================================================
# PLOTTING FUNCTION (USED BY STREAMLIT)
# =====================================================

def plot_metric_inline(metric_dict: Dict[str, List[float]], title: str, ylabel: str):
    """Plot a metric inline (Streamlit supported)."""
    plt.figure(figsize=(8, 5))
    for label, series in metric_dict.items():
        plt.plot(series, label=label)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(plt)
plt.close()



# =====================================================
# FULL TRAINING (OPTIONAL LOCAL USE)
# =====================================================

def run_all():
    ensure_results_dir(CONFIG["results_dir"])

    avg_wait, avg_queue, throughput, total_reward = train_and_collect(CONFIG)

    plot_metric_inline(avg_wait, "Average Waiting Time per Episode", "Avg Wait")
    plot_metric_inline(avg_queue, "Average Queue Length per Episode", "Avg Queue")
    plot_metric_inline(total_reward, "Total Reward", "Reward")
    plot_metric_inline(throughput, "Throughput per Episode", "Vehicles Departed")

    print("Training and comparison complete.")
