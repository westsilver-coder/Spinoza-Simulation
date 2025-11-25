"""
Spinoza Simulation: Statistical Verification & Analysis
=======================================================

This script performs a Monte Carlo simulation to statistically verify the robustness and dynamics of the Spinoza agent-based model.
It runs multiple simulations (batch processing) without visualization to collect data on average Conatus (power), population survival, and inequality (Gini coefficient).
It then generates four key analytical plots:
1. Convergence Analysis: Checks if the results stabilize as the population size (N) increases.
2. Sensitivity Analysis: Identifies the 'Tipping Point' where system collapse becomes likely due to disaster intensity.
3. Monte Carlo Summary: Shows the mean trajectory and variance of 100 simulation runs.
4. Inequality Dynamics: Tracks the Gini coefficient over time to observe wealth/power distribution trends.

Dependencies:
- numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings

# Suppress RuntimeWarnings (e.g., division by zero when population is 0) for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# [1] Experiment Configuration & Constants
# ==========================================
NUM_SIMULATIONS = 100   # Number of Monte Carlo runs for statistical significance
ROUNDS = 300            # Duration of each simulation run (time steps)
N_AGENTS = 100          # Default population size
THRESH_COLLAPSE = 0.2   # Threshold below which a run is considered 'Collapsed'
THRESH_STABLE = 0.8     # Threshold above which a run is considered 'Stable'

# ==========================================
# [2] Helper Functions
# ==========================================

def calculate_gini_batch(agents_conatus):
    """
    Calculates the Gini Coefficient for a batch of agents.
    Gini index measures inequality (0.0 = perfect equality, 1.0 = perfect inequality).
    """
    # Handle empty or zero-sum cases to avoid errors
    if len(agents_conatus) == 0 or np.sum(agents_conatus) == 0: 
        return 0.0
    
    sorted_conatus = np.sort(agents_conatus)
    n = len(agents_conatus)
    index = np.arange(1, n + 1)
    # Gini formula using sorted values
    return ((np.sum((2 * index - n - 1) * sorted_conatus)) / (n * np.sum(sorted_conatus)))

def get_ci95(data, axis=0):
    """
    Calculates the Mean and the 95% Confidence Interval (CI).
    Returns: mean, lower_bound, upper_bound
    Uses Standard Error (SE) * 1.96 for 95% CI.
    """
    # Use nanmean/nanstd to handle potential NaN values safely
    mean = np.nanmean(data, axis=axis)
    std = np.nanstd(data, axis=axis)
    n = data.shape[axis]
    se = std / np.sqrt(n) # Standard Error
    return mean, mean - 1.96 * se, mean + 1.96 * se

# ==========================================
# [3] Simulation Engine (Optimized for Speed)
# ==========================================

def run_simulation_batch(n_sims=100, n_agents=100, rounds=300, disaster_intensity=1.0):
    """
    Runs a batch of simulations with specified parameters.
    Returns history arrays for Average Conatus and Gini Coefficient.
    """
    # Arrays to store history for all runs
    history_avg = np.zeros((n_sims, rounds))
    history_gini = np.zeros((n_sims, rounds))
    
    for s in range(n_sims):
        # Initialize Agents (Vectorized)
        # Columns: 0:Reason, 1:Joy, 2:Sadness, 3:Metabolism, 4:Recovery, 5:Generosity, 6:Alive(1/0)
        agents = np.zeros((n_agents, 7))
        agents[:, 0] = np.random.beta(2, 2, n_agents) # Reason (Beta distribution)
        agents[:, 1] = np.random.uniform(0.3, 1.0, n_agents) # Initial Joy
        agents[:, 2] = np.random.uniform(0.0, 0.3, n_agents) # Initial Sadness
        agents[:, 3] = np.random.uniform(0.005, 0.02, n_agents) # Metabolism (Energy decay)
        agents[:, 4] = np.random.uniform(0.01, 0.05, n_agents) # Recovery rate
        agents[:, 5] = np.clip(agents[:, 0] + np.random.uniform(-0.2, 0.2, n_agents), 0.1, 1.0) # Generosity
        agents[:, 6] = 1 # Alive status
        
        # Event tracking variables
        event_streak = 0
        last_event = None
        global_meta_mult = 1.0
        global_recov_mult = 1.0

        for r in range(rounds):
            alive_mask = agents[:, 6] == 1
            alive_count = np.sum(alive_mask)
            
            # Stop if population is extinct
            if alive_count == 0: 
                history_avg[s, r:] = 0
                history_gini[s, r:] = 0
                break

            # 1. Natural Flux (Entropy & Random Fluctuation)
            # Apply metabolism multiplier (influenced by Innovation event)
            effective_meta = agents[alive_mask, 3] * np.clip(global_meta_mult, 0.5, 2.0)
            agents[alive_mask, 1] -= effective_meta 
            agents[alive_mask, 1] += np.random.uniform(-0.05, 0.05, alive_count) 

            # 2. Conatus Calculation
            # Conatus = (Joy - Sadness / Resilience) + Base Vitality (0.5)
            resilience = 1.0 + (agents[alive_mask, 0] * 1.5)
            conatus = (agents[alive_mask, 1] - (agents[alive_mask, 2] / resilience)) + 0.5
            
            # 3. Event System (Probabilistic)
            if random.random() < 0.1: # 10% chance of event per round
                event = random.choice(['DISASTER', 'BOOM', 'EPIDEMIC', 'INNOVATION'])
                
                # Combo system: Consecutive events increase multiplier
                if event == last_event: event_streak += 1
                else: event_streak = 1
                last_event = event
                
                multiplier = min(3.0, (1.0 + np.log1p(0.5 * (event_streak - 1))))
                
                if event == 'DISASTER':
                    # Scale impact by disaster_intensity parameter for sensitivity analysis
                    impact = np.random.uniform(0.4, 0.8, alive_count) * multiplier * disaster_intensity
                    agents[alive_mask, 1] *= (1.0 - np.clip(impact, 0, 0.95)) # Reduce Joy
                    agents[alive_mask, 2] += impact * 0.5 # Increase Sadness
                elif event == 'BOOM':
                    boost = np.random.uniform(0.2, 0.6, alive_count) * multiplier
                    agents[alive_mask, 1] += boost
                elif event == 'EPIDEMIC':
                    global_recov_mult *= 0.8 # Permanent recovery penalty
                elif event == 'INNOVATION':
                    global_meta_mult *= 0.9 # Permanent metabolism improvement
            
            # Parameter regression to mean
            if global_recov_mult > 1.0: global_recov_mult *= 0.99
            if global_meta_mult < 1.0: global_meta_mult *= 1.01

            # 4. Recovery & Death Logic
            # Crisis: Conatus < 0.6
            crisis = conatus < 0.6
            roll = np.random.rand(alive_count)
            # Recovery depends on Reason (Reason > Luck)
            recover = crisis & (roll < agents[alive_mask, 0]) 
            
            eff_recov = agents[alive_mask, 4] * np.clip(global_recov_mult, 0.2, 2.0)
            joy_temp = agents[alive_mask, 1]
            joy_temp[recover] += eff_recov[recover]
            agents[alive_mask, 1] = joy_temp
            
            # Recalculate Conatus for death check
            conatus_final = (agents[alive_mask, 1] - (agents[alive_mask, 2] / resilience)) + 0.5
            
            # Death Check
            dead_indices = np.where(alive_mask)[0][conatus_final <= 0]
            agents[dead_indices, 6] = 0
            
            # 5. Birth (Population Maintenance)
            # If population drops below 50%, attempt to spawn new agents
            if np.sum(agents[:, 6]) < n_agents * 0.5 and random.random() < 0.1:
                dead_idx = np.where(agents[:, 6] == 0)[0]
                if len(dead_idx) > 0:
                    # Respawn: New agent with fresh random stats
                    agents[dead_idx[0], :] = [np.random.beta(2,2), 0.5, 0.1, 0.01, 0.03, 0.5, 1]

            # 6. Record Data
            if np.sum(agents[:, 6]) > 0:
                survivor_conatus = conatus_final[conatus_final > 0]
                # Clip outliers for cleaner stats
                survivor_conatus = np.clip(survivor_conatus, 0, 3.0)
                history_avg[s, r] = np.mean(survivor_conatus)
                history_gini[s, r] = calculate_gini_batch(survivor_conatus)
            else:
                history_avg[s, r] = 0
                history_gini[s, r] = 0

    return history_avg, history_gini

# ==========================================
# [4] Visualization & Analysis Reporting
# ==========================================
try:
    import matplotlib
    matplotlib.use('TkAgg')
except:
    pass

# Setup Figure Layout
fig = plt.figure(figsize=(16, 12), constrained_layout=True)
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.4, wspace=0.3)

# --- Panel A: Convergence Analysis ---
ax1 = fig.add_subplot(gs[0, 0])
N_LIST = [50, 100, 200]
COLORS = ['red', 'green', 'blue']

print("1. Running Convergence Check...")
for n, col in zip(N_LIST, COLORS):
    data_c, _ = run_simulation_batch(n_sims=50, n_agents=n, rounds=300)
    mean, lower, upper = get_ci95(data_c)
    ax1.plot(mean, color=col, linewidth=2, label=f'N={n}')
    ax1.fill_between(range(300), lower, upper, color=col, alpha=0.15)

ax1.set_title("(A) Convergence Analysis (95% CI)", fontsize=14, fontweight='bold')
ax1.set_ylabel("Avg Conatus", fontsize=12)
ax1.set_xlabel("Time Steps", fontsize=12)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

# --- Panel B: Tipping Point (Sensitivity) ---
ax2 = fig.add_subplot(gs[0, 1])
print("2. Analyzing Tipping Point...")
# Dense sampling around the critical region (1.2 - 1.8)
intensities = np.concatenate([np.linspace(0.5, 1.2, 4), np.linspace(1.2, 1.8, 8), np.linspace(1.8, 2.5, 4)])
collapse_probs = []
errors = []

for intensity in intensities:
    d_c, _ = run_simulation_batch(n_sims=40, rounds=200, disaster_intensity=intensity)
    # Collapse Definition: Final avg conatus < THRESH_COLLAPSE
    collapsed = (d_c[:, -1] < THRESH_COLLAPSE).astype(int)
    prob = np.mean(collapsed)
    err = np.sqrt(prob * (1-prob) / 40) # Standard Error of Proportion
    collapse_probs.append(prob)
    errors.append(err)

ax2.errorbar(intensities, collapse_probs, yerr=errors, fmt='-o', color='purple', capsize=4, label='Collapse Prob ± SE')
ax2.axvline(x=1.5, color='red', linestyle='--', alpha=0.5, label='Critical Threshold')
ax2.axvspan(1.3, 1.7, color='red', alpha=0.1, label='Transition Zone')
ax2.set_title("(B) Sensitivity: Disaster Intensity vs Collapse", fontsize=14, fontweight='bold')
ax2.set_xlabel("Disaster Multiplier", fontsize=12)
ax2.set_ylabel("Probability", fontsize=12)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)

# --- Panel C: Monte Carlo Summary ---
ax3 = fig.add_subplot(gs[1, 0])
print("3. Final Monte Carlo Run...")
data_c, data_g = run_simulation_batch(n_sims=100, n_agents=100, rounds=300)
mean, lower, upper = get_ci95(data_c)

# Plot individual runs (faintly)
for i in range(15):
    ax3.plot(data_c[i], color='gray', alpha=0.1, linewidth=0.5)

ax3.plot(mean, color='navy', linewidth=2.5, label='Mean Conatus')
ax3.fill_between(range(300), lower, upper, color='blue', alpha=0.3, label='95% CI')

# Summary Text Box
final_vals = data_c[:, -1]
stats_text = (f"N_Sims: 100\n"
              f"Survival: {np.sum(final_vals>0)}%\n"
              f"Stable(>0.8): {np.sum(final_vals>THRESH_STABLE)}\n"
              f"Collapsed(<0.2): {np.sum(final_vals<THRESH_COLLAPSE)}")
ax3.text(0.03, 0.05, stats_text, transform=ax3.transAxes, 
         bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'), fontsize=10)

ax3.set_title("(C) Monte Carlo Summary (Conatus)", fontsize=14, fontweight='bold')
ax3.set_xlabel("Time Steps", fontsize=12)
ax3.set_ylabel("Conatus", fontsize=12)
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.3)

# --- Panel D: Inequality Dynamics ---
ax4 = fig.add_subplot(gs[1, 1])
mean_g, lower_g, upper_g = get_ci95(data_g)

ax4.plot(mean_g, color='darkred', linewidth=2.5, label='Mean Gini Index')
ax4.fill_between(range(300), lower_g, upper_g, color='red', alpha=0.2, label='95% CI')

ax4.set_title("(D) Inequality Dynamics (Gini Coefficient)", fontsize=14, fontweight='bold')
ax4.set_xlabel("Time Steps", fontsize=12)
ax4.set_ylabel("Gini Index", fontsize=12)
ax4.legend(loc='upper right', fontsize=10)
ax4.grid(True, alpha=0.3)

# Layout Adjustment & Display
plt.tight_layout(pad=4.0) 
print("Analysis Complete. Displaying plot...")
plt.show()