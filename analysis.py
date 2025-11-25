"""
Spinoza Simulation: Statistical Verification & Analysis
=======================================================

This script performs a Monte Carlo simulation to statistically verify the robustness and dynamics of the Spinoza agent-based model.
It mirrors the logic of the visual simulation (main.py), including Age-based death and Logistic population growth, but runs in 'headless' mode for high-speed batch processing.

The script generates four key analytical plots:
1. Convergence Analysis: Checks if results stabilize as population size (N) increases.
2. Sensitivity Analysis: Identifies the 'Tipping Point' of system collapse regarding disaster intensity.
3. Monte Carlo Summary: Shows the mean trajectory and variance of 100 simulation runs.
4. Inequality Dynamics: Tracks the Gini coefficient over time.

Dependencies:
- numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings

# Suppress RuntimeWarnings (e.g., division by zero) for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# [1] Experiment Configuration
# ==========================================
NUM_SIMULATIONS = 100   # Number of Monte Carlo runs
ROUNDS = 300            # Duration of each run
N_AGENTS = 100          # Default population size
THRESH_COLLAPSE = 0.2   # Threshold for 'Collapsed' state
THRESH_STABLE = 0.8     # Threshold for 'Stable' state

# ==========================================
# [2] Helper Functions
# ==========================================

def calculate_gini_batch(agents_conatus):
    """Calculates Gini Coefficient for inequality analysis."""
    if len(agents_conatus) == 0 or np.sum(agents_conatus) == 0: 
        return 0.0
    sorted_conatus = np.sort(agents_conatus)
    n = len(agents_conatus)
    index = np.arange(1, n + 1)
    return ((np.sum((2 * index - n - 1) * sorted_conatus)) / (n * np.sum(sorted_conatus)))

def get_ci95(data, axis=0):
    """Calculates Mean and 95% Confidence Interval."""
    mean = np.nanmean(data, axis=axis)
    std = np.nanstd(data, axis=axis)
    n = data.shape[axis]
    se = std / np.sqrt(n)
    return mean, mean - 1.96 * se, mean + 1.96 * se

# ==========================================
# [3] Simulation Engine (Optimized with NumPy)
# ==========================================

def run_simulation_batch(n_sims=100, n_agents=100, rounds=300, disaster_intensity=1.0):
    """
    Runs batch simulations.
    Syncs logic with main.py:
    - Relational Entropy
    - Age-based Death (Gompertz law)
    - Logistic Birth Growth
    - Cumulative Event Impacts
    """
    # Data Containers
    history_avg = np.zeros((n_sims, rounds))
    history_gini = np.zeros((n_sims, rounds))
    
    for s in range(n_sims):
        # Initialize Agents Data Structure (Columns)
        # 0:Reason, 1:Joy, 2:Sadness, 3:Metabolism, 4:Recovery, 5:Generosity, 6:Alive, 7:Age
        agents = np.zeros((n_agents, 8)) 
        
        # Set initial values (Same distribution as main.py)
        agents[:, 0] = np.random.beta(2, 2, n_agents)           # Reason
        agents[:, 1] = np.random.uniform(0.4, 0.8, n_agents)    # Joy (Vitality)
        agents[:, 2] = np.random.uniform(0.0, 0.2, n_agents)    # Sadness
        agents[:, 3] = np.random.uniform(0.001, 0.005, n_agents)# Metabolism (Entropy)
        agents[:, 4] = np.random.uniform(0.02, 0.08, n_agents)  # Recovery Rate
        # Generosity correlated with Reason
        agents[:, 5] = np.clip(agents[:, 0] + np.random.uniform(-0.2, 0.2, n_agents), 0.1, 1.0)
        agents[:, 6] = 1  # Alive status (1=Alive, 0=Dead)
        agents[:, 7] = 0  # Age (starts at 0)
        
        # Event State
        event_streak = 0
        last_event = None
        global_meta_mult = 1.0
        global_recov_mult = 1.0
        event_active = False # Track if an event effect is active for birth logic

        for r in range(rounds):
            alive_mask = agents[:, 6] == 1
            alive_count = np.sum(alive_mask)
            
            # Stop if extinction occurs
            if alive_count == 0: 
                history_avg[s, r:] = 0
                history_gini[s, r:] = 0
                break

            # --- 1. Natural Flux & Aging ---
            agents[alive_mask, 7] += 1 # Increment Age
            
            # Apply Entropy (Metabolism)
            effective_meta = agents[alive_mask, 3] * np.clip(global_meta_mult, 0.5, 2.0)
            agents[alive_mask, 1] -= effective_meta 
            agents[alive_mask, 1] += np.random.uniform(-0.02, 0.02, alive_count) # Random fluctuation

            # --- 2. Conatus Calculation ---
            # Formula: (Joy - Sadness / Resilience) + Base(0.5)
            resilience = 1.0 + (agents[alive_mask, 0] * 1.5)
            # Using 0.5 as approximate base vitality for vectorized calculation
            conatus = (agents[alive_mask, 1] - (agents[alive_mask, 2] / resilience)) + 0.5
            
            # --- 3. Event System (Cumulative Effects) ---
            event_active = False
            last_event_type_for_birth = None

            if random.random() < 0.1: # 10% event chance
                event = random.choice(['DISASTER', 'BOOM', 'EPIDEMIC', 'INNOVATION'])
                event_active = True
                last_event_type_for_birth = event

                if event == last_event: event_streak += 1
                else: event_streak = 1
                last_event = event
                
                # Cumulative Multiplier (Max 3.0x)
                multiplier = min(3.0, (1.0 + np.log1p(0.5 * (event_streak - 1))))
                
                if event == 'DISASTER':
                    # Impact scales with intensity parameter
                    impact = np.random.uniform(0.2, 0.5, alive_count) * multiplier * disaster_intensity
                    agents[alive_mask, 1] *= (1.0 - np.clip(impact * 0.6, 0, 0.95)) # Joy loss
                    agents[alive_mask, 2] += impact * 0.5 # Sadness gain
                elif event == 'BOOM':
                    boost = np.random.uniform(0.2, 0.6, alive_count) * multiplier
                    agents[alive_mask, 1] += boost
                elif event == 'EPIDEMIC':
                    global_recov_mult *= 0.8 # Permanent resilience damage
                elif event == 'INNOVATION':
                    global_meta_mult *= 0.9 # Permanent efficiency gain
            
            # Parameter Regression (Return to normal over time)
            if global_recov_mult > 1.0: global_recov_mult *= 0.99
            if global_meta_mult < 1.0: global_meta_mult *= 1.01

            # --- 4. Recovery & Death Logic ---
            
            # Crisis Recovery
            crisis = conatus < 0.6
            roll = np.random.rand(alive_count)
            recover = crisis & (roll < agents[alive_mask, 0]) # Chance based on Reason
            
            eff_recov = agents[alive_mask, 4] * np.clip(global_recov_mult, 0.2, 2.0)
            joy_temp = agents[alive_mask, 1]
            joy_temp[recover] += eff_recov[recover]
            agents[alive_mask, 1] = joy_temp
            
            # Recalculate Conatus
            conatus_final = (agents[alive_mask, 1] - (agents[alive_mask, 2] / resilience)) + 0.5
            
            # [New] Age-based Death Probability (Gompertz Law)
            ages = agents[alive_mask, 7]
            age_prob = 0.0001 + ((np.maximum(0, ages - 50) / 50.0) ** 4) * 0.1
            death_roll = np.random.rand(alive_count)
            
            # Death Condition: Conatus <= 0 OR Bad Luck (Age)
            deaths = (conatus_final <= 0) | (death_roll < age_prob)
            
            # Apply Death
            current_indices = np.where(alive_mask)[0]
            dead_indices = current_indices[deaths]
            agents[dead_indices, 6] = 0
            
            # --- 5. Birth Logic (Logistic Growth) ---
            current_alive = np.sum(agents[:, 6])
            pop_ratio = current_alive / n_agents
            
            # Base birth probability based on population density
            birth_prob = 0.0
            if pop_ratio < 0.95:
                if pop_ratio < 0.2: birth_prob = 0.2
                else: birth_prob = 0.1 * (1.0 - pop_ratio)
            
            # Event impact on birth
            if event_active:
                if last_event_type_for_birth in ['BOOM', 'INNOVATION']:
                    birth_prob += 0.2
                elif last_event_type_for_birth in ['DISASTER', 'EPIDEMIC']:
                    birth_prob *= 0.1

            # Attempt Birth
            if random.random() < birth_prob:
                dead_idx = np.where(agents[:, 6] == 0)[0]
                if len(dead_idx) > 0:
                    # Respawn with new random stats (Age 0)
                    # Params: Reason, Joy, Sadness, Meta, Recov, Generosity, Alive, Age
                    agents[dead_idx[0], :] = [np.random.beta(2,2), 0.6, 0.1, 0.003, 0.05, 0.5, 1, 0]

            # --- 6. Record Stats ---
            if np.sum(agents[:, 6]) > 0:
                survivor_conatus = conatus_final[conatus_final > 0]
                survivor_conatus = np.clip(survivor_conatus, 0, 3.0) # Remove extreme outliers
                history_avg[s, r] = np.mean(survivor_conatus)
                history_gini[s, r] = calculate_gini_batch(survivor_conatus)
            else:
                history_avg[s, r] = 0
                history_gini[s, r] = 0

    return history_avg, history_gini

# ==========================================
# [4] Visualization (Report Generation)
# ==========================================
try:
    import matplotlib
    matplotlib.use('TkAgg')
except:
    pass

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

# --- Panel B: Tipping Point Sensitivity ---
ax2 = fig.add_subplot(gs[0, 1])
print("2. Analyzing Tipping Point...")
intensities = np.concatenate([np.linspace(0.5, 1.2, 4), np.linspace(1.2, 1.8, 8), np.linspace(1.8, 2.5, 4)])
collapse_probs = []
errors = []

for intensity in intensities:
    d_c, _ = run_simulation_batch(n_sims=40, rounds=200, disaster_intensity=intensity)
    collapsed = (d_c[:, -1] < THRESH_COLLAPSE).astype(int)
    prob = np.mean(collapsed)
    err = np.sqrt(prob * (1-prob) / 40)
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

for i in range(15):
    ax3.plot(data_c[i], color='gray', alpha=0.1, linewidth=0.5)

ax3.plot(mean, color='navy', linewidth=2.5, label='Mean Conatus')
ax3.fill_between(range(300), lower, upper, color='blue', alpha=0.3, label='95% CI')

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

print("Analysis Complete. Displaying plot...")
plt.show() 