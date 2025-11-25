"""
Spinoza: Dynamic Equilibrium Simulation
=======================================

This simulation models a society of agents based on Spinozan philosophy, specifically the concepts of 'Conatus' (striving for perseverance in being) and 'Affects' (Joy/Sadness).
It integrates agent-based modeling (ABM) with complex systems theory to observe the emergence of social structures and population dynamics under various conditions like natural entropy, social interactions, and external shocks (Disasters, Booms).

Key Features:
1. Agent Heterogeneity: Agents have diverse traits (Reason, Vitality, Metabolism, etc.).
2. Relational Dynamics: Interactions affect emotional states and relationship building (Love/Hate).
3. Population Dynamics: Implements logistic growth for birth rates and age-based/conatus-based death logic.
4. Event System: Simulates external shocks like Disasters, Booms, Epidemics, and Innovations with cumulative effects.
5. Analytical Reporting: Generates time-series plots for average power (Conatus) and population size, visualizing the impact of events.

Dependencies:
- ursina (for 3D visualization and real-time interaction)
- matplotlib (for generating analytical reports)
- numpy (for numerical operations)
"""

import matplotlib
# Ensure matplotlib uses the TkAgg backend for compatible popup windows
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ursina import *
import numpy as np
import random
import copy
import sys

# ==========================================
# [1] Configuration & Global Variables
# ==========================================

CONFIG = {
    'N_AGENTS': 100,                # Total number of agents in the simulation
    'REASON_ALPHA': 2,              # Beta distribution parameter for Reason (Alpha)
    'REASON_BETA': 2,               # Beta distribution parameter for Reason (Beta)
    'VITALITY_RANGE': (0.3, 0.7),   # Range for base vitality (intrinsic resilience)
    'JOY_RANGE': (0.3, 1.0),        # Initial range for Joy
    'SADNESS_RANGE': (0.0, 0.3),    # Initial range for Sadness
    
    # Biological Parameters
    'META_RANGE': (0.005, 0.02),    # Range for Metabolism (Entropy rate)
    'RECOV_RANGE': (0.01, 0.05),    # Range for Recovery Rate (Self-healing)
    
    # Probabilities
    'NATURAL_DEATH_PROB': 0.001,    # Probability of natural death per tick (0.1%)
    'INTERACTION_RATE': 0.7,        # Probability of interaction occurring per tick
    
    # Advanced Settings (Optional)
    'MAX_AGE': 100                  # (Logic placeholder) Maximum age before death prob increases
}

# Global Data Containers for Analysis
stats_conatus = []      # Tracks average conatus over time
stats_population = []   # Tracks population count over time
events_log = []         # Logs event occurrences (frame, type)
session_history = []    # Stores data from multiple sessions

# Simulation State Variables
current_round = 0
focused_agent = None    # ID of the agent currently being inspected
last_event_type = None  # Type of the last triggered event
event_streak = 0        # Counter for consecutive events of the same type
event_cooldown = 0      # Timer for event effect duration

# --- Helper Functions ---

def calculate_gini(conatus_values):
    """Calculates the Gini Coefficient to measure inequality in Conatus distribution."""
    if len(conatus_values) == 0: return 0.0
    conatus_values = np.sort(conatus_values)
    n = len(conatus_values)
    if n == 0 or np.sum(conatus_values) == 0: return 0.0
    index = np.arange(1, n + 1)
    return ((np.sum((2 * index - n - 1) * conatus_values)) / (n * np.sum(conatus_values)))

# ==========================================
# [2] Agent Logic Class
# ==========================================

class SpinozaAgent:
    """
    Represents an individual agent in the Spinozan world.
    Each agent has intrinsic properties (Reason, Vitality) and mutable states (Joy, Sadness).
    """
    def __init__(self, id):
        self.id = id
        self.age = 0  # Agent age counter
        
        # 1. Intrinsic Nature (Fixed)
        # Reason determines the capability to manage affects.
        self.base_reason = np.random.beta(CONFIG['REASON_ALPHA'], CONFIG['REASON_BETA'])
        # Base Vitality is the baseline conatus level the agent strives to maintain.
        self.base_vitality = random.uniform(*CONFIG['VITALITY_RANGE'])
        
        # 2. Current State (Mutable)
        self.joy = random.uniform(*CONFIG['JOY_RANGE'])
        self.sadness = random.uniform(*CONFIG['SADNESS_RANGE'])
        
        # 3. Biological Traits
        self.metabolism = random.uniform(*CONFIG['META_RANGE'])   # Energy decay rate
        self.recovery_rate = random.uniform(*CONFIG['RECOV_RANGE']) # Healing rate
        
        # 4. Social Trait
        # Generosity: Propensity to help others or cooperate.
        self.generosity = np.clip(self.base_reason + random.uniform(-0.2, 0.2), 0.1, 1.0)
        
        # Relationship Memory: {target_id: affinity_score (-1.0 to 1.0)}
        self.relationships = {} 
    
    @property
    def conatus(self):
        """
        Calculates the current 'Conatus' (Power of Acting).
        Formula: Conatus = (Joy - Sadness / Resilience) + Base Vitality
        * Higher Reason -> Higher Resilience against Sadness.
        """
        active_reason = self.base_reason * (1.0 if self.joy > self.sadness else 0.5)
        resilience = 1.0 + (active_reason * 1.5) 
        # Calculate value, ensuring Joy contributes and Sadness subtracts (dampened by resilience)
        val = (self.joy - (self.sadness / resilience)) + self.base_vitality
        return max(0.0, val)

    def interact_nature(self, other, alive_count):
        """
        Simulates interaction with another agent.
        Can result in Conflict (Sadness increase) or Cooperation (Joy increase).
        """
        # Calculate social pressure based on population density
        pressure = 1.0 - (alive_count / CONFIG['N_AGENTS'])
        
        # Probability of conflict depends on generosity and pressure
        # Reduced base conflict probability for stability
        base_conflict = 0.1 + (1.0 - self.generosity) * 0.15 + pressure * 0.1
        conflict_prob = base_conflict * 0.5 
        
        if random.random() < conflict_prob:
            # [Conflict] Both parties suffer sadness
            dmg = 0.1 * (2.0 - self.base_reason) # Lower reason -> Higher damage
            self.sadness += other.joy * dmg
            other.sadness += self.joy * dmg
            impact = -0.3
        else:
            # [Cooperation] Joy is shared, Sadness is alleviated
            benefit = 0.04 * self.generosity
            self.joy += other.joy * benefit
            self.sadness *= 0.95 # Healing effect
            impact = 0.15

        # Update Relationship Affinity
        current_aff = self.relationships.get(other.id, 0.0)
        self.relationships[other.id] = np.clip(current_aff + impact * 2.0, -1.0, 1.0)

    def natural_flux(self):
        """
        Updates agent state based on natural entropy and self-recovery.
        """
        self.age += 1
        
        # 1. Entropy (Diminishing Returns)
        # Higher metabolism leads to faster Joy decay.
        # '0.8' factor softens the decay for better survival rates.
        self.joy -= (self.metabolism * 0.8)
        self.joy += random.uniform(-0.02, 0.02) # Random fluctuation
        
        # 2. Natural Recovery (Homeostasis)
        # Small chance to recover spontaneously
        if random.random() < 0.3:
            self.joy += 0.005 
            
        # Crisis Recovery: If Conatus falls below base vitality, attempt to heal
        if self.conatus < self.base_vitality:
            if random.random() < self.base_reason: 
                self.joy += self.recovery_rate
                self.sadness *= 0.96

        # Clamp values to reasonable limits
        self.joy = np.clip(self.joy, 0.0, 3.0)
        self.sadness = np.clip(self.sadness, 0.0, 2.0)

# ==========================================
# [3] Ursina Application Setup
# ==========================================

app = Ursina()
window.title = 'Spinoza: Dynamic Equilibrium'
window.borderless = False
window.color = color.black

# Camera Setup
EditorCamera()
camera.position = (0, 0, -120) 
camera.look_at((0, 0, 0))

# UI Elements
info_text = Text(
    text="[D] Disaster [B] Boom [E] Epidemic [I] Innovation\n[Enter] Save & Reset  [Esc] Show All Reports", 
    position=(-0.85, 0.45), scale=1.1, color=color.white
)
session_text = Text(text="Session: 1", position=(-0.85, 0.40), scale=1.2, color=color.yellow)
live_status = Text(text="Init...", position=(-0.85, 0.35), scale=1.2, color=color.green)
feedback_text = Text(text="", position=(0, -0.4), origin=(0,0), scale=2, color=color.red, enabled=False)

# Entity Management Lists
agents_logic = []       # Stores logic objects (SpinozaAgent)
agent_entities = []     # Stores visual objects (Ursina Entity)
connection_lines = []   # Stores visual connection lines

def get_sphere_pos(i, total):
    """Calculates 3D coordinates for spherical distribution (Fibonacci Sphere)."""
    phi = np.pi * (3. - np.sqrt(5.))
    radius = 35
    y = 1 - (i / float(total - 1)) * 2 
    r = np.sqrt(1 - y * y) * radius
    theta = phi * i
    x = np.cos(theta) * r
    z = np.sin(theta) * r
    y_pos = y * radius
    return (x, y_pos, z)

def create_agent(i):
    """Instantiates a single agent and its visual representation."""
    logic = SpinozaAgent(i)
    agents_logic.append(logic)
    pos = get_sphere_pos(i, CONFIG['N_AGENTS'])
    entity = Entity(model='sphere', color=color.gray, scale=1, position=pos, collider='sphere')
    entity.logic_idx = i
    entity.on_click = Func(toggle_focus, i)
    agent_entities.append(entity)

def create_world():
    """Initializes or Resets the simulation world."""
    global current_round, stats_conatus, stats_population, events_log, focused_agent
    global last_event_type, event_streak, event_cooldown
    
    # Clear existing entities
    for e in agent_entities: destroy(e)
    for l_data in connection_lines: destroy(l_data[0])
    
    agent_entities.clear()
    connection_lines.clear()
    agents_logic.clear()
    
    # Reset Stats
    stats_conatus = []
    stats_population = []
    events_log = []
    current_round = 0
    focused_agent = None
    last_event_type = None
    event_streak = 0
    event_cooldown = 0

    # Create Agents
    for i in range(CONFIG['N_AGENTS']):
        create_agent(i)
    
    session_text.text = f"Session: {len(session_history) + 1}"
    print(f"--- Session {len(session_history) + 1} Started ---")

def toggle_focus(idx):
    """Toggles focus mode on a specific agent."""
    global focused_agent
    if focused_agent == idx: focused_agent = None
    else: focused_agent = idx
    update_visibility()

def update_visibility():
    """Updates visibility of agents and lines based on focus mode."""
    global focused_agent
    for i, entity in enumerate(agent_entities):
        if not entity.enabled: continue
        if focused_agent is None:
            entity.alpha = 1
        else:
            if focused_agent >= len(agents_logic): 
                focused_agent = None
                entity.alpha = 1
                continue
            is_connected = (i in agents_logic[focused_agent].relationships) or (focused_agent in agents_logic[i].relationships)
            entity.alpha = 1 if i == focused_agent or is_connected else 0.1

    for line_data in connection_lines:
        line_ent, start, end = line_data
        if focused_agent is None:
            line_ent.enable()
        else:
            if start == focused_agent or end == focused_agent: line_ent.enable()
            else: line_ent.disable()

# ==========================================
# [4] Main Update Loop (Simulation Logic)
# ==========================================

def update():
    global current_round, event_cooldown
    current_round += 1
    if event_cooldown > 0: event_cooldown -= 1
    
    # Filter alive agents
    alive_indices = [i for i, e in enumerate(agent_entities) if e.enabled]
    alive_count = len(alive_indices)
    
    # 1. Interaction Phase
    # Randomly pair agents for interaction
    if len(alive_indices) > 1 and random.random() < CONFIG['INTERACTION_RATE']: 
        i, j = random.sample(alive_indices, 2)
        a, b = agents_logic[i], agents_logic[j]
        a.interact_nature(b, alive_count)
        b.interact_nature(a, alive_count)
        
        # Visualize Relationship
        aff = a.relationships.get(b.id, 0)
        if aff > 0.2: draw_line(i, j, color.rgba(0, 255, 0, 150))  # Green for Love/Support
        elif aff < -0.2: draw_line(i, j, color.rgba(255, 0, 0, 150)) # Red for Hate/Conflict

    # 2. Natural Flux & Death Phase
    for i in alive_indices:
        agent = agents_logic[i]
        agent.natural_flux() 
        
        is_dead = False
        # [A] Death by Exhaustion (Conatus <= 0)
        if agent.conatus <= 0.0: is_dead = True
        # [B] Age-based Natural Death (Gompertz-like probability)
        # Probability increases significantly after age 50
        age_factor = max(0, (agent.age - 50)) / 50.0 
        natural_death_prob = 0.0001 + (age_factor ** 4) * 0.1 
        
        if random.random() < natural_death_prob: is_dead = True
        
        # Execute Death
        if is_dead:
            agent_entities[i].disable() # Hide entity
            agent_entities[i].scale = 0
            # Remove relationships involving the dead agent
            for other in agents_logic:
                if agent.id in other.relationships: del other.relationships[agent.id]

    # 3. Birth Phase (Logistic Growth)
    # dN/dt = r * N * (1 - N/K)
    capacity = CONFIG['N_AGENTS']
    
    # Adjust capacity based on events
    if event_cooldown > 0:
        if last_event_type in ['BOOM', 'INNOVATION']: capacity = int(CONFIG['N_AGENTS'] * 1.2)
        elif last_event_type in ['DISASTER', 'EPIDEMIC']: capacity = int(CONFIG['N_AGENTS'] * 0.6)
    
    # Calculate current population ratio
    current_ratio = alive_count / capacity if capacity > 0 else 1.0
    
    # Birth logic: Higher probability when population is low (Logistic Growth)
    if current_ratio < 0.95:
        if current_ratio < 0.2:
            birth_chance = 0.2 # Boost birth if dangerously low
        else:
            birth_chance = 0.1 * (1.0 - current_ratio) # Density-dependent
            
        if random.random() < birth_chance: 
            # Find a dead/disabled slot to respawn
            for i, e in enumerate(agent_entities):
                if not e.enabled:
                    agents_logic[i] = SpinozaAgent(i) # New agent (Age 0)
                    e.enabled = True
                    e.scale = 0.1 
                    e.animate_scale(1, duration=0.5) # Spawn animation
                    break

    # Update Visuals (Color/Scale)
    update_visuals()
    
    # 4. Statistics Recording
    if current_round % 10 == 0:
        if alive_count > 0:
            vals = [agents_logic[i].conatus for i, e in enumerate(agent_entities) if e.enabled]
            avg_c = np.mean(vals) if vals else 0
            stats_conatus.append(avg_c)
            stats_population.append(alive_count)
            
            # Update UI Text
            live_status.text = f"Round: {current_round}\nPop: {alive_count}/{CONFIG['N_AGENTS']}\nAvg: {avg_c:.2f}"
            
            if avg_c < 0.6: live_status.color = color.red
            elif avg_c < 1.2: live_status.color = color.yellow
            else: live_status.color = color.green
        else:
            stats_conatus.append(0)
            stats_population.append(0)
            live_status.text = "EXTINCTION"
            live_status.color = color.red
        
    # Fade out feedback text
    if feedback_text.enabled:
        feedback_text.alpha -= time.dt 
        if feedback_text.alpha <= 0: feedback_text.disable()

def draw_line(i, j, col):
    """Draws a connection line between two agents."""
    global focused_agent
    # Limit number of lines for performance
    if len(connection_lines) > 400: 
        old_line = connection_lines.pop(0)
        destroy(old_line[0])
        
    if not agent_entities[i].enabled or not agent_entities[j].enabled: return
    
    p1 = agent_entities[i].position
    p2 = agent_entities[j].position
    line = Entity(model=Mesh(vertices=[p1, p2], mode='line', thickness=2), color=col)
    connection_lines.append((line, i, j))
    
    if focused_agent is not None:
        if i != focused_agent and j != focused_agent: line.disable()

def update_visuals():
    """Updates visual properties (Scale, Color) of agents based on their state."""
    for i, entity in enumerate(agent_entities):
        if not entity.enabled: continue
        logic = agents_logic[i]
        c_val = logic.conatus
        
        target_scale = max(0.4, c_val)
        entity.scale = lerp(entity.scale, (target_scale, target_scale, target_scale), time.dt * 5)
        
        # Color gradient based on Conatus value
        if c_val < 0.8: entity.color = lerp(color.red, color.orange, c_val / 0.8)
        elif c_val < 1.5: entity.color = lerp(color.orange, color.green, (c_val - 0.8) / 0.7)
        else: entity.color = lerp(color.green, color.cyan, (c_val - 1.5) / 0.5)

# ==========================================
# [5] Input Handling & Event Processing
# ==========================================

def handle_event(e_type):
    global last_event_type, event_streak, event_cooldown
    if e_type == last_event_type: event_streak += 1
    else: event_streak = 1; last_event_type = e_type
    
    event_cooldown = 50 
    multiplier = min(3.0, 1.0 + (event_streak - 1) * 0.3) # Cumulative effect multiplier
    
    ursina_colors = {'DISASTER':color.red, 'BOOM':color.green, 'EPIDEMIC':color.orange, 'INNOVATION':color.cyan}
    feedback_text.text = f"{e_type} x{event_streak}"
    feedback_text.color = ursina_colors.get(e_type, color.white)
    feedback_text.alpha = 1; feedback_text.enable()
    
    events_log.append((len(stats_conatus), e_type))
    
    # Apply event effects
    for a in agents_logic:
        if a.conatus <= 0: continue
        if e_type == 'DISASTER':
            impact = random.uniform(0.3, 0.6) * multiplier
            a.joy *= (1.0 - np.clip(impact, 0, 0.9))
            a.sadness += 0.3 * multiplier
        elif e_type == 'BOOM':
            a.joy += 0.3 * multiplier
            a.sadness *= 0.5
        elif e_type == 'EPIDEMIC':
            a.recovery_rate = max(CONFIG['RECOV_RANGE'][0], a.recovery_rate * 0.8)
            a.sadness += 0.1 * multiplier
        elif e_type == 'INNOVATION':
            a.metabolism = max(CONFIG['META_RANGE'][0], a.metabolism * 0.8)
            a.joy += 0.1 * multiplier

def input(key):
    global focused_agent
    if key == 'space': focused_agent = None; update_visibility()
    if key == 'd': handle_event('DISASTER')
    if key == 'b': handle_event('BOOM')
    if key == 'e': handle_event('EPIDEMIC')
    if key == 'i': handle_event('INNOVATION')
    
    if key == 'enter': 
        save_current_session()
        create_world()
    
    if key == 'escape': 
        save_current_session()
        application.quit()

def save_current_session():
    """Saves current simulation data to history before reset."""
    if len(stats_conatus) > 0:
        session_data = {
            'id': len(session_history) + 1,
            'conatus': list(stats_conatus),
            'population': list(stats_population),
            'events': copy.deepcopy(events_log)
        }
        session_history.append(session_data)
        print(f"Session {session_data['id']} saved.")

def show_all_reports():
    """Generates and displays reports for all recorded sessions."""
    if not session_history:
        print("No data to report.")
        return

    n_sessions = len(session_history)
    print(f"Generating reports for {n_sessions} sessions...")
    
    for session in session_history:
        fig, ax1 = plt.subplots(figsize=(12, 7))
        plt.subplots_adjust(bottom=0.15)
        
        ax1.set_title(f"Report: Session {session['id']}", fontsize=14, fontweight='bold')
        
        # Avg Power Plot
        line1 = ax1.plot(session['conatus'], label='Avg Power', color='blue', linewidth=2)
        ax1.set_ylabel('Average Conatus', color='blue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Population Plot
        ax2 = ax1.twinx()
        line2 = ax2.plot(session['population'], label='Population', color='green', linestyle='--', linewidth=2)
        ax2.set_ylabel('Alive Agents', color='green', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='green')
        
        # Event Markers
        colors = {'DISASTER':'red', 'BOOM':'green', 'EPIDEMIC':'orange', 'INNOVATION':'cyan'}
        for idx, event in session['events']:
            c = colors.get(event, 'gray')
            if idx < len(session['conatus']):
                plt.axvline(x=idx, color=c, alpha=0.5, linewidth=2)
                ax1.text(idx, max(session['conatus'])*0.95, event[0], transform=ax1.get_xaxis_transform(), 
                         color=c, fontweight='bold', ha='center', fontsize=9)
        
        # Legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        patches = [mpatches.Patch(color=c, label=l, alpha=0.5) for l, c in colors.items()]
        ax1.legend(lines + patches, labels + list(colors.keys()), loc='upper left', framealpha=0.9)
        
        ax1.grid(True, alpha=0.3)
        plt.show(block=False)
    
    plt.show()

if __name__ == "__main__":
    create_world()
    try:
        app.run()
    finally:
        show_all_reports()