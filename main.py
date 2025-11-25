"""
Spinoza: Dynamic Equilibrium & Complexity Simulation
====================================================
A Computational Social Science Project simulating Spinoza's philosophy of 'Conatus' and 'Affects'.

Author: [Seoeun Choi / westsilver]
Description:
  This simulation models a society of agents based on Spinoza's 'Ethics'.
  It visualizes how individual striving (Conatus), reason, and emotional contagion create social structures.
  The model integrates Agent-Based Modeling (ABM) and Network Theory to explore:
    - The formation of communities through shared affects.
    - The role of Reason in resilience against external shocks (Disasters).
    - The dynamics of population growth and inequality (Gini Coefficient).
    - The balance between natural entropy and social cooperation.

Key Features:
  - 3D Visualization using Ursina Engine (Fibonacci Sphere distribution).
  - Real-time interaction (Disasters, Booms, Epidemics, Innovations).
  - Statistical reporting (Matplotlib) for quantitative verification.
  - (Optional) GIF generation for recording the simulation.

Dependencies:
  - ursina
  - matplotlib
  - numpy
  - imageio (Optional, for GIF generation)
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
import os

# --- [OPTIONAL] GIF Generation Import -----------------------------
# Requires 'imageio' library (pip install imageio)
# Uncomment the lines below if you want to generate GIFs.
try:
    import imageio.v2 as imageio
except ImportError:
    print("Note: 'imageio' not found. GIF generation will be disabled.")
# -------------------------------------------------------------------

# ==========================================
# [1] Configuration & Global Variables
# ==========================================

CONFIG = {
    'N_AGENTS': 100,                # Maximum capacity of the society
    'REASON_ALPHA': 2,              # Beta distribution param for Reason (Alpha)
    'REASON_BETA': 2,               # Beta distribution param for Reason (Beta)
    'VITALITY_RANGE': (0.4, 0.8),   # Base resilience range
    'JOY_RANGE': (0.4, 1.0),        # Initial Joy (Energy) range
    'SADNESS_RANGE': (0.0, 0.2),    # Initial Sadness range
    
    # Biological Parameters (Tuned for stability)
    'META_RANGE': (0.001, 0.005),   # Metabolism (Entropy): Low value for better survival
    'RECOV_RANGE': (0.03, 0.1),     # Recovery Rate: High value for resilience
    
    'NATURAL_DEATH_PROB': 0.0005,   # Probability of random natural death per tick
    'INTERACTION_RATE': 0.8,        # Probability of interaction per tick
    
    # GIF Settings
    'ENABLE_GIF_RECORDING': True,  # Set to True to enable recording
    'FRAME_SAVE_INTERVAL': 5,       # Save every Nth frame
    'FRAME_FOLDER': "gif_frames",   # Folder to save frames
    'GIF_FILENAME': "spinoza_simulation.gif"
}

# Global Statistics Containers
stats_conatus = []      # Average Power over time
stats_population = []   # Population count over time
events_log = []         # Log of events (frame, type)
session_history = []    # Stores data for multiple sessions

# Runtime Variables
current_round = 0
frame_counter = 0
focused_agent = None    # ID of the agent currently in focus
last_event_type = None
event_streak = 0
event_cooldown = 0

# --- Helper Functions ---

def calculate_gini(conatus_values):
    """Calculates the Gini Coefficient (Inequality Index)."""
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
    Represents an individual with Conatus, Reason, and Affects.
    """
    def __init__(self, id):
        self.id = id
        self.age = 0
        
        # 1. Nature (Immutable)
        # Reason is distributed using Beta distribution (bell-shaped)
        self.base_reason = np.random.beta(CONFIG['REASON_ALPHA'], CONFIG['REASON_BETA'])
        self.base_vitality = random.uniform(*CONFIG['VITALITY_RANGE'])
        
        # 2. State (Mutable)
        self.joy = random.uniform(*CONFIG['JOY_RANGE'])
        self.sadness = random.uniform(*CONFIG['SADNESS_RANGE'])
        
        # 3. Traits
        self.metabolism = random.uniform(*CONFIG['META_RANGE'])
        self.recovery_rate = random.uniform(*CONFIG['RECOV_RANGE'])
        
        # Generosity: Tendency to cooperate (correlated with Reason)
        self.generosity = np.clip(self.base_reason + random.uniform(-0.2, 0.2), 0.1, 1.0)
        
        # Relationships: {target_id: affinity (-1.0 to 1.0)}
        self.relationships = {} 
    
    @property
    def conatus(self):
        """
        Calculates current power of acting.
        Formula: (Joy - Sadness / Resilience) + BaseVitality
        """
        # Active reason helps to mitigate sadness
        active_reason = self.base_reason * (1.0 if self.joy > self.sadness else 0.5)
        resilience = 1.0 + (active_reason * 1.5) 
        val = (self.joy - (self.sadness / resilience)) + self.base_vitality
        return max(0.0, val)

    def interact_nature(self, other, alive_count):
        """
        Interaction logic: Conflict vs Cooperation based on population pressure.
        """
        pressure = 1.0 - (alive_count / CONFIG['N_AGENTS'])
        # Probability of conflict increases with low generosity and high pressure
        conflict_prob = 0.05 + (1.0 - self.generosity) * 0.1 + pressure * 0.1
        
        if random.random() < conflict_prob:
            # [Conflict] Mutual decrease in power
            dmg = 0.1 * (2.0 - self.base_reason) 
            self.sadness += other.joy * dmg
            other.sadness += self.joy * dmg
            impact = -0.3
        else:
            # [Cooperation] Mutual increase
            benefit = 0.05 * self.generosity
            self.joy += other.joy * benefit
            self.sadness *= 0.9 # Healing effect
            impact = 0.2

        # Update relationship memory
        current_aff = self.relationships.get(other.id, 0.0)
        self.relationships[other.id] = np.clip(current_aff + impact * 2.0, -1.0, 1.0)

    def natural_flux(self):
        """
        Applies natural entropy and self-recovery mechanisms.
        """
        self.age += 1
        
        # Entropy: Constant energy decay
        self.joy -= self.metabolism 
        self.joy += random.uniform(-0.02, 0.02) # Random fluctuation
        
        # Natural Recovery (Survival Instinct)
        # Probability increases when Conatus is low
        if self.conatus < self.base_vitality:
            if random.random() < 0.5: 
                self.joy += self.recovery_rate
                self.sadness *= 0.95

        # Clamp values to prevent overflow/underflow
        self.joy = np.clip(self.joy, 0.0, 3.0)
        self.sadness = np.clip(self.sadness, 0.0, 2.0)

# ==========================================
# [3] Ursina Application Setup
# ==========================================

app = Ursina()
window.title = 'Spinoza Simulation: Final Release'
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

# Entity Lists
agents_logic = []
agent_entities = []
connection_lines = [] 

def get_sphere_pos(i, total):
    """Fibonacci Sphere Algorithm for 3D distribution."""
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
    
    # 3D Sphere Representation
    entity = Entity(model='sphere', color=color.gray, scale=1, position=pos, collider='sphere')
    entity.logic_idx = i
    entity.on_click = Func(toggle_focus, i)
    agent_entities.append(entity)

def create_world():
    """Initializes the simulation world."""
    global current_round, stats_conatus, stats_population, events_log, focused_agent
    global last_event_type, event_streak, event_cooldown, frame_counter
    
    # Clean up existing entities
    for e in agent_entities: destroy(e)
    for l_data in connection_lines: destroy(l_data[0])
    
    agent_entities.clear()
    connection_lines.clear()
    agents_logic.clear()
    
    # Reset Variables
    stats_conatus = []
    stats_population = []
    events_log = []
    current_round = 0
    frame_counter = 0
    focused_agent = None
    last_event_type = None
    event_streak = 0
    event_cooldown = 0

    # Initialize Agents
    for i in range(CONFIG['N_AGENTS']):
        create_agent(i)
    
    session_text.text = f"Session: {len(session_history) + 1}"
    print(f"--- Session {len(session_history) + 1} Started ---")

    # [OPTIONAL] Create folder for GIF frames
    if CONFIG['ENABLE_GIF_RECORDING']:
        if not os.path.exists(CONFIG['FRAME_FOLDER']):
            os.makedirs(CONFIG['FRAME_FOLDER'])

def toggle_focus(idx):
    global focused_agent
    if focused_agent == idx: focused_agent = None
    else: focused_agent = idx
    update_visibility()

def update_visibility():
    global focused_agent
    # Hide/Show nodes based on focus
    for i, entity in enumerate(agent_entities):
        if not entity.enabled: continue
        if focused_agent is None:
            entity.alpha = 1
        else:
            # Show only connected agents
            is_connected = (i in agents_logic[focused_agent].relationships) or (focused_agent in agents_logic[i].relationships)
            if i == focused_agent or is_connected:
                entity.alpha = 1 
            else:
                entity.alpha = 0.1 # Dim others

    # Hide/Show lines
    for line_data in connection_lines:
        line_ent, start, end = line_data
        if focused_agent is None:
            line_ent.enable()
        else:
            if start == focused_agent or end == focused_agent: line_ent.enable()
            else: line_ent.disable()

# ==========================================
# [4] Main Update Loop
# ==========================================

def update():
    global current_round, event_cooldown
    current_round += 1
    if event_cooldown > 0: event_cooldown -= 1

    # --- [OPTIONAL] Frame Capture for GIF ---
    if CONFIG['ENABLE_GIF_RECORDING']:
        global frame_counter
        if current_round % CONFIG['FRAME_SAVE_INTERVAL'] == 0:
            if not os.path.exists(CONFIG['FRAME_FOLDER']):
                os.makedirs(CONFIG['FRAME_FOLDER'])
            
            from panda3d.core import PNMImage 
            img = PNMImage()
            application.base.win.getScreenshot(img)
            path = os.path.join(CONFIG['FRAME_FOLDER'], f"frame_{frame_counter:05d}.png")
            img.write(path)
            frame_counter += 1
    # ----------------------------------------

    alive_indices = [i for i, e in enumerate(agent_entities) if e.enabled]
    alive_count = len(alive_indices)
    
    # 1. Interaction Phase
    if len(alive_indices) > 1 and random.random() < CONFIG['INTERACTION_RATE']: 
        i, j = random.sample(alive_indices, 2)
        a, b = agents_logic[i], agents_logic[j]
        a.interact_nature(b, alive_count)
        b.interact_nature(a, alive_count)
        
        # Draw connection lines
        aff = a.relationships.get(b.id, 0)
        if aff > 0.2: draw_line(i, j, color.rgba(0, 255, 0, 150))  # Green (Love)
        elif aff < -0.2: draw_line(i, j, color.rgba(255, 0, 0, 150)) # Red (Hate)

    # 2. Update State & Death
    for i in alive_indices:
        agent = agents_logic[i]
        agent.natural_flux() 
        
        is_dead = False
        # Death Condition
        if agent.conatus <= 0.0: is_dead = True
        elif random.random() < CONFIG['NATURAL_DEATH_PROB']: is_dead = True
        
        if is_dead:
            agent_entities[i].disable()
            agent_entities[i].scale = 0
            for other in agents_logic:
                if agent.id in other.relationships: del other.relationships[agent.id]

    # 3. Birth Phase (Dynamic Growth)
    pop_ratio = alive_count / CONFIG['N_AGENTS']
    
    # Determine birth probability
    if pop_ratio < 0.3: birth_prob = 0.5  # Emergency birth
    elif pop_ratio < 0.6: birth_prob = 0.2
    else: birth_prob = 0.05 

    # Event Impact on Birth
    spawn_limit = 1
    if event_cooldown > 0:
        if last_event_type in ['BOOM', 'INNOVATION']:
            birth_prob += 0.2
            spawn_limit = 3
        elif last_event_type in ['DISASTER', 'EPIDEMIC']:
            birth_prob *= 0.1
            spawn_limit = 1

    # Execute Birth
    if alive_count < CONFIG['N_AGENTS']: 
        for _ in range(spawn_limit):
            if random.random() < birth_prob: 
                for i, e in enumerate(agent_entities):
                    if not e.enabled: # Find empty slot
                        agents_logic[i] = SpinozaAgent(i) 
                        e.enabled = True
                        e.scale = 0.1 
                        e.animate_scale(1, duration=0.5)
                        break

    update_visuals()
    
    # 4. Statistics
    if current_round % 10 == 0:
        if alive_count > 0:
            vals = [agents_logic[i].conatus for i, e in enumerate(agent_entities) if e.enabled]
            avg_c = np.mean(vals) if vals else 0
            stats_conatus.append(avg_c)
            stats_population.append(alive_count)
            
            live_status.text = f"Round: {current_round}\nPop: {alive_count}/{CONFIG['N_AGENTS']}\nAvg: {avg_c:.2f}"
            if avg_c < 0.6: live_status.color = color.red
            elif avg_c < 1.2: live_status.color = color.yellow
            else: live_status.color = color.green
        else:
            stats_conatus.append(0)
            stats_population.append(0)
            live_status.text = "EXTINCTION"
            live_status.color = color.red
        
    if feedback_text.enabled:
        feedback_text.alpha -= time.dt 
        if feedback_text.alpha <= 0: feedback_text.disable()

def draw_line(i, j, col):
    global focused_agent
    # Limit max lines for performance
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
    for i, entity in enumerate(agent_entities):
        if not entity.enabled: continue
        logic = agents_logic[i]
        c_val = logic.conatus
        
        target_scale = max(0.4, c_val)
        entity.scale = lerp(entity.scale, (target_scale, target_scale, target_scale), time.dt * 5)
        
        if c_val < 0.8: entity.color = lerp(color.red, color.orange, c_val / 0.8)
        elif c_val < 1.5: entity.color = lerp(color.orange, color.green, (c_val - 0.8) / 0.7)
        else: entity.color = lerp(color.green, color.cyan, (c_val - 1.5) / 0.5)

# ==========================================
# [5] Input & Event Handling
# ==========================================

def handle_event(e_type):
    global last_event_type, event_streak, event_cooldown
    if e_type == last_event_type: event_streak += 1
    else: event_streak = 1; last_event_type = e_type
    
    event_cooldown = 50 
    multiplier = min(3.0, 1.0 + (event_streak - 1) * 0.3)
    
    ursina_colors = {'DISASTER':color.red, 'BOOM':color.green, 'EPIDEMIC':color.orange, 'INNOVATION':color.cyan}
    feedback_text.text = f"{e_type} x{event_streak}"
    feedback_text.color = ursina_colors.get(e_type, color.white)
    feedback_text.alpha = 1; feedback_text.enable()
    
    events_log.append((len(stats_conatus), e_type))
    
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
            a.recovery_rate = max(0.001, a.recovery_rate * 0.8)
            a.sadness += 0.1 * multiplier
        elif e_type == 'INNOVATION':
            a.metabolism = max(0.001, a.metabolism * 0.8)
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
    """Displays multiple reports for all sessions."""
    if not session_history:
        print("No data to report.")
        return

    print(f"Generating reports for {len(session_history)} sessions...")
    
    for session in session_history:
        fig, ax1 = plt.subplots(figsize=(12, 7))
        plt.subplots_adjust(bottom=0.15)
        
        ax1.set_title(f"Report: Session {session['id']}", fontsize=14, fontweight='bold')
        
        line1 = ax1.plot(session['conatus'], label='Avg Power', color='blue', linewidth=2)
        ax1.set_ylabel('Average Conatus', color='blue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='blue')
        
        ax2 = ax1.twinx()
        line2 = ax2.plot(session['population'], label='Population', color='green', linestyle='--', linewidth=2)
        ax2.set_ylabel('Alive Agents', color='green', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='green')
        
        colors = {'DISASTER':'red', 'BOOM':'green', 'EPIDEMIC':'orange', 'INNOVATION':'cyan'}
        for idx, event in session['events']:
            c = colors.get(event, 'gray')
            if idx < len(session['conatus']):
                plt.axvline(x=idx, color=c, alpha=0.5, linewidth=2)
                ax1.text(idx, max(session['conatus'])*0.95, event[0], transform=ax1.get_xaxis_transform(), 
                         color=c, fontweight='bold', ha='center', fontsize=9)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        patches = [mpatches.Patch(color=c, label=l, alpha=0.5) for l, c in colors.items()]
        ax1.legend(lines + patches, labels + list(colors.keys()), loc='upper left', framealpha=0.9)
        
        ax1.grid(True, alpha=0.3)
        plt.show(block=False)
    
    plt.show()

# --- [OPTIONAL] GIF Creation Function ---
def make_gif():
    """Combines saved frames into a GIF."""
    if not CONFIG['ENABLE_GIF_RECORDING']: return

    frames = []
    if not os.path.exists(CONFIG['FRAME_FOLDER']): return
    
    files = sorted(os.listdir(CONFIG['FRAME_FOLDER']))
    for f in files:
        if f.endswith(".png"):
            frames.append(imageio.imread(os.path.join(CONFIG['FRAME_FOLDER'], f)))

    if len(frames) > 0:
        imageio.mimsave(CONFIG['GIF_FILENAME'], frames, fps=12)
        print(f"\nGIF successfully created -> {CONFIG['GIF_FILENAME']}\n")
    else:
        print("No frames captured. Check if ENABLE_GIF_RECORDING is True.")
# ----------------------------------------

if __name__ == "__main__":
    create_world()
    try:
        app.run()
    finally:
        show_all_reports()
        
        # [OPTIONAL] Enable to create GIF after closing
        make_gif()