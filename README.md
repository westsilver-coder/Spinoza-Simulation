# Spinoza Simulation 🌀  
*A dynamic agent-based model inspired by Spinoza’s philosophy of conatus and affects.*

---

## 🎥 Preview  
The following GIF demonstrates a short run of the simulation created inside the project:

![Simulation Preview](preview/spinoza_simulation.gif)

---

## 📘 Overview  
This project implements a **philosophically grounded agent-based simulation** based on:

- **Spinoza’s concept of Conatus** (self-preservation power)    
- **Affects (Joy & Sadness)** and their modulation  
- **Emergent behavioral dynamics** in a social network  
- **Population cycles**, **external shocks**, and **complex systems**  

The simulation is rendered in **3D using Ursina Engine**, and includes features such as relationship lines, collective mood visualization, event triggers, and analytical plots.

---

## ✨ Features  
- **100 fully autonomous agents** with heterogenous traits  
- Joy/Sadness dynamics + metabolism, recovery, and volatility  
- Relationships (love / hate) and spherical spatial distribution  
- **Events**: Disaster, Boom, Epidemic, Innovation  
- Real-time 3D visualization  
- Automatic **report generation** (matplotlib)    
- Automatic **GIF preview generator** (imageio)  
Spinoza-Simulation/  
│  
├─ docs/ -> Additional documentation  
├─ gif_frames/ -> Raw screenshot frames (auto-generated)  
├─ preview/  
│ └─ spinoza_simulation.gif  -> Preview GIF used in README  
│  
├─ analysis.py -> Plot generator for multi-session analytics  
├─ main.py -> Main Ursina simulation  
└─ requirements.txt  

---

## ▶️ How to Run  
### 1) Install dependencies  
pip install -r requirements.txt

### 2) Run simulation
python main.py

### 3) Generate Preview GIF (auto)

A GIF is automatically created at:  
preview/spinoza_simulation.gif

### ⌨️ Controls   
Key  -  Action  
- D: Trigger Disaster  
- B: Trigger Boom  
- E: Trigger Epidemic  
- I: Trigger Innovation  
- Enter: Save session & reset  
- Esc: Show all reports  
- Space: Clear focus  

### 📊 Reports  
All completed sessions are stored and can be viewed through matplotlib plots showing:  
- Average Conatus  
- Population Size  
- Event markers

📜 License  
MIT License.

🤝 Contribution  
Contributions, ideas, and philosophical discussions are welcome.


