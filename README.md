# LennardJonesMD: A Minimal Molecular Dynamics Simulation in Python

This project implements a velocity-Verlet and Langevin dynamics integrator for simulating Lennard-Jones particles in 3D. It supports both NVE and NVT ensembles and includes periodic boundary conditions, trajectory output, and energy diagnostics.

## 1. Features

- Velocity Verlet (NVE) and BAOAB Langevin (NVT) integration
- Lennard-Jones pairwise interaction
- Periodic and reflective boundary conditions
- Temperature control via Langevin thermostat
- Trajectory output in `.xyz` format
- Energy diagnostics and plotting
- Unit handling in SI-compatible conventions

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- SciPy

## Getting started

python run_simulation.py
---

Here is the revised **Markdown user manual**, incorporating your additional points:

---

# Molecular Dynamics Simulation Manual: `LJ_gas_run_MD.py`

## Overview

This script runs a molecular dynamics (MD) simulation of Lennard-Jones (LJ) particles in either the **NVE** or **NVT** ensemble. It initializes the system, integrates the equations of motion, records energies and trajectories, and visualizes the results.

The simulation:

* Uses **periodic boundary conditions (PBC)** in all directions (always on)
* Implements a **Langevin thermostat** in NVT simulations
* Uses **Lennard-Jones interactions** for all pairwise forces

---

## Units

| Quantity    | Unit          |
| ----------- | ------------- |
| Distance    | nm            |
| Time        | ps            |
| Temperature | K             |
| Energy      | kJ/mol        |
| Pressure    | kJ mol⁻¹ nm⁻³ |

---

## Requirements

* Python 3
* NumPy
* SciPy
* Matplotlib

---

## How to Use

### 1. **Configure the Simulation**

Configure the SimulationParameters class in the script to set simulation parameters:

```python
params = SimulationParameters(
    n_particles=50,
    ensemble='NVT',             # Choose 'NVE' or 'NVT'
    temperature=300,            # Kelvin
    box_length=10.0,            # nm
    time_step=0.002,            # ps
    n_steps=10000,
    output_interval=100,
)
```

---

### 2. **Run the Simulation**

From a terminal, execute:

```bash
python LJ_gas_run_MD.py
```

This will:

* Initialize particle positions (on a cubic grid) and Maxwell-Boltzmann velocities
* Run the MD loop (NVE or NVT with Langevin thermostat)
* Write particle positions to a `.xyz` trajectory file
* Plot energy and temperature over time

---

### 3. **Output Files**

* `trajectory.xyz` – Atom positions (open with VMD or Ovito)
* `energy_plot.png` – Time evolution of potential, kinetic, and total energy
* `temperature_plot.png` – Instantaneous temperature vs. time

---
### 4. **Visualizing the Trajectory in VMD**

To inspect particle motion, you can open the trajectory in **VMD (Visual Molecular Dynamics)**:

1. Launch VMD.
2. Go to `File` → `New Molecule`.
3. Under *Filename*, select `trajectory.xyz`.
4. Click **Load**.

💡 **Tips**:

* VMD assumes XYZ files contain atomic symbols. If the file only contains numerical data, you may need to:

  * Edit the `.xyz` file to assign a dummy element label (e.g., "Ar" or "X") to each atom line.
  * Alternatively, modify `write_xyz_trajectory()` in the code to output proper element tags.

* In the **Graphics → Representations** menu, adjust drawing methods (e.g., "VDW" or "Points") and box size display to better view the particle system in a periodic box.

* Use the **Animation** controls to play, pause, or step through the simulation frames.

---

## Code Components

### Main Simulation Workflow

* `initialize_positions` – Places particles on a grid
* `initialize_velocities` – Draws initial velocities from the Maxwell-Boltzmann distribution
* `simulate_NVE_step` – Velocity Verlet integration (energy-conserving)
* `simulate_NVT_step` – Langevin dynamics integration (temperature-controlled)
* `write_xyz_trajectory` – Saves snapshots of particle coordinates
* `potential_energy`, `kinetic_energy`, `instantaneous_temperature` – Diagnostics

---

## Notes for Students

* You can modify the number of particles, temperature, time step, or box size for experimentation.
* The **NVE ensemble** should conserve total energy (check energy plot).
* The **NVT ensemble** uses Langevin dynamics and should regulate the temperature.
* Periodic boundary conditions mean particles that leave one side of the box re-enter on the opposite side and switched in per default
* Read the file LJ_gas.py to understand how the code works.



Let me know if you'd like a companion script to generate VMD-ready XYZ files with element names or a simple Tcl script for coloring particles, showing the box, etc.
