# LJ-MD: A Minimal Molecular Dynamics Simulation in Python

This project implements a velocity-Verlet and Langevin dynamics integrator for simulating Lennard-Jones particles in 3D. It supports both NVE and NVT ensembles and includes periodic boundary conditions, trajectory output, and energy diagnostics.

## Features

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
