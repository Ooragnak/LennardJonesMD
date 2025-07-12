"""
Run python simulation to compare equality with Julia version.
"""
#----------------------------------------------------------------
#   I M P O R T S
#----------------------------------------------------------------
import numpy as np
from scipy.constants import R, elementary_charge, Avogadro

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# importing
from LJ_gas import(
    ParticleSystem,
    SimulationParameters,
    simulate_NVE_step,
    initialize_velocities,
    initialize_LJ,
    calculate_force,
    )

#----------------------------------------------------------------
#   P A R A M E T E R S
#----------------------------------------------------------------
# system
n_particles = 200
mass_argon =  39.95             # mass in u = 1e-3 kg/mol
sigma_argon = 0.34              # sigma in nm     Argon: 0.34
epsilon_argon = 120*R*1e-3      # epsilon in kJ/mol Argon: 120

# see http://www.sklogwiki.org/SklogWiki/index.php/Neon
mass_neon =  20.18             # mass in u = 1e-3 kg/mol
sigma_neon= 0.2782           # sigma in nm     
epsilon_neon = 3.2135 * 10e-3 * elementary_charge * Avogadro   # epsilon in kJ/mol 
epsilon_neon = 1.0   # epsilon in kJ/mol 


# simulation
dt = 0.2           # ps
n_steps = 1000 
temperature = 300     # K
box_length = 100      # nm
tau_thermostat = 1  # thermostat coupling constant in 1/ps
rij_min = 1e-2      # nm
NVT = False          # switch to decide between NVT and NVE

#----------------------------------------------------------------
#   P R O G R A M
#----------------------------------------------------------------
#
# initialize simulation parameters
#
sim = SimulationParameters(dt = dt, 
                           n_steps = n_steps, 
                           temperature = temperature, 
                           box_length = box_length, 
                           tau_thermostat = tau_thermostat,
                           rij_min=rij_min,
                           is_NVT = NVT
                           )

#
# initialize ParticleSystem 
#
ps = ParticleSystem(n_particles)

# Split systn_particles in two parts
n_argon = n_particles//2
n_neon = n_particles//2 + n_particles%2

# fill in the parameters
for i in range(n_argon): 
    ps.set_parameters(i, type="Ar", mass=mass_argon, sigma=sigma_argon, epsilon=epsilon_argon)

for i in range(n_neon): 
    ps.set_parameters(n_argon + i, type="Ne", mass=mass_neon, sigma=sigma_neon, epsilon=epsilon_neon)

# initialize LJ params
initialize_LJ(ps)

# set initial positions     
ps.position[:n_argon, 0] = np.random.uniform(0, box_length / 2, n_argon)
ps.position[:n_argon, 1] = np.random.uniform(0, box_length, n_argon)
ps.position[:n_argon, 2] = np.random.uniform(0, box_length, n_argon)

ps.position[n_argon:, 0] = np.random.uniform(box_length / 2, box_length, n_neon)
ps.position[n_argon:, 1] = np.random.uniform(0, box_length, n_neon)
ps.position[n_argon:, 2] = np.random.uniform(0, box_length, n_neon)

# set initial velocities     
initialize_velocities(ps, sim.temperature)

# SAVE INITIAL ARRAYS TO COMPARE TO JULIA
np.save("tests/initial_position.npy", ps.position)
np.save("tests/initial_velocity.npy", ps.velocity)

# calculate force according to initial positions
calculate_force(ps, sim)

#--------------------------------------------------
#  The acutal MD simulation
#--------------------------------------------------
for i in range(sim.n_steps):
    simulate_NVE_step(ps, sim)

np.save("tests/final_position.npy", ps.position)
np.save("tests/final_velocity.npy", ps.velocity)