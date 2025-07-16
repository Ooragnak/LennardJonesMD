include("LJ_gas.jl")

using AMDGPU
using ProgressMeter

# system
n_particles = 1000
type_A = "Ar"
mass_A =  39.95             # mass in u = 1e-3 kg/mol
sigma_A = 0.34              # sigma in nm     Argon: 0.34
epsilon_A = 120*R*1e-3      # epsilon in kJ/mol Argon: 120

# see http://www.sklogwiki.org/SklogWiki/index.php/Neon
type_B = "Ne"
mass_B =  20.18             # mass in u = 1e-3 kg/mol
sigma_B= 0.2782           # sigma in nm     
epsilon_B = 3.2135 * 10e-3 * elementary_charge * Avogadro   # epsilon in kJ/mol 
epsilon_B = 1.0   # epsilon in kJ/mol 

# simulation
dt = 0.1          # ps
n_steps = 10000
temperature = 40     # K
box_length = 100      # nm
tau_thermostat = 1  # thermostat coupling constant in 1/ps
rij_min = 1e-1      # nm
NVT = true          # switch to decide between NVT and NVE

# Metadata 
save_frequency = 50 # save every n-th step
filename_base = "simulations/JuliaNeArTest"

# Switch between CPU and GPU computation by choosing the array type used for initialization
ARRAYTPE = ROCArray{Float32} # GPU
#ARRAYTPE = Array{Float32} # CPU single precision
#ARRAYTPE = Array{Float64} # CPU double precision

#calculate save points
n_saves = div(n_steps, save_frequency) + 1

sim = SimulationParameters(dt, n_steps, temperature, box_length, tau_thermostat, rij_min, Inf, NVT, save_frequency)

# initialize ParticleSystem 
#
ps = ParticleSystem(n_particles, ARRAYTPE)

# Split systn_particles in two parts
n_A = div(n_particles, 2)
n_B = n_particles - n_A

# fill in the parameters

ps.type[1:n_A] = fill(type_A,n_A)
ps.mass[1:n_A] = fill(mass_A,n_A)
ps.sigma[1:n_A]= fill(sigma_A,n_A)
ps.epsilon[1:n_A] = fill(epsilon_A,n_A)


ps.type[n_A+1:end] = fill(type_B,n_B)
ps.mass[n_A+1:end] = fill(mass_B,n_B)
ps.sigma[n_A+1:end] = fill(sigma_B,n_B)
ps.epsilon[n_A+1:end] = fill(epsilon_B,n_B)

# initialize LJ params
initialize_LJ!(ps)

# set initial positions     
ps.position[1, 1:n_A] = rand(Uniform(0, box_length / 2), n_A)
ps.position[2, 1:n_A] = rand(Uniform(0, box_length), n_A)
ps.position[3, 1:n_A] = rand(Uniform(0, box_length), n_A)

ps.position[1, n_A+1:end] = rand(Uniform(box_length / 2, box_length), n_B)
ps.position[2, n_A+1:end] = rand(Uniform(0, box_length), n_B)
ps.position[3, n_A+1:end] = rand(Uniform(0, box_length), n_B)

# set initial velocities     
initialize_velocities!(ps, sim.temperature)

# calculate force according to initial positions
calculate_force!(ps, sim)

# calculate initial values of variable properties
E_pot_init = potential_energy(ps, sim)
E_kin_init = kinetic_energy(ps)
T_init = instantaneous_temperature(ps)
P_init = ideal_gas_pressure(ps, sim)

position_trajectory = similar(ps.position, (3, n_particles, n_saves))
position_trajectory[:,:,1] = ps.position

# initialize energy trajectory
energy_trajectory = zeros((4,n_saves))
energy_trajectory[1,1] = potential_energy(ps, sim)       # potential energy
energy_trajectory[2,1] = kinetic_energy(ps)               # kinetic energy
energy_trajectory[3,1] = instantaneous_temperature(ps)    # instantaneous temperature
energy_trajectory[4,1] = ideal_gas_pressure(ps, sim)      # ideal gas pressure

# Equilibrate by simulating a few NVT steps
for k in 1:1000
    simulate_NVT_step!(ps, sim)
    if instantaneous_temperature(ps) < sim.temperature * 1.1
        println("Rough equilibration finished after $k steps.")
        break
    end
end

for k in 1:50
    simulate_NVT_step!(ps, sim)
end

println("Equilibration done with current temperature: ", instantaneous_temperature(ps))


p = Progress(sim.n_steps, dt = 1, showspeed=true)
generate_showvalues(iter, current_save_pos) = () -> [(:iter,iter), ("Potential Energy", energy_trajectory[1,current_save_pos+1]), ("Temperature", energy_trajectory[3,current_save_pos+1])]

stats = @timed for i in 1:n_steps
    if NVT
        simulate_NVT_step!(ps, sim)
    else
        simulate_NVE_step!(ps, sim)
    end

    current_save_pos = div(i,save_frequency)
    if mod(i,save_frequency) == 0
        # store updated positions
        position_trajectory[:,:,current_save_pos+1] = ps.position # store updated positions

        # store updated energies, temperature and pressure
        energy_trajectory[1,current_save_pos+1] = potential_energy( ps, sim)     # potential energy
        energy_trajectory[2,current_save_pos+1] = kinetic_energy(ps)             # kinetic energy
        energy_trajectory[3,current_save_pos+1] = instantaneous_temperature(ps)  # instantaneous temperature
        energy_trajectory[4,current_save_pos+1] = ideal_gas_pressure(ps, sim)    # ideal gas pressure

        T = 1 + sim.temperature * (1 - div(i,save_frequency) / div(n_steps,save_frequency))
        global sim = SimulationParameters(dt, n_steps, T, box_length, tau_thermostat, rij_min, Inf, NVT, save_frequency)

    end
    next!(p, showvalues = generate_showvalues(i, current_save_pos))
end

energies = energy_trajectory[1,:] .+ energy_trajectory[2,:]

write_xyz_trajectory(filename_base, ps, position_trajectory)
write_energy_trajectory(filename_base, energy_trajectory)
write_output_file(filename_base,ps,sim,stats.time)
