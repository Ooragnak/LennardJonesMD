include("LJ_gas.jl")

using AMDGPU
using ProgressMeter
using LinearAlgebra

# system
n_particles = 2000

# LENNARD JONES PARAMS: https://doi.org/10.1080/00268976.2016.1246760
#type_A = "Kr"
#mass_A =  83.798             # mass in u = 1e-3 kg/mol
#sigma_A = 0.3627              # sigma in nm     
#epsilon_A = 162.58 * R * 1e-3     # epsilon in kJ/mol

type_A = "Ne"
mass_A =  20.18             # mass in u = 1e-3 kg/mol
sigma_A = 0.2801              # sigma in nm     
epsilon_A = 33.921 * R * 1e-3     # epsilon in kJ/mol

type_B = "Ar"
mass_B =  39.948             # mass in u = 1e-3 kg/mol
sigma_B = 0.33952              # sigma in nm     
epsilon_B = 116.79 * R * 1e-3     # epsilon in kJ/mol

#type_B = "Rn"
#mass_B =  222             # mass in u = 1e-3 kg/mol 
#sigma_B= 0.417           # sigma in nm     
#epsilon_B = 300 * R * 1e-3   # epsilon in kJ/mol 

#type_B = "He"
#mass_B =  4.002602             # mass in u = 1e-3 kg/mol 
#sigma_B= 0.228           # sigma in nm     
#epsilon_B = 10.2 * R * 1e-3  # epsilon in kJ/mol 

# simulation
dt = 0.02          # ps
n_steps = 50000
temperature = 300     # K
end_temperature = 0
box_length = 30      # nm
tau_thermostat = 1  # thermostat coupling constant in 1/ps
rij_min = 1e-2      # nm
rij_min_starting = 0.3 # nm
NVT = true          # switch to decide between NVT and NVE
equilibrate = false # turn on NVT equilibration for all simulations

# Metadata 
save_frequency = 10 # save every n-th step
filename_base = "simulations/NeAr_Large_NVT_300"

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

box_A = ((0, box_length / 2), (0, box_length), (0, box_length))
box_B = ((box_length / 2, box_length), (0, box_length), (0, box_length))


# fill in the parameters

ps.type[1:n_A] = fill(type_A,n_A)
ps.mass[1:n_A] = fill(mass_A,n_A)
ps.sigma[1:n_A]= fill(sigma_A,n_A)
ps.epsilon[1:n_A] = fill(epsilon_A,n_A)


ps.type[n_A+1:end] = fill(type_B,n_B)
ps.mass[n_A+1:end] = fill(mass_B,n_B)
ps.sigma[n_A+1:end] = fill(sigma_B,n_B)
ps.epsilon[n_A+1:end] = fill(epsilon_B,n_B)

function get_position(xrange, yrange, zrange)
    pos = zeros(3)
    pos[1] = rand(Uniform(xrange...))
    pos[2] = rand(Uniform(yrange...))
    pos[3] = rand(Uniform(zrange...))
    return pos
end

# initialize LJ params
initialize_LJ!(ps)

# guess initial position array
pos_initial = zeros(3, n_particles)

for i in 1:n_A
    pos_initial[:, i] = get_position(box_A...)
end

for i in n_A+1:n_particles
    pos_initial[:, i] = get_position(box_B...)
end

# remove distances which are to small
for i in 1:100
    r = [p1[i] - p2[i] for i in 1:3, p1 in eachslice(pos_initial, dims=2), p2 in eachslice(pos_initial, dims=2)]
    # Apply periodic boundary conditions
    @. r = r - round(r / box_length) * box_length

    dist = reshape(norm.(eachslice(r,dims=(2, 3))), n_particles^2)

    # set diagonal elements to inf
    for i in 1:n_particles
        dist[LinearIndices((n_particles, n_particles))[i,i]] = Inf
    end

    cutoff = findfirst(sort(dist) .<= rij_min_starting)

    if !isnothing(cutoff)
        for i in sortperm(dist)[1:cutoff]
            index = CartesianIndices((n_particles, n_particles))[i][1]
            if index > n_A
                pos_initial[:,index] = get_position(box_B...)
            else
                pos_initial[:,index] = get_position(box_A...)
            end
        end
    else
        println("Placed particles respecting rij_min_starting. Current minimum distance: ", minimum(dist))
        break
    end
    
    i == 100 && println("Could not place all particles with rij_min_starting. Current minimum distance: ", minimum(dist))
end

# set initial positions     
ps.position[:, :] = pos_initial


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
if equilibrate
    sim_equil = SimulationParameters(dt/500, n_steps, temperature, box_length, tau_thermostat, rij_min, Inf, NVT, save_frequency)

    for k in 1:1000
        simulate_NVT_step!(ps, sim_equil)
        if instantaneous_temperature(ps) < sim_equil.temperature * 1.1
            println("Rough equilibration finished after $k steps.")
            break
        end
    end

    for k in 1:50
        simulate_NVT_step!(ps, sim_equil)
    end

    println("Equilibration done with current temperature: ", instantaneous_temperature(ps))
end


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

        if !iszero(end_temperature)
            @assert end_temperature <= temperature
            T = end_temperature + (temperature - end_temperature) * (1 - div(i,save_frequency) / div(n_steps,save_frequency))
            global sim = SimulationParameters(dt, n_steps, T, box_length, tau_thermostat, rij_min, Inf, NVT, save_frequency)
        end

    end
    next!(p, showvalues = generate_showvalues(i, current_save_pos))
end

energies = energy_trajectory[1,:] .+ energy_trajectory[2,:]
temperatures = energy_trajectory[3,:]
pressures = energy_trajectory[3,:]


write_xyz_trajectory(filename_base, ps, position_trajectory)
write_energy_trajectory(filename_base, energy_trajectory)
write_output_file(filename_base,ps,sim,stats.time)

using GLMakie
lines(energies)
