#----------------------------------------------------------------
#   I M P O R T S
#----------------------------------------------------------------

using KernelAbstractions
using StaticArrays
using Printf
using Distributions
using LinearAlgebra

import Dates
import StatsBase

############### CONSTANTS ###############
using PhysicalConstants: CODATA2022 as constants

R = constants.R.val
elementary_charge = constants.ElementaryCharge.val
Avogadro = constants.AvogadroConstant.val
#########################################

#----------------------------------------------------------------
#   S T R U C T S 
#----------------------------------------------------------------

mutable struct ParticleSystem{T}
    const n::Int
    mass::AbstractVector
    sigma::AbstractVector
    epsilon::AbstractVector
    type::AbstractVector{String}

    position::AbstractMatrix{T}
    velocity::AbstractMatrix{T}
    force::AbstractMatrix{T}

    combined_sigma::AbstractMatrix
    combined_epsilon::AbstractMatrix
end

ParticleSystem(n, arrType) = ParticleSystem(
    n,
    arrType(zeros(n)),
    zeros(n),
    zeros(n),
    fill("ATOM",n),
    arrType(zeros(3, n)),
    arrType(zeros(3, n)),
    arrType(zeros(3, n)),
    arrType(zeros(n, n)),
    arrType(zeros(n, n)),
)

struct SimulationParameters
    dt
    n_steps
    temperature
    box_length
    tau_thermostat
    rij_min
    rij_cutoff
    is_NVT::Bool
end

#----------------------------------------------------------------
#   K E R N E L S
#----------------------------------------------------------------

@kernel function calculatePotentialEnergiesKernel!(energies, @Const(px), @Const(py), @Const(pz), @Const(combined_sigma), @Const(combined_epsilon), @Const(box_length), @Const(n_particles), @Const(rij_min), @Const(rij_cutoff))
    i = @index(Global, Linear)

    E = zero(eltype(energies))
    r = zero(MVector{3,eltype(energies)})
    
    for j in 1:n_particles if j != i

        # Calculate distances seperately (to work on GPUs)
        r[1] = px[i] - px[j]
        r[2] = py[i] - py[j]
        r[3] = pz[i] - pz[j]
        
        # Apply periodic boundary conditions
        @. r = r - round(r / box_length) * box_length

        # Calculate scalar distance
        d = norm(r)

        if d < rij_min
            d = rij_min

            sr6 = (combined_sigma[i,j] / d)^6
            E += 4 * combined_epsilon[i,j] * (sr6^2 - sr6)
        elseif d > rij_cutoff
            E += zero(E)
        else 
            sr6 = (combined_sigma[i,j] / d)^6
            E += 4 * combined_epsilon[i,j] * (sr6^2 - sr6)
        end

    end end

    energies[i] = E
end

@kernel function calculateForcesKernel!(forces, @Const(px), @Const(py), @Const(pz), @Const(combined_sigma), @Const(combined_epsilon), @Const(box_length), @Const(n_particles), @Const(rij_min), @Const(rij_cutoff))
    i = @index(Global, Linear)

    f = zero(MVector{3,eltype(forces)})
    r = zero(MVector{3,eltype(forces)})

    for j in 1:n_particles if j != i

        # Calculate distances seperately (to work on GPUs)
        r[1] = px[i] - px[j]
        r[2] = py[i] - py[j]
        r[3] = pz[i] - pz[j]
        
        # Apply periodic boundary conditions
        @. r = r - round(r / box_length) * box_length

        # Calculate scalar distance
        d = norm(r)

        normalize!(r)

        if d < rij_min
            d = rij_min
            
            sr6 = (combined_sigma[i,j] / d)^6
            dV_dr = 24 * combined_epsilon[i,j] / d * (-2 * sr6^2 + sr6)

            @. f -= r * dV_dr
        elseif d > rij_cutoff
            @. f += zero(f)
        else 
            sr6 = (combined_sigma[i,j] / d)^6
            dV_dr = 24 * combined_epsilon[i,j] / d * (-2 * sr6^2 + sr6)

            @. f -= r * dV_dr
        end
    end end

    forces[:,i] .= f
end

@kernel function calculateOstepKernel!(velocities, @Const(masses), @Const(scalar), @Const(dissipation))
    # j: 1,2,3 ≣ x,y,z
    j, i = @index(Global, NTuple)

    # Scalar mass factor (1e3) taken from python implementation, not sure about the physical reasoning 
    velocities[j, i] = dissipation * velocities[j,i] + sqrt(scalar / (masses[i] * 1e3)) * randn()
end

#----------------------------------------------------------------
#   F U N C T I O N S
#----------------------------------------------------------------

"""
Initializes velocities of a ParticleSystem according to the Maxwell-Boltzmann
distribution at a given temperature T (in Kelvin), using vectorized NumPy operations.
Each velocity component is sampled from a Gaussian with:
    variance = sigma^2 = R*T / M

Velocities are returned in units of nm/ps.
"""
function initialize_velocities!(ps::ParticleSystem{T}, temperature) where {T}
    # molar masses in kg/mol (convert from u)
    # Convert to CPU array to avoid scalar index on gpu arrays
    M = Array(ps.mass .* 1e-3)
    
    # Sample velocities:
    velocities_m_s = [rand(Normal(0.0, sqrt(R * temperature / m))) for i in 1:3, m in M]

    # Convert to nm/ps
    velocities_nm_ps = velocities_m_s * 1e-3

    # Remove center-of-mass velocity
    v_cm = StatsBase.mean(velocities_nm_ps, StatsBase.weights(M), dims=2)
    velocities_nm_ps = velocities_nm_ps .- v_cm

    # Set velocities
    ps.velocity = convert(typeof(ps.velocity), velocities_nm_ps)
end

"""Initialize combined Lennard-Jones parameters using Lorentz-Berthelot combining rules."""
function initialize_LJ!(ps::ParticleSystem)
    ps.combined_sigma = convert(typeof(ps.combined_sigma) ,[0.5 * (sigma_A + sigma_B) for sigma_A in ps.sigma, sigma_B in ps.sigma])
    ps.combined_epsilon = convert(typeof(ps.combined_epsilon) ,[sqrt(epsilon_A * epsilon_B) for epsilon_A in ps.epsilon, epsilon_B in ps.epsilon])
end

"""
Computes the total Lennard-Jones potential energy of the system.

Units:
    Energy is in the same units as epsilon (kJ/mol). 
    Positions must be in the same units as sigma (nm).
"""
function potential_energy(ps::ParticleSystem, sim::SimulationParameters; per_particle = false)
    backend = get_backend(ps.position)
    energies = zero(ps.position[1,:])

    px = ps.position[1,:]
    py = ps.position[2,:]
    pz = ps.position[3,:]

    kernel! = calculatePotentialEnergiesKernel!(backend)

    kernel!(energies, px, py, pz, ps.combined_sigma, ps.combined_epsilon, sim.box_length, ps.n, sim.rij_min, sim.rij_cutoff, ndrange = ps.n)

    if per_particle
        return energies .* 0.5
    else
        return sum(energies) * 0.5
    end
end

function kinetic_energy(ps::ParticleSystem)
    return 0.5 *  sum(ps.velocity.^2 .* transpose(ps.mass))
end 

function instantaneous_temperature(p::ParticleSystem)
    # kinetic energy is returned in kJ/mol, convert to J/mol
    E_kin = kinetic_energy(ps)*1e3
    # degrees of freedom: 3 per particle
    dof = ps.n*3
    return (2* E_kin) / (dof *R)
end

function density(ps::ParticleSystem, sim::SimulationParameters)
    L_in_nm = sim.box_length
    # nm^3 = 10^{-27} m^3 = 10^{-27} m^3* 1000 L/m^3 = 10^{-24} L

    V_in_cm3 = L_in_nm^3 * 1e-21 
    # Mass is stored in u = g/mol
    # Total mass in g (sum of all molar masses divided by Avogadro)
    m_in_g = sum(ps.mass) / Avogadro 
    return m_in_g/V_in_cm3
end

function ideal_gas_pressure(ps::ParticleSystem, sim::SimulationParameters)
    L_in_nm = sim.box_length
    V_in_m3 = L_in_nm^3 * 1e-27  # Convert volume to m³
    n_mol = ps.n / Avogadro  # Amount of substance in mol
    T = instantaneous_temperature(ps)  # in Kelvin
    return n_mol * R * T / V_in_m3  # Pressure in Pascals (Pa)
end

function calculate_force!(ps::ParticleSystem, sim::SimulationParameters)
    backend = get_backend(ps.force)

    kernel! = calculateForcesKernel!(backend)

    px = ps.position[1,:]
    py = ps.position[2,:]
    pz = ps.position[3,:]

    kernel!(ps.force, px, py, pz, ps.combined_sigma, ps.combined_epsilon, sim.box_length, ps.n, sim.rij_min, sim.rij_cutoff, ndrange = ps.n)

end

function A_step!(ps::ParticleSystem, sim::SimulationParameters; half_step=false)
    # set time step, depending on whether a half- or full step is performed
    if half_step == true
        dt = 0.5 * sim.dt
    else
        dt = sim.dt
    end 
    @. ps.position = ps.position + ps.velocity * dt
end

function B_step!(ps::ParticleSystem, sim::SimulationParameters; half_step=false)
    # set time step, depending on whether a half- or full step is performed
    if half_step == true
        dt = 0.5 * sim.dt
    else
        dt = sim.dt
    end 
    
    ps.velocity .= ps.velocity .+ transpose(1 ./ ps.mass) .* dt .* ps.force
end

function O_step!(ps::ParticleSystem, sim::SimulationParameters; half_step=false)
    # set time step, depending on whether a half- or full step is performed
    if half_step == true
        dt = 0.5 * sim.dt
    else
        dt = sim.dt
    end

    xi = 1 / sim.tau_thermostat

    # dissipation term
    d = exp(- xi * dt)

    # fluctuation term
    scalar = sim.temperature * R * (1.0 - exp(-2 * xi * dt))

    backend = get_backend(ps.velocity)

    kernel! = calculateOstepKernel!(backend)

    kernel!(ps.velocity, ps.mass, scalar, d,  ndrange = (3,ps.n))
end

function apply_periodic_boundary!(ps::ParticleSystem, sim::SimulationParameters)
    ps.position = mod.(ps.position, sim.box_length)
end

function simulate_NVE_step!(ps::ParticleSystem, sim::SimulationParameters)
    B_step!(ps, sim, half_step=true)   # update velocity by a half-step
    A_step!(ps, sim, half_step=false)  # update position by a full time step
    apply_periodic_boundary!(ps, sim)

    calculate_force!(ps, sim)          # udpate force  
    B_step!(ps, sim, half_step=true)   # update velocity by a second half-step
end

function simulate_NVT_step!(ps::ParticleSystem, sim::SimulationParameters)
    if sim.tau_thermostat == 0 || isnothing(sim.tau_thermostat)
        Error(ValueError(), "Thermostat coupling time (tau_thermostat) is not set. Cannot run NVT simulation.")
    end
    B_step!(ps, sim, half_step=true)   # update velocity by a half-step
    A_step!(ps, sim, half_step=true)  # update position by a half-step
    apply_periodic_boundary!(ps, sim)

    # thermostat
    O_step!(ps, sim, half_step=false)  # Full-step velocity update using the Langevin thermostat (friction + noise)
    A_step!(ps, sim, half_step=true)  # update position by a half-step
    apply_periodic_boundary!(ps, sim)

    calculate_force!(ps, sim)          # udpate force  
    B_step!(ps, sim, half_step=true)   # update velocity by a second half-step

end

#----------------------------------------------------------------
#   O U T P U T
#----------------------------------------------------------------

function write_xyz_trajectory(filename_base, ps::ParticleSystem, trajectory)
    trajectory = Array(10.0 * trajectory)  # convert nm to Å
    _, n_atoms, n_frames = size(trajectory)
    open(filename_base * "_pos.xyz", "w") do file
        for j in 1:n_frames
            write(file, "$n_atoms\n")
            write(file, "Generated by write_xyz_trajectory\n")
            for i in 1:n_atoms
                write(file, @sprintf "%s %.8f %.8f %.8f\n" ps.type[i] trajectory[1,i,j] trajectory[2,i,j] trajectory[3,i,j])
            end
        end
    end
end

function write_energy_trajectory(filename_base, energy_trajectory)
    traj = Array(energy_trajectory)
    _, n_frames = size(traj)

    open(filename_base * "_ene.dat", "w") do file
        write(file, "#E_pot  E_kin  T  P")
        for i in 1:n_frames
            write(file, @sprintf "%s %.8f %.8f %.8f\n" traj[1,i] traj[2,i] traj[3,i] traj[4,i])
        end
    end
end

function write_output_file(filename_base, ps::ParticleSystem, sim::SimulationParameters, elapsed_time = nothing)
    """
    Writes the simulation properties to an .out file.

    Parameters:
        ps (ParticleSystem): Contains velocities, masses, and random number storage.
        sim (SimulationParameters): Contains xi, temperature, dt, and constants.
    optional:
        elapsed_time: Float containing the time elapsed during the calculation.

    Returns:
        None. Writes file to disk.    
    """
    output_lines = []

    push!(output_lines, "")
    push!(output_lines, "----------------------------------------------------------")
    push!(output_lines, "Simulation parameters ")
    push!(output_lines, "----------------------------------------------------------")
    push!(output_lines, @sprintf("%-30s%10.0f ", "Number of particles:", ps.n))
    push!(output_lines, @sprintf("%-30s%10.3e nm", "Box length:", sim.box_length))
    push!(output_lines, @sprintf("%-30s%10.3e nm^3", "Box volume:", sim.box_length^3))
    push!(output_lines, @sprintf("%-30s%10.3e g/cm^3", "Density:", density(ps, sim)))
    push!(output_lines, "")
    push!(output_lines, @sprintf("%-30s%10.3f ps", "Time step:", sim.dt))
    push!(output_lines, @sprintf("%-30s%10.0f", "Number of time steps:", sim.n_steps))
    push!(output_lines, @sprintf("%-30s%10.3e ps", "Simulation time:", sim.n_steps * sim.dt))
    push!(output_lines, "")

    if sim.is_NVT
        push!(output_lines, @sprintf("%-30s%10s", "Ensemble:", "NVT"))
        push!(output_lines, @sprintf("%-30s%10.0f K", "Thermostat temperature:", sim.temperature))
        push!(output_lines, @sprintf("%-30s%10.3e ps", "Thermostat coupling:", sim.tau_thermostat))
    else
        push!(output_lines, @sprintf("%-30s%10s", "Ensemble:", "NVE"))
        push!(output_lines, @sprintf("%-30s%10.0f K", "Initial velocities:", sim.temperature))
    end

    push!(output_lines, "")
    push!(output_lines, @sprintf("%-30s%10.3f nm", "Lower cutoff radius:", sim.rij_min))
    push!(output_lines, @sprintf("%-30s%10.3f nm", "Upper cutoff radius:", sim.rij_cutoff))
    push!(output_lines, "----------------------------------------------------------")

    if !isnothing(elapsed_time)
        time_per_time_step = elapsed_time / sim.n_steps
        push!(output_lines, Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
        push!(output_lines, "")
        push!(output_lines, @sprintf("%-30s%10.3f s", "Elapsed time:", elapsed_time))
        if time_per_time_step > 1
            push!(output_lines, @sprintf("%-30s%10.3f s", "Elapsed time per time step:", time_per_time_step))
        else
            push!(output_lines, @sprintf("%-30s%10.3f ms", "Elapsed time per time step:", time_per_time_step*1000 ))
        end
        push!(output_lines, "----------------------------------------------------------")
    end
    push!(output_lines, @sprintf("%-30s%10s", "Used backend:", string(get_backend(ps.force))))
    push!(output_lines, @sprintf("%-30s%10s", "Accuracy:", string(eltype(ps.force))))
    push!(output_lines, "----------------------------------------------------------")
    push!(output_lines, "END")
    push!(output_lines, "----------------------------------------------------------")

    open(filename_base * ".out", "w") do file
        write(file,join(output_lines, "\n"))
    end
    
end