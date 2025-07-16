include("../LJ_gas.jl")
using NPZ

# system
n_particles = 200
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
dt = 0.2          # ps
n_steps = 1000
temperature = 40     # K
box_length = 100      # nm
tau_thermostat = 1  # thermostat coupling constant in 1/ps
rij_min = 1e-2      # nm
NVT = false

# Metadata 
save_frequency = 1 # save every n-th step
filename_base = "simulations/NeArTestEqualityJulia"

ARRAYTPE = Array{Float64} # CPU double precision

sim = SimulationParameters(dt, n_steps, temperature, box_length, tau_thermostat, rij_min, Inf, NVT, save_frequency)

# initialize ParticleSystem 
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

# Python comparison run
t1 = @timed run(`python tests/run_python_simulation.py`)

# Import starting velocities and positions
ps.position = convert(typeof(ps.position), transpose(npzread("tests/initial_position.npy")))
ps.velocity = convert(typeof(ps.velocity), transpose(npzread("tests/initial_velocity.npy")))

# calculate force according to initial positions
calculate_force!(ps, sim)

t2 = @timed for i in 1:n_steps
    simulate_NVE_step!(ps, sim)
end

final_position = transpose(npzread("tests/final_position.npy"))
final_velocity = transpose(npzread("tests/final_velocity.npy"))

final_position_julia = Array(ps.position)
final_velocity_julia = Array(ps.velocity)

println("Simulations run. Runtime:\n Python: $(t1.time) s\n Julia: $(t2.time) s")

if all(final_position .≈ final_position_julia)
    println("Final positions equal.")
else
    println("Final positions deviate. Maximum difference: $(maximum(final_position .- final_position_julia))")
end
if all(final_velocity .≈ final_velocity_julia)
    println("Final velocities equal.")
else
    println("Final velocities deviate. Maximum difference: $(maximum(final_velocity .- final_velocity_julia))")
end