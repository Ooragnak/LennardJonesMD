using GLMakie
using DelimitedFiles
using Statistics

############### CONSTANTS ###############
using PhysicalConstants: CODATA2022 as constants

R = constants.R.val
elementary_charge = constants.ElementaryCharge.val
Avogadro = constants.AvogadroConstant.val
Boltzmann = constants.BoltzmannConstant.val
#########################################

mutable struct simulationResults
    n_particles::Int
    boxlength::Float64
    dt::Float64
    timesteps::Int64
    potential_energies::AbstractVector
    kinetic_energies::AbstractVector
    temperatures::AbstractVector
    pressures::AbstractVector
    particleTypes::AbstractVector
    trajectory::AbstractArray
end



function  importSimulationResults(filenameBase)
    a, b, c, d = open(filenameBase * ".out") do f
        file = readlines(f)
        n_particles = parse(Int64, split(file[5])[end])
        boxlength = parse(Float64, split(file[6])[end - 1])
        dt = parse(Float64, split(file[10])[end - 1])
        timesteps = parse(Int64, split(file[11])[end])
        return n_particles, boxlength, dt, timesteps
    end
    
    e, f, g, h = open(filenameBase * "_ene.dat") do file
        data = readdlm(file, ' ', Float64, '\n', header = false, skipstart = 1)
        return data[:,1], data[:,2], data[:,3], data[:,4]
    end

    traj, types = open(filenameBase * "_pos.xyz") do file
        data = read(file, String)
        n_particles = parse(Int64, split(data, "\n")[1])
        frames = 0
        positions = Float64[]
        types = []
        delim = split(data, "\n")[1] * "\n" * split(data, "\n")[2]

        for frame in split(data, delim)[2:end]
            frames += 1
            for atom in split(frame, "\n")
                if !isempty(atom)
                    line = split(atom)
                    if frames == 1
                        push!(types, line[1])
                    end
                    append!(positions,parse.(Float64, line[2:end]))
                end
            end
        end
        return reshape(positions, (3, n_particles, frames)), types
    end

    return simulationResults(a,b,c,d,e,f,g,h,types,traj)
end

function approximate_entropy(position, box_length, indices_A, indices_B; split=(2,1,1))
    dim, n = size(position)
    @assert dim == 3

    entropy = 0

    xs = range(0, box_length, split[1]+1)
    ys = range(0, box_length, split[2]+1)
    zs = range(0, box_length, split[3]+1)

    boxes = zeros(Bool, (split..., n))

    for (i, ps) in enumerate(eachslice(position, dims=2))
        # added max to ensure value is always at least one in the edgecase ps[j] == 0.0
        ix = max(findfirst(x -> x >= ps[1], xs) - 1, 1)
        iy = max(findfirst(x -> x >= ps[2], ys) - 1, 1)
        iz = max(findfirst(x -> x >= ps[3], zs) - 1, 1)
        boxes[ix,iy,iz,i] = true
    end

    for box in eachslice(boxes, dims = (1,2,3))
        n_A = sum(box[indices_A])
        n_B = sum(box[indices_B])
        n_total = n_A + n_B
        if !iszero(n_A) && !iszero(n_B)
            x_A = n_A / n_total
            x_B = n_B / n_total
            entropy += -n * R * (x_A*log(x_A) + x_B*log(x_B))
        end
    end
    return entropy
end

function distances(positions, box_length)
    dim, n = size(positions)
    @assert dim == 3
    dist = zeros(n,n)
    for i in 1:n
        p1 = positions[:,i]
        for j in 1:i
            p2 = positions[:,j]

            r = p1 - p2
            # Apply periodic boundary conditions to find nearest periodic image
            @. r = r - round(r / box_length) * box_length
            dist[i,j] = norm(r)
            dist[j,i] = dist[i,j]
        end
    end
    return dist
end

data = importSimulationResults("simulations/NeRn_phase_separation_NVE_constant_temperature")

collections = [findall(x -> x == type, data.particleTypes) for type in unique(data.particleTypes)]

A = collections[1]
B = collections[2]

averages = ([],[],[])

for collection in collections
    for i in 1:3
        push!(averages[i],[mean(traj[i,collection]) for traj in eachslice(data.trajectory, dims=3)])
    end
end

r_A = 1.54
r_B = 2.20
timestep = 20000

f, ax3d, p1 = meshscatter(data.trajectory[:,A,timestep], markersize = r_A)
meshscatter!(ax3d, data.trajectory[:,B,timestep], markersize = r_A)

entropies = [approximate_entropy(pos, 160, A, B, split = (10,10,10)) for pos in eachslice(data.trajectory, dims = 3)]