using GLMakie
using DelimitedFiles
using Statistics

# Install from https://github.com/tillhanke/EntMix/tree/main, see https://doi.org/10.1021/acs.jpclett.4c02819
# Steps to install:
# - Clone to subdirectory ./EntMix
# - In Julia Pkg Manager mode (]) install using: dev "./EntMix"
using EntMix

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
                    push!(types, line[1])
                    append!(positions,parse.(Float64, line[2:end]))
                end
            end
        end
        return reshape(positions, (3, n_particles, frames)), types
    end

    return simulationResults(a,b,c,d,e,f,g,h,types,traj)
end

data = importSimulationResults("simulations/NeArTest")

average_x_Ar = [mean(traj[1,1:100]) for traj in eachslice(data.trajectory, dims=3)]
average_y_Ar = [mean(traj[2,1:100]) for traj in eachslice(data.trajectory, dims=3)]
average_z_Ar = [mean(traj[3,1:100]) for traj in eachslice(data.trajectory, dims=3)]

average_x_Ne = [mean(traj[1,101:200]) for traj in eachslice(data.trajectory, dims=3)]
average_y_Ne = [mean(traj[2,101:200]) for traj in eachslice(data.trajectory, dims=3)]
average_z_Ne = [mean(traj[3,101:200]) for traj in eachslice(data.trajectory, dims=3)]

