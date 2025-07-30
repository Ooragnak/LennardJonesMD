using GLMakie
using DelimitedFiles
using Statistics
using ColorSchemes
using LaTeXStrings

############### CONSTANTS ###############
using PhysicalConstants: CODATA2022 as constants

R = constants.R.val
elementary_charge = constants.ElementaryCharge.val
Avogadro = constants.AvogadroConstant.val
Boltzmann = constants.BoltzmannConstant.val
#########################################

#----------------------------------------------------------------
#   I M P O R T I N G 
#----------------------------------------------------------------

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
            entropy += -n * Boltzmann * (x_A*log(x_A) + x_B*log(x_B))
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

function get_collections(data)
    collections = (findall(x -> x == type, data.particleTypes) for type in unique(data.particleTypes))
    return collections
end

#----------------------------------------------------------------
#   P L O T T I N G
#----------------------------------------------------------------

simpleColors() = Theme(
    palette = Attributes(color = ColorSchemes.tol_bright, linestyle = [:solid, :dash, :dot]),
    colormap = :lipari,
    Band = Attributes(
        alpha = 0.1,
        cycle = [:color],
    ),
    Lines = Attributes(
        cycle = [:color, :linestyle],
    ),
)

highDPI() = Theme(
    fontsize = 36,
    size=(2000,1500),
    figure_padding = 48,
    Axis = Attributes(
        xautolimitmargin = (0, 0),
        xminorticksvisible = true,
        xminorgridvisible = true,
        yminorgridvisible = true,
        yminorticksvisible = true,
        xminorticks = IntervalsBetween(4),
        yminorticks = IntervalsBetween(4),
    ),
    Colorbar = Attributes(
        minorticksvisible = true,
        minorticks = IntervalsBetween(4),
    ),
)

set_theme!(merge(simpleColors(),highDPI(), theme_latexfonts()))

function compare_mixing(NVE_simulations, NVT_simulations, NVE_labels, NVT_labels, timesteps; entropy_split = (4,4,4))
    f = Figure(size = (2000, 1200))
    ax = Axis(f[1:2,1], xlabel = L"t \text{ [ns]}", ylabel = L"\Delta S_\text{mix} \text{ (approx.)}")

    NVEs = []
    NVTs = []

    colors = ColorSchemes.tol_bright

    for (i,sim) in enumerate(NVE_simulations)
        tvals = sim.dt .* 1e-3 .* collect(0:timesteps-1)
        entropies = [approximate_entropy(pos, sim.boxlength * 10, get_collections(sim)..., split = entropy_split) for pos in eachslice(sim.trajectory, dims = 3)[1:timesteps]] ./ sim.n_particles .* Avogadro
        p = lines!(ax, tvals, entropies, color = colors[i])
        push!(NVEs, p)
    end

    for (i,sim) in enumerate(NVT_simulations)
        tvals = sim.dt .* 1e-3 .* collect(0:timesteps-1)
        entropies = [approximate_entropy(pos, sim.boxlength * 10, get_collections(sim) ..., split = entropy_split) for pos in eachslice(sim.trajectory, dims = 3)[1:timesteps]] ./ sim.n_particles .* Avogadro
        p = lines!(ax, tvals, entropies, color = colors[i], linestyle = :dash)
        push!(NVTs, p)
    end

    Legend(f[1,2], NVEs, NVE_labels, "NVE")
    Legend(f[2,2], NVTs, NVT_labels, "NVT")
    return f
end

function plot_mixing(data, types, radii; entropy_split = (4,4,4))
    f = Figure(size = (2000, 1200))
    ax3d1 = Axis3(f[1:6,3], xlabel = "x", ylabel = "y", zlabel = "z", title = "Initial", azimuth = 0.4π)
    ax3d2 = Axis3(f[7:12,3], xlabel = "x", ylabel = "y", zlabel = "z", title = "Final"  , azimuth = 0.4π)
    ax1 = Axis(f[1:4,1:2], ylabel = types[1] * "\n Position [nm]")
    ax2 = Axis(f[5:8,1:2], ylabel = types[2] * "\n Position [nm]")
    ax3 = Axis(f[9:12,1:2], xlabel = L"t \text{ [ns]}", ylabel = L"\Delta S_\text{mix} \text{ (approx.)}")
    hidedecorations!(ax3d1)
    hidedecorations!(ax3d2)

    hidexdecorations!(ax1, grid = false, minorgrid = false)
    hidexdecorations!(ax2, grid = false, minorgrid = false)

    A, B = get_collections(data)
    meshscatter!(ax3d1, data.trajectory[:,A,1], markersize = radii[1], label = types[1], color = ColorSchemes.tol_bright[1])
    meshscatter!(ax3d1, data.trajectory[:,B,1], markersize = radii[2], label = types[2], color = ColorSchemes.tol_bright[2])

    meshscatter!(ax3d2, data.trajectory[:,A,end], markersize = radii[1], label = types[1], color = ColorSchemes.tol_bright[1])
    meshscatter!(ax3d2, data.trajectory[:,B,end], markersize = radii[2], label = types[2], color = ColorSchemes.tol_bright[2])

    labels = [L"\bar{x}", L"\bar{y}", L"\bar{z}"]
    tvals = data.dt .* 1e-3 .* collect(0:length(data.temperatures)-1)

    for i in 1:3
        averageA = [mean(traj[i,A]) for traj in eachslice(data.trajectory, dims=3)]
        averageB = [mean(traj[i,B]) for traj in eachslice(data.trajectory, dims=3)]
        lines!(ax1, tvals, averageA, label = labels[i])
        lines!(ax2, tvals, averageB, label = labels[i])
    end

    entropies = [approximate_entropy(pos, data.boxlength * 10, A, B, split = entropy_split) for pos in eachslice(data.trajectory, dims = 3)] ./ data.n_particles * Avogadro

    lines!(ax3, tvals, entropies, color = :black)

    axislegend(ax3d1)
    axislegend(ax3d2)
    axislegend(ax1)
    axislegend(ax2)
    return f
end

function plot_phase_Separation(data, types, radii; entropy_split = (2,2,2))
    f = Figure(size = (2000, 1200))
    ax3d1 = Axis3(f[1:6,3], xlabel = "x", ylabel = "y", zlabel = "z", title = "Initial", azimuth = 0.4π)
    ax3d2 = Axis3(f[7:12,3], xlabel = "x", ylabel = "y", zlabel = "z", title = "Final"  , azimuth = 0.4π)
    ax1 = Axis(f[1:4,1:2], ylabel = L"E \text{ [kJ mol}^{-1}\text{]}")
    ax2 = Axis(f[5:8,1:2], ylabel = L"T \text{ [K]}")
    ax3 = Axis(f[9:12,1:2], xlabel = L"t \text{ [ns]}", ylabel = L"\Delta S_\text{mix} \text{ (approx.)}")
    hidedecorations!(ax3d1)
    hidedecorations!(ax3d2)

    hidexdecorations!(ax1, grid = false, minorgrid = false)
    hidexdecorations!(ax2, grid = false, minorgrid = false)

    A, B = get_collections(data)
    meshscatter!(ax3d1, data.trajectory[:,A,1], markersize = radii[1], label = types[1], color = ColorSchemes.tol_bright[1])
    meshscatter!(ax3d1, data.trajectory[:,B,1], markersize = radii[2], label = types[2], color = ColorSchemes.tol_bright[2])

    meshscatter!(ax3d2, data.trajectory[:,A,end], markersize = radii[1], label = types[1], color = ColorSchemes.tol_bright[1])
    meshscatter!(ax3d2, data.trajectory[:,B,end], markersize = radii[2], label = types[2], color = ColorSchemes.tol_bright[2])

    tvals = data.dt .* 1e-3 .* collect(0:length(data.temperatures)-1)
    E_kin = data.kinetic_energies
    E_pot = data.potential_energies
    E_tot = E_kin + E_pot

    lines!(ax1, tvals, E_kin, label = L"E_\text{kin}")
    lines!(ax1, tvals, E_pot, label = L"E_\text{pot}")
    lines!(ax1, tvals, E_tot, label = L"E_\text{total}")

    lines!(ax2, tvals, data.temperatures)

    entropies = [approximate_entropy(pos, data.boxlength * 10, A, B, split = entropy_split) for pos in eachslice(data.trajectory, dims = 3)] ./ data.n_particles * Avogadro

    lines!(ax3, tvals, entropies, color = :black)

    axislegend(ax3d1)
    axislegend(ax3d2)

    axislegend(ax1)
    return f
end

function LJpot(rs, ϵ, σ)
    return @. 4 * ϵ * ((σ/rs)^12 - (σ/rs)^6)
end