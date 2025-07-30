include("LJ_analysis.jl")

# Simulation output files are available upon request
NVE_300 = importSimulationResults("simulations/NeAr_Large_NVE_300")
NVE_250 = importSimulationResults("simulations/NeAr_Large_NVE_250")
NVE_200 = importSimulationResults("simulations/NeAr_Large_NVE_200")
NVE_150 = importSimulationResults("simulations/NeAr_Large_NVE_150")
NVE_100 = importSimulationResults("simulations/NeAr_Large_NVE_100")
NVT_300 = importSimulationResults("simulations/NeAr_Large_NVT_300")
NVT_250 = importSimulationResults("simulations/NeAr_Large_NVT_250")
NVT_200 = importSimulationResults("simulations/NeAr_Large_NVT_200")
NVT_150 = importSimulationResults("simulations/NeAr_Large_NVT_150")
NVT_100 = importSimulationResults("simulations/NeAr_Large_NVT_100")

NVE_labels = [L"300 \text{ K}", L"250 \text{ K}", L"200 \text{ K}", L"150 \text{ K}", L"100 \text{ K}"]
NVT_labels = NVE_labels

f = compare_mixing([NVE_300, NVE_250, NVE_200, NVE_150, NVE_100], [NVT_300, NVT_250, NVT_200, NVT_150, NVT_100], NVE_labels, NVT_labels, 5000, entropy_split = (2,2,2))

save("plots/mixing_entropy.png", f, px_per_unit = 2)


f2 = plot_mixing(NVT_300, ["Neon", "Argon"], [1.54, 1.88])

save("plots/mixing_complete.png", f2, px_per_unit = 2)

PhaseSeperationData = importSimulationResults("simulations/NeRn_phase_separation_NVT")
PhaseSeperationDataNVE = importSimulationResults("simulations/NeRn_phase_separation_NVE_constant_temperature")
PhaseSeperationDataNVE_alt = importSimulationResults("simulations/NeRn_phase_separation")


f3 = plot_mixing(PhaseSeperationDataNVE, ["Neon", "Radon"], [1.54, 2.20])

save("plots/phase_separation.png", f3, px_per_unit = 2)

f4 = plot_phase_Separation(PhaseSeperationDataNVE_alt, ["Neon", "Radon"], [1.54, 2.20], entropy_split = (8,8,8))

save("plots/phase_separation_energies.png", f4, px_per_unit = 2)

f5 = Figure(size = (1200, 1200))
ax = Axis(f5[1,1], xlabel = L"r \text{ [Å]}", ylabel = L"V_{LJ} / k_B \text{ [K]}", title = "Lennard-Jones potential for selected gases")
ylims!(ax, (-350,350))

epsilon = [33.921, 116.79, 300]
sigmas = [2.801, 3.395, 4.17]
labels = ["Ne", "Ar", "Rn"]

for (i,l) in enumerate(labels)
    rs = sigmas[i]-0.7:0.001:10
    lines!(ax, rs, LJpot.(rs, epsilon[i], sigmas[i]), label = labels[i])
end
axislegend(ax)
save("plots/Lennard_Jones.png", f5, px_per_unit = 2)
