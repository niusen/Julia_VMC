# Want to code for the tightbinding model for landau gauge, and compare the spectrum with the C++ code I have. 
# Here we have a flux of 2pi*m/L in every plaquette.
# Here too I assume the same hopping hamiltonian for the up and down spin, so the Hamiltonian and the eigenvector matrix will be LxL.

# Update: I confirmed that the spectrum matches the one obtained by the C++ code.

using LinearAlgebra
using Printf
using DelimitedFiles
using CSV
using DataFrames
using Pkg
using JLD2

include("/tmpdir/budaraju/Julia_basic/sq_constants.jl")


function check_flux(H)
    # Loop over each site
    for x in 1:Lx
        for y in 1:Ly
            # Get the indices for the top left corner of the square
            site1 = (x - 1) * Ly + y           # top-left corner

            # Apply periodic boundary conditions using % operator
            site2 = (x - 1) * Ly + y % Ly + 1  # top-right corner
            site3 = (x%Lx) * Ly + y # bottom-left corner
            site4 = (x%Lx) * Ly + y % Ly + 1  # bottom-right corner
            # Multiply the hoppings around the plaquette
            flux1 = H[site1, site2] * H[site2, site4] * H[site4, site3] * H[site3, site1]

            # Print the product (flux) for this plaquette
            println("Plaquette at site ($x, $y) has flux: $flux1")
        end
    end
end


neighbor_file = "/tmpdir/budaraju/Julia_basic/Neighbor_matrix/all_nearest_neighbors_$(Lx)x$(Ly).jld2"
NNmatrix = JLD2.load(neighbor_file, "NNmatrix")

# Initialize Hamiltonian
H = Complex{Float64}[0.0 + 0.0im for _ in 1:L, _ in 1:L]

p = 1 # How much flux through each plaquette, in units of 2pi/Lx

# Construct the Hamiltonian
for i in 1:L
    b1 = NNmatrix[i, 1]  # Neighbor to the right
    b4 = NNmatrix[i, 2]  # Neighbor below

    rem = i % Lx
    flux = 2*pi*p*rem/Lx


    # Update Hamiltonian matrix elements
    H[b1, i] = t1
    H[i, b1] = t1

    H[b4, i] = t1*(cos(flux) + im*sin(flux))
    H[i, b4] = t1*(cos(flux) - im*sin(flux))

    # Apply boundary conditions
    if (i % Lx) == 0  # Right boundary
        H[b1, i] *= cos(theta_bc_right) + im * sin(theta_bc_right)
        H[i, b1] *= cos(theta_bc_right) - im * sin(theta_bc_right)
    end

    if i > (L - Lx)  # Bottom boundary
        H[b4, i] *= cos(theta_bc_down) + im * sin(theta_bc_down)
        H[i, b4] *= cos(theta_bc_down) - im * sin(theta_bc_down)
    end
end

# Check Hermiticity
println("Hermiticity check (should be ~0): ", norm(H - H')) 
# norm(H-H') computes the Frobenius norm of the matrix H-H'. This is the square root of the sums of absolute squared of all elements
# So if the frobenius norm is 0, this necessarily implies that the matrix has all zero entries.

# Diagonalize the Hamiltonian
evals, evecs = eigen(H)

df = DataFrame(eigenvalues=evals)

# Output eigenvalues and eigenvectors
#outputname = "evals_$(Lx)_landau_flux_$(p)_pbcpbc.csv"
#CSV.write(outputname, df;header=false)

outputname = "evecs_$(Lx)_landau_flux_$(p)_pbcpbc.jld2"
jldsave(outputname; evecs)

check_flux(H)
