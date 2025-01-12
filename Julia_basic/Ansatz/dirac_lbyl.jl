# This code constructs the tightbinding hamiltonian for the Dirac (pi-flux) ansatz on the square lattice, with a LxL Hamiltonian. 
# Note: L denotes the total number of sites, not linear system size.
# Please see the sq_constants.jl file for all the lattice parameters I use.

# We're restricting ourselves to the case where the hopping hamiltonian is exactly the same for both the up and down spins. 
# This is why an LxL Hamiltonian is sufficient, instead of a 2Lx2L Hamiltonian.

# It is important to ensure that the many body ground state is well defined i.e. we have a closed shell configuration i.e there is a gap at the fermi level between the last occupied state and the first unoccupied state.

using LinearAlgebra
using Printf
using DelimitedFiles
using CSV
using DataFrames
using Pkg
using JLD2

include("/tmpdir/budaraju/Julia_basic/sq_constants.jl")


neighbor_file = "/tmpdir/budaraju/Julia_basic/Neighbor_matrix/all_nearest_neighbors_$(Lx)x$(Ly).jld2"
NNmatrix = JLD2.load(neighbor_file, "NNmatrix")

# Initialize Hamiltonian
H = Complex{Float64}[0.0 + 0.0im for _ in 1:L, _ in 1:L]

# Construct the Hamiltonian
for i in 1:L
    b1 = NNmatrix[i, 1]  # Neighbor to the right
    b4 = NNmatrix[i, 2]  # Neighbor below

    # Set hopping amplitude for Dirac state
    tnn = if iseven(i)
        t1
    else
        -t1
    end

    # Update Hamiltonian matrix elements
    H[b1, i] = t1
    H[i, b1] = t1

    H[b4, i] = tnn
    H[i, b4] = tnn

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


if abs(evals[L÷N] - evals[L÷N + 1]) < 1e-14
    println("Warning: Degeneracy at fermi level, its an open shell!")
end

# Output eigenvalues and eigenvectors
#outputname = "evals_$(Lx)_dirac_pbcpbc.csv"
#CSV.write(outputname, df;header=false)

#outputname = "evecs_$(Lx)_dirac_pbcpbc.jld2"
outputname = "evecs_$(Lx)_dirac_apbcpbc.jld2"
jldsave(outputname; evecs)

