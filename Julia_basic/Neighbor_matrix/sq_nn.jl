# Code to print the Matrix of nearest neighbors for each site of the lattice.
# Dimensions of the nearest neighbor matrix ("NNmatrix") are L x fnn, where fnn = 4 for the square lattice.

using LinearAlgebra
using Printf
using DelimitedFiles
using CSV
using DataFrames
using Pkg
using JLD2

include("/tmpdir/budaraju/Julia_basic/sq_constants.jl")

# Function to find the nearest neighbor matrix
function NN(M::Matrix{Int64})
    NNmatrix = zeros(Int, L, fnn)
    for label in 1:L
        # Find coordinates of the site
        coords = findfirst(c -> M[c] == label, CartesianIndices(M))
        x, y = Tuple(coords)  # Convert CartesianIndex to (x, y)
       
        # Nearest neighbors
        NNmatrix[label, 2] = M[mod1(x + 1, Ly), y]  # Below
        NNmatrix[label, 4] = M[mod1(x - 1, Ly), y]  # Above
        NNmatrix[label, 3] = M[x, mod1(y - 1, Lx)]  # Left
        NNmatrix[label, 1] = M[x, mod1(y + 1, Lx)]  # Right
    end
    return NNmatrix
end


# Main script
M = collect(reshape(1:L, Lx, Ly)')  # Label the lattice sites
#println(M)  # Output type of M

#println(M[1,3], M[1,4])
NNmatrix = NN(M)
#println(NNmatrix)  # Print the nearest neighbor matrix


# Write the nearest neighbor matrix to a CSV file
outputname = "all_nearest_neighbors_$(Lx)x$(Ly).jld2"
#CSV.write(outputname, DataFrame(NNmatrix, :auto);header=false)
jldsave(outputname; NNmatrix)


println("Nearest neighbor matrix written to $outputname")
