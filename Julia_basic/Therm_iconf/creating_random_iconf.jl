# Basically just creating a random array of +1 (up spin) and -1 (down spins), both equal in number = L/2.
# I will use this as an initial configuration for my vmc simulation. 

using LinearAlgebra
using Printf
using DelimitedFiles
using CSV
using DataFrames
using Pkg
using JLD2

include("/tmpdir/budaraju/Julia_basic/sq_constants.jl")


function create_random_array(L::Int)
    array = vcat(ones(Int, L ÷ 2), -ones(Int, L ÷ 2))  # Create equal number of 1's and -1's, vcat: vertically concatenates the two arrays into a single array.
    shuffle!(array)  # Randomize the order of elements
    return array
end

random_array = create_random_array(L)
println(random_array)

# Count occurrences of 1's and -1's
count_upspin = count(x -> x == 1, random_array)
count_downspin = count(x -> x == -1, random_array)

# Print the results
println("Number of up spins: $count_upspin")
println("Number of down spins: $count_downspin")

outputname = "initial_iconf_$(Lx).jld2"
#CSV.write(outputname, DataFrame(NNmatrix, :auto);header=false)
jldsave(outputname; random_array)
