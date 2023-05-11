#####################################################
# Code uses inverse iterations to obtain eigenvalues of batches of tridiagonal matrices
########################################################
using LinearAlgebra, Distributions, Plots, LinearAlgebra,  BenchmarkTools,Revise, CUDA, Cthulhu, StatsPlots
include("tdma.jl")

test_gpu()
bench_gpu()
Tn(n,b)=SymTridiagonal(rand(Normal(0,sqrt(2b)),n),  [rand(Chi(i*b)) for i=(n-1):-1:1])

#test accuracy
trials=2
n=4
input_matrices=[Tn(n,1) for _ in 1:trials];
v=[randn(n) for _ in 1:trials];
(batched_upper, batched_middle, batched_lower, batched_rhs) =createCuDiagonals(input_matrices, v);
results=batched_tridiag_solver(batched_upper, batched_middle, batched_lower, batched_rhs, length(input_matrices), size(input_matrices[1],1));
for i in 1:2
    println(Array(results[:,i]) ≈ input_matrices[i]\v[i])
end
#benchmarking vs CPU
trials=100
n=1000000
n_reduced=round(Int,10 *(n)^(1/3))
input_matrices=[Tn(n,1)[1:n_reduced,1:n_reduced] for _ in 1:trials];
v=[randn(n_reduced) for _ in 1:trials];
(batched_upper, batched_middle, batched_lower, batched_rhs) =createCuDiagonals(input_matrices, v)
@time ([(input_matrices[i]-2*I)\v[i] for i in 1:trials]);
@time batched_tridiag_solver(batched_upper, bacthed_middle, batched_lower, batched_rhs, length(input_matrices), size(input_matrices[1],1))


#inverse iteration for eigvals of tridiagonals
n_values=[10^6,10^7]
trials=1000
lambda=  −1.2065335745820
totalerrors=zeros(length(n_values), trials)
totallambdas=zeros(length(n_values), trials)
for (i,n) in enumerate(n_values)
    # random matrix theory A. Edelman p 104:  use upper left 10n^1/3 × 10n1/3 of the matrix for good estimates of eigenvalues of tridiagonal matrices
    n_reduced=round(Int,10 *(n)^(1/3))
    input_matrices=[Tn(n,1)[1:n_reduced,1:n_reduced] for _ in 1:trials]
    global lambda_rep=  [lambda for _ in 1:trials]
    global myerror =[1 for _ in 1:trials]
    global v=[randn(n_reduced) for _ in 1:trials]
    v=v./norm.(v)
    global myerror=1
    (batched_upper, batched_middle, batched_lower, batched_rhs) =createCuDiagonals(input_matrices, v)
    input_matrices=[CuMatrix(input_matrices[i]) for i in eachindex(input_matrices)]
    for i=1:1000
        v=batched_tridiag_solver(batched_upper, substract_from_stridebatched(batched_middle, lambda_rep), batched_lower, batched_rhs, length(input_matrices), size(input_matrices[1],1))
        v=v./CuArray(norm.(eachcol(v))')
        batched_rhs=StrideBatched(v[:], batched_rhs.count, batched_rhs.stride, batched_rhs.shape)
        lambdanew=[(view(v,:,i)'*input_matrices[i]*view(v,:,i))[1] for i in 1:trials] #probably needs to be optimized by batching on GPU: https://github.com/JuliaGPU/CUDA.jl/blob/ad4e9b5c73b8471c4698a739dcbedb73b682c152/lib/cublas/wrappers.jl#L977
        myerror=abs.(lambda_rep.-lambdanew)
        if maximum(myerror)<0.001
            break
        end
        lambda_rep=lambdanew
    end

    totalerrors[i,:]=myerror
    totallambdas[i,:]=lambda_rep

end
plot( totallambdas', seriestype = :violin, fillalpha=0.5, linealpha=0.5, markeralpha=0.5, xticks=(1:length(n_values),n_values./10^6), xlabel="n (10^6) ")
plot( totalerrors', seriestype = :violin, fillalpha=0.5, linealpha=0.5, markeralpha=0.5, xticks=(1:length(n_values),n_values./10^6), xlabel="n (10^6) ", yscale=:log10)
