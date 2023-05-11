#from https://gist.github.com/maleadt/1ec91b3b12ede9898958c95596cabe8b Tim Besard

# experimentation with batched tridiagonal solvers on the GPU for Oceananigans.jl
#
# - reference serial CPU implementation
# - batched GPU implementation using cuSPARSE (fastest)
# - batched GPU implementation based on the serial CPU implementation (slow but flexible)
# - parallel GPU implementation (potentially fast and flexible)
#
# see `test_batched` and `bench_batched`

using LinearAlgebra

using CUDA
using CUDA: i32

using Test
using Statistics
using BenchmarkTools

using InteractiveUtils

import LinearAlgebra: SingularException, HermOrSym, AbstractTriangular
############################################################################################
"""
    gtsvStridedBatch!(dl::CuVector, d::CuVector, du::CuVector, X::CuVector, batchCount::Integer, batchStride::Integer)

Performs the batched solution of `A[i] \\ B[i]` where `A[i]` is a tridiagonal matrix, with
lower diagonal `dl`, main diagonal `d`, and upper diagonal `du`. `batchCount` determines
how many elements there are in the batch in total (how many `A`s?), and `batchStride` sets
the separation of each item in the batch (it must be at least `m`, the matrix dimension).
"""



function gtsvStridedBatch!(dl::CuVector, d::CuVector, du::CuVector, X::CuVector, batchCount::Integer, batchStride::Integer) end
for (bufferf, fname,elty) in ((:cusparseSgtsv2StridedBatch_bufferSizeExt, :cusparseSgtsv2StridedBatch, :Float32),
                              (:cusparseDgtsv2StridedBatch_bufferSizeExt, :cusparseDgtsv2StridedBatch, :Float64),
                              (:cusparseCgtsv2StridedBatch_bufferSizeExt, :cusparseCgtsv2StridedBatch, :ComplexF32),
                              (:cusparseZgtsv2StridedBatch_bufferSizeExt, :cusparseZgtsv2StridedBatch, :ComplexF64))
    @eval begin
        function gtsvStridedBatch!(dl::CuVector{$elty},
                                   d::CuVector{$elty},
                                   du::CuVector{$elty},
                                   X::CuVector{$elty},
                                   batchCount::Integer,
                                   batchStride::Integer)
            m = div(length(X),batchCount)

            function bufferSize()
                out = Ref{Csize_t}(1)
                CUDA.CUSPARSE.$bufferf(CUDA.CUSPARSE.handle(), m, dl, d, du, X,batchCount, batchStride, out)
                return out[]
            end
            CUDA.with_workspace(bufferSize()) do buffer
                CUDA.CUSPARSE.$fname(CUDA.CUSPARSE.handle(), m, dl, d, du, X, batchCount, batchStride, buffer)
            end
            X
        end
        function gtsvStridedBatch(dl::CuVector{$elty},
                                  d::CuVector{$elty},
                                  du::CuVector{$elty},
                                  X::CuVector{$elty},
                                  batchCount::Integer,
                                  batchStride::Integer)
            gtsvStridedBatch!(dl,d,du,copy(X),batchCount,batchStride)
        end
    end
end

############################################################################################
# CPU implementation per Numerical Recipes, Press et. al 1992 (sec 2.4)

using Base: require_one_based_indexing

function LinearAlgebra.:(\)(M::Tridiagonal{T}, rhs::AbstractVector{T}) where {T}
    require_one_based_indexing(M, rhs)
    N = length(rhs)
    (size(M,1) == size(M,2) == N) || throw(DimensionMismatch())
    phi = similar(rhs)
    gamma = similar(rhs)

    @inbounds beta = M.d[1]
    @inbounds phi[1] = rhs[1] / beta

    @inbounds for j=2:N
        gamma[j] = M.du[j-1] / beta
        beta = M.d[j] - M.dl[j-1]*gamma[j]
        phi[j] = (rhs[j] - M.dl[j-1]*phi[j-1]) / beta
    end

    @inbounds for j=1:N-1
        k = N-j
        phi[k] = phi[k] - gamma[k+1]*phi[k+1]
    end

    return phi
end

function test_cpu(;N=13, T=Float64)
    # allocate data
    t = Tridiagonal(rand(T, N,N))
    rhs = rand(T, N)

    # solve
    phi = t \ rhs
    @test t * phi ≈ rhs
    @test phi ≈ invoke(\, Tuple{AbstractMatrix, AbstractVector}, t, rhs)
end

function bench_cpu(;N=256, T=Float32)
    # allocate data
    t = Tridiagonal(rand(T, N, N))
    rhs = rand(T, N)

    suite = BenchmarkGroup()

    suite["Base"] = @benchmarkable invoke(\, Tuple{AbstractMatrix, AbstractVector}, $t, $rhs)
    suite["new"] = @benchmarkable $t \ $rhs

    warmup(suite)
    results = run(suite)
    println(results)
    display(judge(median(results["new"]), median(results["Base"])))
end

#=

N=4
  "Base" => Trial(216.000 ns)
  "new" => Trial(86.000 ns)
BenchmarkTools.TrialJudgement:
  time:   -39.30% => improvement (5.00% tolerance)
  memory: -73.33% => improvement (1.00% tolerance)


N=256
  "Base" => Trial(4.331 μs)
  "new" => Trial(2.430 μs)
BenchmarkTools.TrialJudgement:
  time:   -43.84% => improvement (5.00% tolerance)
  memory: -71.32% => improvement (1.00% tolerance)

N=2048
  "Base" => Trial(34.293 μs)
  "new" => Trial(19.305 μs)
BenchmarkTools.TrialJudgement:
  time:   -48.75% => improvement (5.00% tolerance)
  memory: -71.43% => improvement (1.00% tolerance)

=#

############################################################################################

# GPU implementations

# TODO: would be great if we could do length and size on Batches

# TODO: batch(...) function; bunch of arrays or array+count+stride+...

using Adapt

abstract type Batched{AT} end

"""
A batch of `count` arrays that are each `size` large, stored in `data` with each batch
separated by `stride` elements.
"""
struct StrideBatched{AT,N} <: Batched{AT}
    data::AT
    count::Int
    stride::Int
    shape::Dims{N}
end

StrideBatched(data::AT, count::Integer, stride::Integer, shape::Dims{N}) where {AT, N} =
    StrideBatched{AT,N}(data, count, stride, shape)

Adapt.adapt_structure(to, b::StrideBatched) =
    StrideBatched(adapt(to, b.data), b.count, b.stride, b.shape)

@inline Base.@propagate_inbounds function Base.getindex(b::StrideBatched, i::Integer)
    offset = b.stride * (i-1) + 1
    length = prod(b.shape)
    array = view(b.data, offset:(offset+length-1))
    reshape(array, b.shape)
end
function substract_from_stridebatched(sb::StrideBatched, substract::Array)
    num= repeat(substract',sb.stride)[:]
    num=num |>cu
    return StrideBatched(sb.data.-num, sb.count, sb.stride, sb.shape)
end

Base.similar(b::StrideBatched) = StrideBatched(similar(b.data), b.count, b.stride, b.shape)
Base.copy(b::StrideBatched) = StrideBatched(copy(b.data), b.count, b.stride, b.shape)

struct ListBatched{AT,N} <: Batched{AT}
    list::Vector{AT}
    count::Int
    shape::Dims{N}
end

function ListBatched(list::AbstractVector{AT}) where {AT}
    count = length(list)
    shape = size(first(list))
    @assert all(isequal(shape), size.(list))
    ListBatched{AT,CT}(list, count, shape)
end

Adapt.adapt_structure(to, b::ListBatched) = ListBatched(adapt(to, b.batches))

@inline Base.@propagate_inbounds Base.getindex(b::ListBatched, i::Int) = b.batches[i]


Base._reshape(parent::CuDeviceArray, dims::Dims) = CuDeviceArray(dims, pointer(parent))
Base._reshape(parent::CuDeviceArray, dims::Tuple{Int}) = CuDeviceArray(dims, pointer(parent))


# wrapper for convenient benchmarking
function gtsvStridedBatch!(b_a::StrideBatched{CuVector{T, CUDA.Mem.DeviceBuffer}},   # lower diagonal
                                    b_b::StrideBatched{CuVector{T, CUDA.Mem.DeviceBuffer}},   # main diagonal
                                    b_c::StrideBatched{CuVector{T, CUDA.Mem.DeviceBuffer}},   # upper diagonal
                                    b_d::StrideBatched{CuVector{T, CUDA.Mem.DeviceBuffer}},   # right hand side
                                    b_x::StrideBatched{CuVector{T, CUDA.Mem.DeviceBuffer}},   # output
                                    count, stride) where {T}
    @assert b_a.count == b_b.count == b_d.count == b_d.count == b_x.count == count
    @assert b_a.shape == b_b.shape == b_d.shape == b_d.shape == b_x.shape == (stride,)
    @assert b_a.stride == b_b.stride == b_d.stride == b_d.stride == b_x.stride == stride

    copyto!(b_x.data, b_d.data)
    gtsvStridedBatch!(b_a.data, b_b.data, b_c.data, b_x.data, count, stride)
    b_x
end


# parallel tridiagonal solver
function tdma_strided_batched(a, b, c, d, count, stride)
    x = similar(d)
    tdma_strided_batched!(a, b, c, d, x, count, stride)
    return x
end
function tdma_strided_batched!(d_a::CuVector{T}, d_b::CuVector{T}, d_c::CuVector{T},
                               d_d::CuVector{T}, d_x::CuVector{T},
                               count::Integer, stride::Integer) where {T}
    N = length(d_d) ÷ count
    @assert length(d_a) == length(d_b) == length(d_c) == N*count

    b_a = StrideBatched(d_a, count, stride, (N,))
    b_b = StrideBatched(d_b, count, stride, (N,))
    b_c = StrideBatched(d_c, count, stride, (N,))
    b_d = StrideBatched(d_d, count, stride, (N,))
    b_x = StrideBatched(d_x, count, stride, (N,))

    if ispow2(N)
        tdma_batched_cr!(b_a, b_b, b_c, b_d, b_x, count, N)
    else
        tdma_batched_thomas!(b_a, b_b, b_c, b_d, b_x, count, N)
    end
end

# Thomas algorithm on each thread
# - identically shaped batches
function tdma_batched_thomas!(b_a::StrideBatched{<:CuVector{T}},   # lower diagonal
                              b_b::StrideBatched{<:CuVector{T}},   # main diagonal
                              b_c::StrideBatched{<:CuVector{T}},   # upper diagonal
                              b_d::StrideBatched{<:CuVector{T}},   # right hand side
                              b_x::StrideBatched{<:CuVector{T}},   # output
                              count, N) where {T}
    @assert b_a.count == b_b.count == b_d.count == b_d.count == b_x.count == count
    @assert b_a.shape == b_b.shape == b_d.shape == b_d.shape == b_x.shape == (N,)

    function kernel(b_a, b_b, b_c, b_d, b_x, b_workspace)
        # load batches into device arays

        batch = (blockIdx().x-1) * blockDim().x + threadIdx().x

        @inbounds if batch <= count

            a = b_a[batch]
            b = b_b[batch]
            c = b_c[batch]
            d = b_d[batch]
            x = b_x[batch]

            workspace = b_workspace[batch]


            # Thomas algorithm

            beta = b[1]
            x[1] = d[1] / beta

            for j = 2:N
                k = j
                workspace[k] = c[k-1] / beta
                beta = b[k] - a[k]*workspace[k]
                x[k] = (d[k] - a[k]*x[k-1])/beta
            end

            for j = 1:N-1
                k = N-j
                x[k] = x[k] - workspace[k+1]*x[k+1]
            end
        end

        return
    end

    b_workspace = similar(b_d)

    function get_config(kernel)
        fun = kernel.fun
        config = launch_configuration(fun)

        # round up to cover all batches
        blocks = (count + config.threads - 1) ÷ config.threads

        return (threads=config.threads, blocks=blocks)
    end

    @cuda kernel(b_a, b_b, b_c, b_d, b_x, b_workspace)

    return
end

# parallel cyclic reduction within each block:
# - identically shaped batches
# - pow2
function tdma_batched_cr!(b_a::StrideBatched{<:CuVector{T}},   # lower diagonal
                          b_b::StrideBatched{<:CuVector{T}},   # main diagonal
                          b_c::StrideBatched{<:CuVector{T}},   # upper diagonal
                          b_d::StrideBatched{<:CuVector{T}},   # right hand side
                          b_x::StrideBatched{<:CuVector{T}},   # output
                          count, N) where {T}
    @warn "Careful, this function returns incorrect answers in some cases"
    @assert b_a.count == b_b.count == b_d.count == b_d.count == b_x.count == count
    @assert b_a.shape == b_b.shape == b_d.shape == b_d.shape == b_x.shape == (N,)
    @assert ispow2(N)

    function kernel(b_a, b_b, b_c, b_d, b_x)
        iterations = floor(Int, log2(Float32(N ÷ 2)))

        @inbounds begin
            # load batches into device arays

            batch = blockIdx().x+1

            d_a = b_a[batch]
            d_b = b_b[batch]
            d_c = b_c[batch]
            d_d = b_d[batch]
            d_x = b_x[batch]


            # load data into shared memory

            thread = threadIdx().x

            a = @cuDynamicSharedMem(T, (N,))
            b = @cuDynamicSharedMem(T, (N,), N*sizeof(T))
            c = @cuDynamicSharedMem(T, (N,), N*sizeof(T)*2)
            d = @cuDynamicSharedMem(T, (N,), N*sizeof(T)*3)
            x = @cuDynamicSharedMem(T, (N,), N*sizeof(T)*4)

            a[thread] = d_a[thread]
            a[thread + blockDim().x] = d_a[thread + blockDim().x]

            b[thread] = d_b[thread]
            b[thread + blockDim().x] = d_b[thread + blockDim().x]

            c[thread] = d_c[thread]
            c[thread + blockDim().x] = d_c[thread + blockDim().x]

            d[thread] = d_d[thread]
            d[thread + blockDim().x] = d_d[thread + blockDim().x]

            sync_threads()

            # forward elimination
            active_threads = blockDim().x
            stride = 1
            for j = 1:iterations
                sync_threads()
                stride *= 2
                delta = stride ÷ 2

                if threadIdx().x <= active_threads
                    i = stride * (threadIdx().x - 1) + stride
                    iLeft = i - delta
                    iRight = i + delta
                    if iRight > N
                        iRight = N
                    end
                    tmp1 = a[i] / b[iLeft]
                    tmp2 = c[i] / b[iRight]
                    b[i] = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2
                    d[i] = d[i] - d[iLeft] * tmp1 - d[iRight] * tmp2
                    a[i] = -a[iLeft] * tmp1
                    c[i] = -c[iRight] * tmp2
                end

                active_threads ÷= 2
            end

            if thread <= 2
                addr1 = stride;
                addr2 = 2 * stride;
                tmp3 = b[addr2]*b[addr1] - c[addr1]*a[addr2]
                x[addr1] = (b[addr2]*d[addr1]-c[addr1]*d[addr2]) / tmp3
                x[addr2] = (d[addr2]*b[addr1]-d[addr1]*a[addr2]) / tmp3
            end

            # backward substitution
            active_threads = 2
            for j = 1:iterations
                delta = stride ÷ 2
                sync_threads()
                if thread <= active_threads
                    i = stride * (thread - 1) + stride ÷ 2
                    if i == delta
                        x[i] = (d[i] - c[i]*x[i+delta]) / b[i]
                    else
                        x[i] = (d[i] - a[i]*x[i-delta] - c[i]*x[i+delta]) / b[i]
                    end
                end
                stride ÷= 2
                active_threads *= 2
            end

            sync_threads()

            # write back to global memory
            d_x[thread] = x[thread]
            d_x[thread + blockDim().x] = x[thread + blockDim().x]
        end

        return
    end

    threads = N ÷ 2
    shmem = 5 * sizeof(T) * N

    @cuda blocks=count threads=threads shmem=shmem kernel(b_a, b_b, b_c, b_d, b_x)

    return
end

function test_gpu(;X=7, Y=13, Z=64, T=Float64)
    # allocate data
    lower = CUDA.rand(T, Z, Y, X)
    lower[1, :, :] .= 0
    upper = CUDA.rand(T, Z, Y, X)
    upper[Z, :, :] .= 0
    middle = CUDA.rand(T, Z, Y, X)
    rhs = CUDA.rand(T, Z, Y, X)

    # batched interface uses 1d vectors
    flat_upper  = reshape(upper,  X*Y*Z)
    flat_middle = reshape(middle, X*Y*Z)
    flat_lower  = reshape(lower,  X*Y*Z)
    flat_rhs    = reshape(rhs,    X*Y*Z)

    # batched wrappers
    batched_upper = StrideBatched(flat_upper, X*Y, Z, (Z,))
    batched_middle = StrideBatched(flat_middle, X*Y, Z, (Z,))
    batched_lower = StrideBatched(flat_lower, X*Y, Z, (Z,))
    batched_rhs = StrideBatched(flat_rhs, X*Y, Z, (Z,))
    batched_out = similar(batched_rhs)

    for f in (gtsvStridedBatch!, tdma_batched_thomas!, tdma_batched_cr!)
        CUDA.@sync f(batched_lower, batched_middle, batched_upper, batched_rhs, batched_out, X*Y, Z)
        out = reshape(batched_out.data, Z, Y, X)

        # verify
        for x in 1:X, y in 1:Y
            a = Array(lower[:, y, x])
            b = Array(middle[:, y, x])
            c = Array(upper[:, y, x])
            u = Array(out[:, y, x])
            d = Array(rhs[:, y, x])

            t = Tridiagonal(a[2:end], b, c[1:end-1])
            @test t * u ≈ d
            break
        end
    end

    return
end

function bench_gpu(;X=256, Y=256, Z=256, T=Float32)
    CUDA.allowscalar(true)

    # allocate data
    lower = CUDA.rand(T, Z, Y, X)
    lower[1, :, :] .= 0
    upper = CUDA.rand(T, Z, Y, X)
    upper[Z, :, :] .= 0
    middle = CUDA.rand(T, Z, Y, X)
    rhs = CUDA.rand(T, Z, Y, X)

    # batched interface uses 1d vectors
    flat_upper  = reshape(upper,  X*Y*Z)
    flat_middle = reshape(middle, X*Y*Z)
    flat_lower  = reshape(lower,  X*Y*Z)
    flat_rhs    = reshape(rhs,    X*Y*Z)

    # batched wrappers
    batched_upper = StrideBatched(flat_upper, X*Y, Z, (Z,))
    batched_middle = StrideBatched(flat_middle, X*Y, Z, (Z,))
    batched_lower = StrideBatched(flat_lower, X*Y, Z, (Z,))
    batched_rhs = StrideBatched(flat_rhs, X*Y, Z, (Z,))
    batched_out = similar(batched_rhs)

    suite = BenchmarkGroup()

    suite["cuSPARSE"] =
        @benchmarkable begin
            CUDA.@sync gtsvStridedBatch!($batched_lower, $batched_middle, $batched_upper,
                                                      $batched_rhs, $batched_out, $X*$Y, $Z)
        end setup=(GC.gc())
    suite["serial Thomas algorithm"] =
        @benchmarkable begin
            CUDA.@sync tdma_batched_thomas!($batched_lower, $batched_middle, $batched_upper,
                                                $batched_rhs, $batched_out, $X*$Y, $Z)
        end setup=(GC.gc())
    suite["parallel cyclic reduction"] =
        @benchmarkable begin
            CUDA.@sync tdma_batched_cr!($batched_lower, $batched_middle, $batched_upper,
                                            $batched_rhs, $batched_out, $X*$Y, $Z)
        end setup=(GC.gc())

    warmup(suite)
    @show results = run(suite)
    judge(median(results["serial Thomas algorithm"]), median(results["cuSPARSE"]))
end

function createCuDiagonals(tridiag_matrices, rhspacked)
    n_matrices=[size(tridiag_matrices[i],1) for i in eachindex(tridiag_matrices)]
    if maximum(n_matrices) != minimum(n_matrices)
        throw(DimensionMismatch("This function only handles matrices with the same number of elements"))
    end
    n=n_matrices[1]
    T=typeof(tridiag_matrices[1][1,1])
    upper = T[]
    mid = T[]
    lower = T[]
    rhs = T[]
    
    for i in 1:length(tridiag_matrices)
        append!(upper, diag(tridiag_matrices[i],1))
        append!(upper, T(0))
        append!(mid, diag(tridiag_matrices[i]))
        append!(lower, T(0))
        append!(lower, diag(tridiag_matrices[i],-1))
        append!(rhs, rhspacked[i])
    end
    upper=upper |>cu
    mid=mid |>cu
    lower=lower |>cu
    rhs= rhs |>cu
    batched_upper = StrideBatched(upper, length(tridiag_matrices), n, (n,))
    batched_middle = StrideBatched(mid, length(tridiag_matrices), n, (n,))
    batched_lower = StrideBatched(lower, length(tridiag_matrices), n, (n,))
    batched_rhs = StrideBatched(rhs, length(tridiag_matrices), n, (n,))
    return (batched_upper, batched_middle, batched_lower, batched_rhs)
end

function batched_tridiag_solver(batched_upper, batched_middle, batched_lower, batched_rhs, batches, batchsize)
    batched_out = similar(batched_rhs)
    CUDA.@sync gtsvStridedBatch!(batched_lower, batched_middle, batched_upper, batched_rhs, batched_out, batches, batchsize)
    out = reshape(batched_out.data, batchsize, batches)
    return out
end