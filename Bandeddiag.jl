#############################################################
# Implementation of banded diagonal solver based on Jandron et al 2017 (https://link.springer.com/article/10.1007/s11075-016-0251-3)
# Author: Evelyne Ringoot
#
#############################################################
# Solves A X = b with A has q bandwith the upper bandwith
#############################################################

using LinearAlgebra, CUDA, SparseArrays, BenchmarkTools, Plots, BenchmarkTools

struct MyBandedMatrix 
    A_band::Array
    bandnumbers::CuArray
    #with zeros where the sub/superdiagonal is not visible e.g. [1 0 0;1 1 0; 0 1 1] becomes [0 1; 1 1; 1 1]
end

function CreateMyBandedMatrix(A, bandnumbers::Array{Int})
    if size(A)[1] != size(A)[2]
        error("Function is currently not defined for non-square matrices")
    end
    A_band=reduce(hcat,[ [zeros(max(-band,0)); diag(A,band) ; zeros(max(band,0))] for band in bandnumbers])
    bandnumbers=bandnumbers |>cu
    return MyBandedMatrix(A_band, bandnumbers );
end

function BandedSolver(A::MyBandedMatrix,b)
    if size(A.A_band)[1]!=length(b)
        error("Size mismatch between A and b");
    end
    bandnumbers=A.bandnumbers
    q=bandnumbers[end]
    n=length(b)
    k=length(bandnumbers)
    
    A_withalpha=A.A_band|>cu
    A_withalpha[end-q+1:end,end].+=1
    A_withalpha[:,1:end-1]./=A_withalpha[:,end]
    solutions=[ones(q,q)-I; zeros(n-q,q) ; zeros(q,q)+I] |>cu
    solutions[q+1:end,:]./=A_withalpha[:,end]
    solution_b= [ones(q);b] |>cu
    solution_b[q+1:end]./=A_withalpha[:,end]
    
    @sync for i in 0:q
        Threads.@spawn begin
            if i==0
                b=b |>cu
                BandedMatrixSubsystemSolver(solution_b,deepcopy(A_withalpha), bandnumbers,b,n,k,q )
            else
                e=zeros(n)
                e[end-i+1]=1
                e=e |>cu
                BandedMatrixSubsystemSolver(view(solutions,:,i),deepcopy(A_withalpha), bandnumbers,e,n,k,q )
            end
        end
    end
    solutions[n+1:end,:]-=I
    solutions[n+1:end,:].*=-1
    alpha=solutions[n+1:end,:]\solution_b[n+1:end]
    solutions[1:n,:].*=alpha'
    solution_b[1:n]+=sum(solutions[1:n,:],dims=2)
    return Array(solution_b[1:n]), Array(solutions[n+1:end,:]), solution_b[n+1:end]
end


function BandedMatrixSubsystemSolver(X,A::CuArray,BandNumbers::CuArray,b::CuArray,n::Int, k::Int, q::Int)
    for row in 1:n
        @cuda threads=k BandedRowMultiplier!( view(A,row,:) , BandNumbers,X,row,q,n)
        @inbounds view(X,q+row).-=sum(view(A,row,1:k-1))
    end
end

function BandedRowMultiplier!( a_band, band, X,row::Int, q::Int, n::Int)
    index = threadIdx().x
    stride = blockDim().x
    for i = index:stride:(length(band)-1)
        if ! ((band[i]+row)<1 || (band[i]+row)>(n+q)) 
        @inbounds a_band[i] =  a_band[i]*X[band[i]+row];
        end
    end
    return nothing
end


################" TESTING #################################

N=200;
A_matrix=Float64.(diagm(0 => ones(N), 1=>ones(N-1),5=>ones(N-5)));
A=CreateMyBandedMatrix(A_matrix,[0,1,5])
sol=ones(N)*5
b=A_matrix*sol

out, alpha, ab= BandedSolver(A, b)
errorvalue=norm(sol.-out)./norm(sol)

function Helmholtz_matrix(DOF, k)
    n=DOF-1
    h=1/DOF
    helmholtz= M_2D(laplace_FEM(n,h)) + k.^2
    return(helmholtz)
end

function laplace_FEM(n,h)
    diagonal=ones(n)./h^2
    offdiag=diagonal[1:end-1]
    derivative=( spdiagm(0 => diagonal.*-2, 1=>offdiag, -1=>offdiag))
    return derivative
end

function M_2D(M)
    (n,n2)=size(M)
    identity_n=Matrix(1I, n ,n)
    M_2D= kron(M, identity_n) + kron( identity_n, M)
    return M_2D
end

function Gaussian_impulse_2D(xc, yc, sigma , DOF)
    n=DOF-1
    h=1/DOF
    coordinates=h:h:(1-h)
    impulse_2D= exp.(-1/(2*sigma^2)*(((yc.-coordinates).^2).+((xc.-coordinates)').^2))./(2pi*sigma^2)
    return (impulse_2D)
end


condnumbers=[]
errors=[]
N_values=6:2:16
for N=6:2:16
    M=Helmholtz_matrix(N, 1 .*I);
    v=Gaussian_impulse_2D(0.5,0.5, 0.01 ,N )    
    bands=[-N+1,-1,0,1,N-1];
    a=CreateMyBandedMatrix(M,bands );
    sol=reshape(M\vec(v), N-1,N-1);
    mysol, kcond =BandedSolver(a, vec(v))
    mysol=reshape(mysol, N-1,N-1)
    push!(errors,norm((sol).-mysol)./norm(sol))
    push!(condnumbers,kcond)
end

plot(size=(400,350))
plot!( xlabel="Matrix size N", ylabel="Relative error", legend = :none)
plot!(N_values, errors, yaxis=:log10)
savefig("errors.png")
plot(size=(400,350))
plot!( xlabel="Matrix size N", ylabel="Condition number of subsystem", legend = :none)
plot!(N_values, condnumbers, yaxis=:log10)
savefig("conditionnum.png")

timings_ref=[]
timings_mine=[]
N_values=[10,30,100]
for N in N_values
    M=Helmholtz_matrix(N, 1 .*I);
    v=vec(Gaussian_impulse_2D(0.5,0.5, 0.01 ,N ))    
    bands=[-N+1,-1,0,1,N-1];
    a=CreateMyBandedMatrix(M,bands );
    t=@belapsed ($M\$v)
    push!(timings_ref,t)
    t=@belapsed BandedSolver($a,$v)
    push!(timings_mine,t)
end
plot(size=(400,350))
plot!( xlabel="Matrix size N", ylabel="Solution time (s)", legend = :bottomright)
plot!(N_values, timings_mine, xaxis=:log10, yaxis=:log10, label="Own implementation")
plot!(N_values, timings_ref, xaxis=:log10, yaxis=:log10, label="Reference implementation")
savefig("speed.png")

################" zero diagonals #################################

A_zerosuperdiag=[1 0 0 0 0; 0 1 1 0 0; 0 0 1 0 0; 0 0 0 1 1; 0 0 0 0 1];
sol=[1,1,1,1,1]
b=A_zerosuperdiag*sol

A_full=CreateMyBandedMatrix(A_zerosuperdiag,[0,1])

function decomposition(X::MyBandedMatrix)
    n=size(X.A_band,1)
    q=maximum(X.bandnumbers)
    superdiagonal=X.A_band[1:end-q,end]
    indices=findall( x->x==0, superdiagonal)
    U=reduce(hcat,unitvector.(n,indices))
    addones=zeros(n)
    addones[indices].=1
    NewX=MyBandedMatrix(X.A_band.+addones,X.bandnumbers)
    return U, NewX, indices.+q
end 


function unitvector(N,e)
    out=zeros(N)
    out[e]=1
    return out
end


U, A_adjusted, colindices =decomposition(A_full);
n=size(A_full.A_band,1)
AinvU=zeros(size(U))
Ainvb=zeros(size(b))
for adjustment in 1:size(U,2)
    AinvU[:,adjustment]=BandedSolver(A_adjusted,U[:,adjustment])
end

Ainvb=BandedSolver(A_adjusted,b)
LU_VAinvU=lu((AinvU[colindices,:]-I))
Ainvb - AinvU * (LU_VAinvU \ Ainvb*[colindices,:])