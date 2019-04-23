# amelioration a faire: changement de type de U
# generalisation du codes (ces codes sont restraintes au cas ou dim_control = 2)
include("./DDP.jl")

using .DDP
using ForwardDiff, DiffResults
using LinearAlgebra

const T = 1
const N = 100
const H = T/N

const D = 4

const X0 = [1, 1, 3/2 * pi, 0]
const P  = [0.1, 0.1, 0.01, 1]

const C_W = 0.01
const C_A = 0.0001

const WEIGHT = 10000

car = LQR(X0, P, 2, T, N)

"""
F(xu::Vector) -> x::Vector
Car dynamics function. xu is the states-controls variable.
h is the time interval. d is the car length.
"""
function F(xu::Vector) 
    x, y, theta, v, w, a = xu
    
    f = H*v
    b = f*cos(w) + D - sqrt(D^2 - f^2 * sin(w)^2)

    x = x + b*cos(theta)
    y = y + b*sin(theta)
    theta = theta + asin(sin(w) * f/D)
    v = v + H*a
    
    return [x, y, theta, v]
    
end

dF(x, u) = ForwardDiff.jacobian(F, vcat(x, u))

function F(x, u) 
    x, y, theta, v = x
    w, a = u
    
    f = H*v
    b = f*cos(w) + D - sqrt(D^2 - f^2 * sin(w)^2)

    x = x + b*cos(theta)
    y = y + b*sin(theta)
    theta = theta + asin(sin(w) * f/D)
    v = v + H*a
    
    return [x, y, theta, v]
    
end

z(x, p) = sqrt(x^2 + p^2) - p

function L(x_u)
    x, y, theta, v, w, a = x_u
    p_x, p_y, p_theta, p_v = P
    0.01(z(x, p_x) + z(y, p_y)) + C_W * w^2 + C_A * a^2
end

Lf(x) =
    WEIGHT * (z(x[1], P[1]) + z(x[2], P[2]) + z(x[3], P[3]) + z(x[4], P[4]))

function j(U; N = 0)# loss of (x_N, u_N)
    x = X0
    for i in 1:N
        x = F(vcat(x, U[2i-1:2i]))
    end 
    L(vcat(x, U[2N+1:2N+2]))
end

function jf(U; N = 99)# final loss of x_N
    x = X0
    for i in 1:N
        x = F(vcat(x, U[2i-1:2i]))
    end
    Lf(x)
end

"""
Total loss related to U.
"""
_J(U; N = N-1) = sum(j(U, N = i) for i in 0:N-1) + jf(U; N = N) 
J(U) = _J(vcat(U...))

# dF(x, u) = ForwardDiff.jacobian(F, vcat(x, u))

# function dJ(U) 
#     U = ForwardDiff.gradient(_J, vcat(U...))
#     return [U[2i-1:2i] for i in 1:length(U)÷2]
# end

function diffs_L(x, u)
    xu = convert(Array{Float64,1}, vcat(x, u))
    result = DiffResults.HessianResult(xu)
    result = ForwardDiff.hessian!(result, L, xu)
    return DiffResults.gradient(result), DiffResults.hessian(result)
end

function diffs_Lf(x)
    x = convert(Array{Float64,1}, x)
    result = DiffResults.HessianResult(x)
    result = ForwardDiff.hessian!(result, Lf, x)
    return DiffResults.gradient(result), DiffResults.hessian(result)
end

function line_search_J(U, delta_U; a = 0.25, b = 0.5)
    _U = vcat(U...)
    result = DiffResults.GradientResult(_U)
    result = ForwardDiff.gradient!(result, _J, _U)
    J_U = DiffResults.value(result) #value
    _gradient = DiffResults.gradient(result)
    gradient = [_gradient[2i-1:2i] for i in 1:length(_gradient)÷2]
    #redundant
    
    step = 1
    while J(U + step * delta_U) > J_U + a*step* gradient ⋅ delta_U
        step = b * step
    end
    return step
end  

function go!(car)
    
    gradient_Lf, hessian_Lf = diffs_Lf(car.path.X[N])
    DVs = [gradient_Lf']
    D2Vs = [hessian_Lf]
    
    DQs = Array{Adjoint{Float64,Array{Float64,1}},1}(undef, 0)
    D2Qs = Array{Array{Float64,2}}(undef, 0)
    
    #backward pass
    for t in N-1:-1:1
        x, u = car.path.X[t], car.path.U[t]
        gradient_L, hessian_L = diffs_L(x, u)
        jacobian_F = dF(x, u)
        DQ = gradient_L' + DVs[1] * jacobian_F 
        D2Q = reshape([hessian_L[i, j] + #can add stm
                jacobian_F[:,i]' * D2Vs[1] * jacobian_F[:,j] for i in 1:DIM for j in 1:DIM] , (DIM,DIM))
        pushfirst!(DQs, DQ)
        pushfirst!(D2Qs, D2Q)
                        
        inv_Q_uu = inv(D2Q[DIM_x+1:DIM, DIM_x+1:DIM])    
        Q_ux = D2Q[DIM_x+1:DIM, 1:DIM_x]  
        Q_xu = D2Q[1:DIM_x, DIM_x+1:DIM]
        Q_xx = D2Q[1:DIM_x, 1:DIM_x]
        Q_x = DQ[1:DIM_x]'
        Q_u = DQ[DIM_x+1:DIM]'            
        DV = Q_x + Q_u * inv_Q_uu * Q_ux
        D2V = Q_xx + Q_xu * inv_Q_uu * Q_ux  
                        
        pushfirst!(DVs, DV)
        pushfirst!(D2Vs, D2V) 
    end  
     
    #Forward pass  GO!  
    inv_Q_uu_1 = inv(D2Qs[1][DIM_x+1:DIM, DIM_x+1:DIM])
    Q_u_1 = DQs[1][DIM_x+1:DIM]
    delta_U = [- inv_Q_uu_1 * Q_u_1]
    U, X = car.path.U, car.path.X
    u, x, delta_u = U[1], X[1], delta_U[1]                 
    for t in 2:N-1
        u = u + delta_u
        x =                 
                    

# function loop(obj::DDP)#,...)


# function F(xu::Vector, h = 0.01, d = 4) 
#     x, y, theta, v, w, a = xu
# #     w, a = u
    
#     f = h*v
#     b = f*cos(w) + d - sqrt(d^2 - f^2 * sin(w)^2)

#     x = x + b*cos(theta)
#     y = y + b*sin(theta)
#     theta = theta + asin(sin(w) * f/d)
#     v = v + h*a
    
#     return [x, y, theta, v]
    
# end

# z(x, p) = sqrt(x^2 + p^2) - p

# function L(x_u, p = [0.1, 0.1, 0.01, 1] , c_w = 0.01, c_a = 0.0001)
#     x, y, theta, v, w, a = x_u
#     p_x, p_y, p_theta, p_v = p
#     0.01(z(x, p_x) + z(y, p_y)) + c_w * w^2 + c_a * a^2
# end

# Lf(x, p = [0.1, 0.1, 0.01, 1], weight = 100000) =
#     weight * (z(x[1], p[1]) + z(x[2], p[2]) + z(x[3], p[3]) + z(x[4], p[4]))

# function j(U, x0 = [1, 1, 3/2 * pi, 0]; N = 0)# loss of (x_N, u_N)
#     x = x0
#     for i in 1:N
#         x = F(vcat(x, U[2i-1:2i]))
#     end 
#     L(vcat(x, U[2N+1:2N+2]))
# end

# function jf(U, x0 = [1, 1, 3/2 * pi, 0]; N = 100)# final loss of x_N
#     x = x0
#     for i in 1:N
#         x = F(vcat(x, U[2i-1:2i]))
#     end
#     Lf(x)
# end

# """
# Total loss related to U.
# """
# J(U; N = 100) = sum(j(U, N = i) for i in 0:N-1) + jf(U; N = N) 


# dF(x, u) = ForwardDiff.jacobian(F, vcat(x, u))

# function dJ(U) 
#     U = ForwardDiff.gradient(J, vcat(U...))
#     return [U[2i-1:2i] for i in 1:length(U)÷2]
# end

# function diffs_L(x, u)
#     xu = convert(Array{Float64,1}, vcat(x, u))
#     result = DiffResults.HessianResult(xu)
#     result = ForwardDiff.hessian!(result, L, xu)
#     return DiffResults.gradient(result), DiffResults.hessian(result)
# end

# function diffs_Lf(x)
#     x = convert(Array{Float64,1}, x)
#     result = DiffResults.HessianResult(x)
#     result = ForwardDiff.hessian!(result, Lf, x)
#     return DiffResults.gradient(result), DiffResults.hessian(result)
# end

[3] F(::Array{Float64,1}) at .\In[23]:19
[4] j(::Array{Float64,1}, ::Int64) at .\In[9]:4
[5] iterate at .\none:0 [inlined]
[6] mapfoldl_impl(::typeof(identity), ::typeof(Base.add_sum), ::NamedTuple{(:init,),Tuple{Float64}}, ::Base.Generator{UnitRange{Int64},getfield(Main, Symbol("##3#4")){Array{Float64,1}}}, ::Int64) at .\reduce.jl:45
[7] J(::Array{Float64,1}) at .\reduce.jl:59
[8] J at .\In[12]:1 [inlined]
[9] #line_search_J#5(::Float64, ::Float64, ::Int64, ::Function, ::Array{Array{Float64,1},1}, ::Array{Array{Float64,1},1}) at .\In[15]:10
[10] line_search_J at .\In[15]:2 [inlined]
[11] go!(::Main.DDP.Trajectory) at .\In[17]:52
[12] top-level scope at .\In[24]:2