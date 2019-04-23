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

const X1 = [1, 1, 3/2 * pi, 0]
const P  = [0.1, 0.1, 0.01, 1]

const DIM_x = 4
const DIM_u = 2
const DIM = DIM_x + DIM_u

const C_W = 0.01
const C_A = 0.0001

const WEIGHT = 10000

car = LQR(X1, P, DIM_u, T, N)

"""
F(xu::Vector) -> x::Vector
F(x, u) -> x::Vector
Car dynamics function. xu is the states-controls variable.
h is the time interval. d is the car length.
"""
function F(xu::Vector) 
    
    x, y, theta, v, w, a = xu
#     println("v = ", v)
    f = H*v
#     if (D^2 - f^2 * sin(w)^2) < 0
#         println("v = ", v, "f = ", f, ", f^2 = ", f^2, ", w = ", w, ", sin(w)^2 = ", sin(w)^2)
#     end
    b = f*cos(w) + D - sqrt(D^2 - f^2 * sin(w)^2)

    x = x + b*cos(theta)
    y = y + b*sin(theta)
    theta = theta + asin(sin(w) * f/D)
    v = v + H*a
    
    return [x, y, theta, v]
    
end

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

function F(U::Array{Array{Float64,1},1})
    x = X1
    X = [X1]   
    for i in 1:N-1
        x = F(x, U[i])
        push!(X, x)
    end
    return X
end



#---

z(x, p) = sqrt(x^2 + p^2) - p

function L(x_u)
    x, y, theta, v, w, a = x_u
    p_x, p_y, p_theta, p_v = P
    0.01(z(x, p_x) + z(y, p_y)) + C_W * w^2 + C_A * a^2
end

Lf(x) =
    WEIGHT * (z(x[1], P[1]) + z(x[2], P[2]) + z(x[3], P[3]) + z(x[4], P[4]))