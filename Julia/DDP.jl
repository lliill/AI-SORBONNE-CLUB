module DDP
using LinearAlgebra, ForwardDiff, DiffResults
export LQR, interface

mutable struct Trajectory
    X :: Array{Array{Float64,1},1}
    U :: Array{Array{Float64,1},1}
end

"""
LQR(X1, p, DIM_u, T, N) 
Construct an object of DDP problem settings. 
T is the time. N is the discretization of time. 
X1 is the original position. p is the desired final postion after time T.
"""
struct LQR
    X1 :: Array{Float64, 1}
    p :: Array{Float64, 1}
    DIM_x :: Int64
    DIM_u :: Int64
    DIM :: Int64
    T :: Int64
    N :: Int64
    path :: Trajectory
    
    function LQR(X1, p, DIM_u, T, N) 
        if size(X1) == size(p)
            DIM_x = size(X1)[1]
            DIM = DIM_x + DIM_u
            X = [X1 for _ in 1:N]
            U = [zeros(DIM_u) for _ in 1:N-1]
            path = Trajectory(X, U)
            return new(X1, p, DIM_x, DIM_u, DIM, T, N, path)
        else
            error("X1 and p must have the same size") 
        end
        
    end
    
end

"""
Need to provide a Euler explicit schema function F:
L, Lf

"""
function interface(obj::LQR, F, L, Lf)

    T = obj.T
    N = obj.N
    DIM_x = obj.DIM_x
    DIM_u = obj.DIM_u
    DIM = obj.DIM
    X1 = obj.X1



    dF(x, u) = ForwardDiff.jacobian(F, vcat(x, u))

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

    function j(U, k)# loss of (x_k, u_k)
        x = X1
        for i in 1:k-1
            x = F(vcat(x, U[2i-1:2i]))
        end 
        L(vcat(x, U[2k-1:2k]))
    end

    function jf(U)# final loss of x_N
        x = X1
        for i in 1:N-1
            x = F(vcat(x, U[2i-1:2i]))
        end
        Lf(x)
    end

    """
    Total loss related to U. Need to provide L, Lf in the environnement:
    L(x_u) -> R, Lf(x) -> R.
    """
    J(U) = sum(j(U, i) for i in 1:N-2) + jf(U) 
    J(U::Array{Array{Float64,1},1}) = J(vcat(U...))

    """
    line_search_J(U, delta_U; a = 0.25, b = 0.5, jumping = 1) -> step
    """
    function line_search_J(U, delta_U; a = 0.25, b = 0.5, jumping = 1)
        _U = vcat(U...)
        result = DiffResults.GradientResult(_U)
        result = ForwardDiff.gradient!(result, J, _U)
        J_U = DiffResults.value(result) #value
        _gradient = DiffResults.gradient(result)
        gradient = [_gradient[2i-1:2i] for i in 1:length(_gradient)÷2]
    
        step = jumping
        while J(U + step * delta_U) > J_U + a*step* gradient ⋅ delta_U
            step = b * step
        end
        return step
    end     

    function go!(path)
        
        gradient_Lf, hessian_Lf = diffs_Lf(path.X[N])
        DVs = [gradient_Lf']
        D2Vs = [hessian_Lf]
        
        DQs = Array{Adjoint{Float64,Array{Float64,1}},1}(undef, 0)
        D2Qs = Array{Array{Float64,2}}(undef, 0)
        
        #backward pass
        for t in N-1:-1:1
            x, u = path.X[t], path.U[t]
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
        U, X = path.U, path.X
        u, x, delta_u = U[1], X[1], delta_U[1] 
            #construct delta_U
        for t in 2:N-1
            u = u + delta_u
            x_hat = F(x,u)               
            delta_x = x_hat - x
                
            inv_Q_uu = inv(D2Qs[t][DIM_x+1:DIM, DIM_x+1:DIM])
            Q_u = DQs[t][DIM_x+1:DIM]
            Q_ux = D2Qs[t][DIM_x+1:DIM, 1:DIM_x]                
            delta_u = - inv_Q_uu * (Q_u + Q_ux * delta_x)
            push!(delta_U, delta_u)
        end    
        step = line_search_J(U, delta_U)
        U = U + step * delta_U
        println(step)
        X = F(U)                
        println(X[N])
        path.U = U
        path.X = X
                        
    end

    return J, go!

end

end#module end


# """
# Car dynamics function. xu is the states-controls variable.
# h is the time interval. d is the car length.
# """
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

# function j(U, X1 = [1, 1, 3/2 * pi, 0]; N = 0)# loss of (x_N, u_N)
#     x = X1
#     for i in 1:N
#         x = F(vcat(x, U[2i-1:2i]))
#     end 
#     L(vcat(x, U[2N+1:2N+2]))
# end

# function jf(U, X1 = [1, 1, 3/2 * pi, 0]; N = 100)# final loss of x_N
#     x = X1
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

# function loop(obj::DDP)#,...)

