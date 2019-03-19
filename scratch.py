import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial

import math 
from ad import adnumber, jacobian
from ad.admath import * 

from scipy.signal import unit_impulse
basis = [unit_impulse(6,i) for i in range(6)]

#Problem settings
T = 10; N = 1000; h = T/N; 
X = np.array(N * [[1, 1, 3/2*pi, 0]])
U = np.zeros((N,2))

d = 4

#position parameters
p_x = 0.1
p_y = 0.1
p_delta = 0.01
p_v = 1

p = np.array([p_x, p_y, p_delta, p_v])

#control loss parameters:
c_w = 0.01
c_a = 0.0001

#Euler schema function:
def F(x, y, delta, v, w, a) -> "R^4*":
    f = h*v
    b = f*math.cos(w) + d - math.sqrt(d**2 - f**2 * math.sin(w)**2)

    x = x + b*math.cos(delta)
    y = y + b*math.sin(delta)
    delta = delta + math.asin(math.sin(w) * f/d)
    v = v + h*a
    
    return np.array([x, y, delta, v])

#Function to compute differentials of F - Euler schema function:
def DF(x, y, delta, v, w, a) -> "R^4*6, R^4*6*6":
    """Compute the jacobian matrix and hessian tensor of Euler step."""
    x, y, delta, v, w, a = adnumber([x, y, delta, v, w, a])
    
    f = h*v
    b = f*cos(w) + d - sqrt(d**2 - f**2 * sin(w)**2)

    F_x = x + b*cos(delta)
    F_y = y + b*sin(delta)
    F_delta = delta + asin(sin(w) * f/d)
    F_v = v + h*a
    
    jaco = np.array(jacobian([F_x, F_y, F_delta, F_v], [x, y, delta, v, w, a]))
    
    H_x = F_x.hessian([x, y, delta, v, w, a])
    H_y = F_y.hessian([x, y, delta, v, w, a])
    H_delta = F_delta.hessian([x, y, delta, v, w, a])
    H_v = F_v.hessian([x, y, delta, v, w, a])
    
    hess = np.array([H_x, H_y, H_delta, H_v])
    
    return jaco, hess

#Loss function:
def L(x, y, delta, v, w, a) -> "R":
    """Compute the loss function."""
    
    z_x = math.sqrt(x**2 + p_x**2) - p_x
    z_y = math.sqrt(y**2 + p_y**2) - p_y
    
    return 0.01*(z_x + z_y) + c_w * w**2 + c_a * a**2
    

#Function to compute differentials of state-control loss:
def DL(x, y, delta, v, w, a) -> "R^6, R^6*6":
    """Compute the gradient vector and hessian matrix of loss function."""
    x, y, delta, v, w, a = adnumber([x, y, delta, v, w, a])
    
    z_x = sqrt(x**2 + p_x**2) - p_x
    z_y = sqrt(y**2 + p_y**2) - p_y
    
    L = 0.01*(z_x + z_y) + c_w * w**2 + c_a * a**2
    
    return np.array(L.gradient([x, y, delta, v, w, a])), np.array(L.hessian([x, y, delta, v, w, a]))


#Final loss function
def Lf(x, y, delta, v) -> "R":
    """Compute the final loss."""

    z_x = math.sqrt(x**2 + p_x**2) - p_x
    z_y = math.sqrt(y**2 + p_y**2) - p_y
    z_delta = math.sqrt(delta**2 + p_delta**2) - p_delta
    z_v = math.sqrt(v**2 + p_v**2) - p_v

    return 1000*( z_x + z_y + z_delta + z_v)


#Functions to compute differentials of final loss:
def D_Lf(x, y, delta, v) -> "R^4, R^4*4":
    """Compute the gradient vector and hessian matrix of final loss."""
    x, y, delta, v= adnumber([x, y, delta, v])

    z_x = sqrt(x**2 + p_x**2) - p_x
    z_y = sqrt(y**2 + p_y**2) - p_y
    z_delta = sqrt(delta**2 + p_delta**2) - p_delta
    z_v = sqrt(v**2 + p_v**2) - p_v

    L_F = 1000*( z_x + z_y + z_delta + z_v)
    
    return np.array(L_F.gradient([x, y, delta, v])), np.array(L_F.hessian([x, y, delta, v]))

#------------------------------
def V_star(i, ui):
    X_res = [F(*X[i], *ui)]
    for j in range(i+1, N-1):
        X_res.append(F(*X_res[-1], *U[j]))
    return L(*X[i], *ui) + sum([L(*X_res[j-(i+1)], *U[j]) for j in range(i+1, N-1)]) + Lf(*X_res[-1])

def line_search(f, gradient, x, direction, a = 0.25, b = 0.5) -> "step":
    """Backtracking line search."""
    t = 1
    while f(x + t*direction) > f(x) + a*t*gradient@direction or f(x + t*direction) == -np.inf:
#         print(x)
        t = b*t
#     print(f(x + t*direction))
    return t
#------------------------------




def update_line(num, data, line):
    line.set_data(data[num])
    return line,

if __name__ == "__main__":
    # gradient_Lf, hessian_Lf = D_Lf(*X[-1])

    # DVstar_list_inv = [gradient_Lf] #DV*_{n-1}(x_{n-1})
    # D2Vstar_list_inv = [hessian_Lf]

    # DV_list_inv = []
    # D2V_list_inv = []

    # #backward pass, begin with DV_n-2
    # for t in range(N-2, -1, -1): #from N-2 to 0
        
    #     gradient_L, hessian_L = DL(*X[t], *U[t])
    #     jacobian_F, hessian_F = DF(*X[t], *U[t])
    #     DV = gradient_L + DVstar_list_inv[-1] @ jacobian_F
    #     D2V = np.reshape([ei @ hessian_L @ ej + 
    # #                       DVstar_list_inv[-1] @ (ej @ hessian_F @ ei) + 
    #                     (jacobian_F @ ej) @ D2Vstar_list_inv[-1] @ (jacobian_F @ ei) for ei in basis for ej in basis], (6,6))

    #     DV_list_inv.append(DV)
    #     D2V_list_inv.append(D2V)
        
    #     DVstar = DV[:4] + DV[4:] @ np.linalg.inv(D2V[4:, 4:]) @ D2V[4:, :4]
    #     D2Vstar = D2V[:4, :4] + D2V[:4, 4:] @ np.linalg.inv(D2V[4:, 4:]) @ D2V[4:, :4]
    
    #     DVstar_list_inv.append(DVstar)
    #     D2Vstar_list_inv.append(D2Vstar)

    # #Forward pass
    # DV = DV_list_inv[::-1]
    # D2V = D2V_list_inv[::-1]

    # X_hat = np.copy(X)
    # #forward pass
    # for t in range(N-1):
    #     if t == 0:
    #         h_u = -np.linalg.inv(D2V[t][4:, 4:]) @ DV[t][4:]
    #         U[t] = U[t] + h_u
    #         X_hat[t+1] = F(*X_hat[t], *U[t])
    #     else:
    #         h_x = X_hat[t] - X[t]
    #         h_u = -np.linalg.inv(D2V[t][4:, 4:]) @ (DV[t][4:] + D2V[t][4:, :4] @ h_x)
    #         U[t] = U[t] + h_u
    #         X_hat[t+1] = F(*X_hat[t], *U[t])   
    data = []
    data.append(X[:, :2])

    n = 500
    for _ in range(n):
        if _ != 0:
            X = np.copy(X_hat)

        data.append(X[:, :2])

        gradient_Lf, hessian_Lf = D_Lf(*X[-1])

        DVstar_list_inv = [gradient_Lf] #DV*_{n-1}(x_{n-1})
        D2Vstar_list_inv = [hessian_Lf]

        DV_list_inv = []
        D2V_list_inv = []

        #backward pass, begin with DV_n-2
        for t in range(N-2, -1, -1): #from N-2 to 0
            
            gradient_L, hessian_L = DL(*X[t], *U[t])
            jacobian_F, hessian_F = DF(*X[t], *U[t])
            DV = gradient_L + DVstar_list_inv[-1] @ jacobian_F
            D2V = np.reshape([ei @ hessian_L @ ej + 
        #                       DVstar_list_inv[-1] @ (ej @ hessian_F @ ei) + 
                            (jacobian_F @ ej) @ D2Vstar_list_inv[-1] @ (jacobian_F @ ei) for ei in basis for ej in basis], (6,6))

            DV_list_inv.append(DV)
            D2V_list_inv.append(D2V)
            
            DVstar = DV[:4] + DV[4:] @ np.linalg.inv(D2V[4:, 4:]) @ D2V[4:, :4]
            D2Vstar = D2V[:4, :4] + D2V[:4, 4:] @ np.linalg.inv(D2V[4:, 4:]) @ D2V[4:, :4]
        
            DVstar_list_inv.append(DVstar)
            D2Vstar_list_inv.append(D2Vstar)

            DV = DV_list_inv[::-1]
            D2V = D2V_list_inv[::-1]

        X_hat = np.zeros(X.shape)
        #forward pass
        for t in range(N-1):
            if t == 0:
                h_u = -np.linalg.inv(D2V[t][4:, 4:]) @ DV[t][4:]
                step = line_search(partial(V_star,t), DV[t][4:], U[t], h_u)
                #print(step)
                U[t] = U[t] + step * h_u
                X_hat[t+1] = F(*X_hat[t], *U[t])
            else:
                h_x = X_hat[t] - X[t]
                h_u = -np.linalg.inv(D2V[t][4:, 4:]) @ (DV[t][4:] + D2V[t][4:, :4] @ h_x)
                step = line_search(partial(V_star,t), DV[t][4:], U[t], h_u)
                #print(step)
                U[t] = U[t] + step * h_u
                X_hat[t+1] = F(*X_hat[t], *U[t])
        print(1)
        plt.plot(X_hat[:, 0], X_hat[:, 1])
        plt.plot(p[0], p[1], marker = 'x')

    
# plt.show()

    fig1 = plt.figure()

    # Fixing random state for reproducibility

    l, = plt.plot([], [], 'r-')
    plt.xlim(0, 80)
    plt.ylim(0, 80)
    plt.xlabel('x')
    plt.title('test')
    line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
                                    interval=50, blit=True)
    plt.show()