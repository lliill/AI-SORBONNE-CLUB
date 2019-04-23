T = 10; N = 1000; h = T/N; 
d = 4


f(v) = h*v
b(v, w) = f(v) * cos(w) + d - sqrt(d**2 - f(v)**2 * sin(w)**2)
F(x, y, theta, v, w, a) = (x + b(v,w) * cos(theta),\
    y + b(v,w) * sin(theta), theta + arcsin(sin(w) * f(v)/d), v + h*a)