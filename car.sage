T = 10; N = 1000; h = T/N; 
d = 4


f(v) = h*v
b(v, w) = f(v) * cos(w) + d - sqrt(d**2 - f(v)**2 * sin(w)**2)
F(x, y, theta, v, w, a) = (x + b(v,w) * cos(theta), y + b(v,w) * sin(theta), theta + arcsin(sin(w) * f(v)/d), v + h*a)
dF = diff(F)

z(x, p) = sqrt(x**2 + p**2) -p
L(x, y, theta, v, w, a) = 0.01*(z(x, p_x) + z(y, p_y)) + c_w * w**2 + c_a * a**2
Lf(x, y, theta, v) = 100000* (z(x, p_x) + z(y, p_y) + z(theta, p_theta) + z(v, p_v))

dL = diff(L)
dLf = diff(Lf)