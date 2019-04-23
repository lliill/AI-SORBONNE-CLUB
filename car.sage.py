
# This file was *autogenerated* from the file car.sage
from sage.all_cmdline import *   # import sage library

_sage_const_2 = Integer(2); _sage_const_4 = Integer(4); _sage_const_10 = Integer(10); _sage_const_1000 = Integer(1000); _sage_const_0p01 = RealNumber('0.01'); _sage_const_100000 = Integer(100000)
T = _sage_const_10 ; N = _sage_const_1000 ; h = T/N; 
d = _sage_const_4 


__tmp__=var("v"); f = symbolic_expression(h*v).function(v)
__tmp__=var("v,w"); b = symbolic_expression(f(v) * cos(w) + d - sqrt(d**_sage_const_2  - f(v)**_sage_const_2  * sin(w)**_sage_const_2 )).function(v,w)
__tmp__=var("x,y,theta,v,w,a"); F = symbolic_expression((x + b(v,w) * cos(theta), y + b(v,w) * sin(theta), theta + arcsin(sin(w) * f(v)/d), v + h*a)).function(x,y,theta,v,w,a)
dF = diff(F)

__tmp__=var("x,p"); z = symbolic_expression(sqrt(x**_sage_const_2  + p**_sage_const_2 ) -p).function(x,p)
__tmp__=var("x,y,theta,v,w,a"); L = symbolic_expression(_sage_const_0p01 *(z(x, p_x) + z(y, p_y)) + c_w * w**_sage_const_2  + c_a * a**_sage_const_2 ).function(x,y,theta,v,w,a)
__tmp__=var("x,y,theta,v"); Lf = symbolic_expression(_sage_const_100000 * (z(x, p_x) + z(y, p_y) + z(theta, p_theta) + z(v, p_v))).function(x,y,theta,v)

dL = diff(L)
dLf = diff(Lf)
