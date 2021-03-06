
# This file was *autogenerated* from the file car - Copy.sage
from sage.all_cmdline import *   # import sage library

_sage_const_2 = Integer(2); _sage_const_10 = Integer(10); _sage_const_1000 = Integer(1000); _sage_const_4 = Integer(4)
T = _sage_const_10 ; N = _sage_const_1000 ; h = T/N; 
d = _sage_const_4 


__tmp__=var("v"); f = symbolic_expression(h*v).function(v)
__tmp__=var("v,w"); b = symbolic_expression(f(v) * cos(w) + d - sqrt(d**_sage_const_2  - f(v)**_sage_const_2  * sin(w)**_sage_const_2 )).function(v,w)
__tmp__=var("x,y,theta,v,w,a"); F = symbolic_expression((x + b(v,w) * cos(theta), * BackslashOperator() * ).function(x,y,theta,v,w,a)
    y + b(v,w) * sin(theta), theta + arcsin(sin(w) * f(v)/d), v + h*a)

