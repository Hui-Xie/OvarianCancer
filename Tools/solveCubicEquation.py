# ax^3+ bx^2+cx+d =0, where a !=0
#formula from: https://sciencing.com/how-to-factor-polynomials-of-degree-3-12751796.html
# but this formula only gives one solution which may has 3 solution.

a=1
b=-6.0
c=11.0
d=-6
# this cubic equation has 3 real root solution: 1,2,3

def f(x):
    return a*x**3+b*x**2+c*x+d


p =-b/(3*a)
q =p**3+(b*c-3*a*d)/(6*a*a)
r =c/(3*a)

x = (q+(q*q+(r-p*p)**3)**(1/2))**(1/3) + (q-(q*q+(r-p*p)**3)**(1/2))**(1/3) +p


print(f"x= {x}")
print(f"f(x)= {f(x)}")




