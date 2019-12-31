# best solution:  Cardano's Formula
#  https://brilliant.org/wiki/cardano-method/

'''
 in WolframeAlpha

 solve { x^3-(a+b+c)*x^2+(a*b+a*c+b*c-2*f)*x+f*(a+c)-a*b*c==0  && x>a && x<c && 0.1>f>0 && a>0 && b>0  && c>0 }

 output:
 0<f<1/10 and
  x = -((1 - i sqrt(3)) (2 a^3 - 3 a^2 b - 3 a^2 c + sqrt(4 (-a^2 + a b + a c - b^2 + b c - c^2 - 6 f)^3 + (2 a^3 - 3 a^2 b - 3 a^2 c - 3 a b^2 + 12 a b c - 3 a c^2 - 9 a f + 2 b^3 - 3 b^2 c - 3 b c^2 + 18 b f + 2 c^3 - 9 c f)^2) - 3 a b^2 + 12 a b c - 3 a c^2 - 9 a f + 2 b^3 - 3 b^2 c - 3 b c^2 + 18 b f + 2 c^3 - 9 c f)^(1/3))/(6 2^(1/3)) + ((1 + i sqrt(3)) (-a^2 + a b + a c - b^2 + b c - c^2 - 6 f))/(3 2^(2/3) (2 a^3 - 3 a^2 b - 3 a^2 c + sqrt(4 (-a^2 + a b + a c - b^2 + b c - c^2 - 6 f)^3 + (2 a^3 - 3 a^2 b - 3 a^2 c - 3 a b^2 + 12 a b c - 3 a c^2 - 9 a f + 2 b^3 - 3 b^2 c - 3 b c^2 + 18 b f + 2 c^3 - 9 c f)^2) - 3 a b^2 + 12 a b c - 3 a c^2 - 9 a f + 2 b^3 - 3 b^2 c - 3 b c^2 + 18 b f + 2 c^3 - 9 c f)^(1/3)) + 1/3 (a + b + c)
  and c>0 and b>0 and 0<a<c

  In form understood by WolframeAlpha

  x = -((1-I*sqrt(3))*(2*a^3-3*a^2*b - 3*a^2*c + sqrt(4*(-a^2 + a*b + a*c - b^2 + b*c - c^2 - 6*f)^3 + (2*a^3 - 3*a^2*b - 3*a^2*c - 3*a*b^2 + 12*a*b*c - 3*a*c^2 - 9*a*f + 2*b^3 - 3*b^2*c - 3*b*c^2 + 18*b*f + 2*c^3 - 9*c*f)^2) - 3*a*b^2 + 12*a*b*c - 3*a*c^2 - 9*a*f + 2*b^3 - 3*b^2*c - 3*b*c^2 + 18*b*f + 2*c^3 - 9*c*f)^(1/3))/(6*2^(1/3))  + ((1 + I*sqrt(3))*(-a^2 + a*b + a*c - b^2 + b*c - c^2 - 6*f))/(3*2^(2/3)* (2*a^3 - 3*a^2*b - 3*a^2*c + sqrt(4*(-a^2 + a*b + a*c - b^2 + b*c - c^2 - 6*f)^3  + (2*a^3 - 3*a^2*b - 3*a^2*c - 3*a*b^2 + 12*a*b*c - 3*a*c^2 - 9*a*f + 2*b^3 - 3*b^2*c - 3*b*c^2 + 18*b*f + 2*c^3 - 9*c*f)^2) - 3*a*b^2 + 12*a*b*c  - 3*a*c^2 - 9*a*f + 2*b^3 - 3*b^2*c - 3*b*c^2 + 18*b*f + 2*c^3 - 9*c*f)^(1/3)) + 1/3*(a + b + c)



 or
 an example:
 solve {x^3-(60+100+110)*x^2 +(6000+6600+11000-2*0.01)*x+0.01*(170)-60*100*110=0 and x>60 and x<110}
       get x = 100 plus a minimal imaginary part.





'''



def cubicFunc(x, a, b, c, d):
    return a*(x**3)+b*(x**2)+c*x+d

def solveCubicEquation(a,b,c,d):
    # ref: https://brilliant.org/wiki/cardano-method/
    Q = (3*a*c-b*b)/(9*a*a)
    R = (9*a*b*c-27*a*a*d-2*b*b*b)/(54*a*a*a)
    S = (R+(Q*Q*Q+R*R)**(1/2))**(1/3)
    T = (R-(Q*Q*Q+R*R)**(1/2))**(1/3)
    x1 = S+T-b/(3*a)
    x2 = -(S+T)/2.0-b/(3*a)+ 1j*(S-T)*(3**(1/2))/2.0
    x3 = -(S+T)/2.0-b/(3*a)- 1j*(S-T)*(3**(1/2))/2.0
    return x1, x2, x3

def main():

    print("Verify a cubic equation having 3 real root solution: 1,2,3" )
    a = 1
    b = -6.0
    c = 11.0
    d = -6
    x1,x2,x3 = solveCubicEquation(a,b,c,d)
    print(f"roots: x1={x1}, x2={x2}, x3={x3}")
    print("Verify:")
    print(f" x1={x1}, f(x1) = {cubicFunc(x1,a,b,c,d)}")
    print(f" x2={x2}, f(x2) = {cubicFunc(x2, a, b, c, d)}")
    print(f" x3={x3}, f(x3) = {cubicFunc(x3, a, b, c, d)}")


    print("\n")
    print("Verify a cubic equation whose solution should be 60-110 range, whith expection 100")
    a = 1.0
    b = -270.0
    c = 23600.0 - 0.00002
    d = -660000.0 + 0.00001 * 170
    x1, x2, x3 = solveCubicEquation(a, b, c, d)
    print(f"roots: x1={x1}, x2={x2}, x3={x3}")
    print("Verify:")
    print(f" x1={x1}, f(x1) = {cubicFunc(x1, a, b, c, d)}")
    print(f" x2={x2}, f(x2) = {cubicFunc(x2, a, b, c, d)}")
    print(f" x3={x3}, f(x3) = {cubicFunc(x3, a, b, c, d)}")


    print ("======= The end ==========")

if __name__ == "__main__":

    main()


