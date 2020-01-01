# best solution:  Cardano's Formula
#  https://brilliant.org/wiki/cardano-method/

'''
 in WolframeAlpha

 solve { x^3-(a+b+c)*x^2+(a*b+a*c+b*c-2*f)*x+f*(a+c)-a*b*c==0  && x>a && x<c && 0.1>f>0 && a>0 && b>0  && c>0 } over the reals

 output:
 0<f<1/10 and
  x = -((1 - i sqrt(3)) (2 a^3 - 3 a^2 b - 3 a^2 c + sqrt(4 (-a^2 + a b + a c - b^2 + b c - c^2 - 6 f)^3 + (2 a^3 - 3 a^2 b - 3 a^2 c - 3 a b^2 + 12 a b c - 3 a c^2 - 9 a f + 2 b^3 - 3 b^2 c - 3 b c^2 + 18 b f + 2 c^3 - 9 c f)^2) - 3 a b^2 + 12 a b c - 3 a c^2 - 9 a f + 2 b^3 - 3 b^2 c - 3 b c^2 + 18 b f + 2 c^3 - 9 c f)^(1/3))/(6 2^(1/3)) + ((1 + i sqrt(3)) (-a^2 + a b + a c - b^2 + b c - c^2 - 6 f))/(3 2^(2/3) (2 a^3 - 3 a^2 b - 3 a^2 c + sqrt(4 (-a^2 + a b + a c - b^2 + b c - c^2 - 6 f)^3 + (2 a^3 - 3 a^2 b - 3 a^2 c - 3 a b^2 + 12 a b c - 3 a c^2 - 9 a f + 2 b^3 - 3 b^2 c - 3 b c^2 + 18 b f + 2 c^3 - 9 c f)^2) - 3 a b^2 + 12 a b c - 3 a c^2 - 9 a f + 2 b^3 - 3 b^2 c - 3 b c^2 + 18 b f + 2 c^3 - 9 c f)^(1/3)) + 1/3 (a + b + c)
  and c>0 and b>0 and 0<a<c

  In form understood by WolframeAlpha

  x = -((1-I*sqrt(3))*(2*a^3-3*a^2*b - 3*a^2*c + sqrt(4*(-a^2 + a*b + a*c - b^2 + b*c - c^2 - 6*f)^3 + (2*a^3 - 3*a^2*b - 3*a^2*c - 3*a*b^2 + 12*a*b*c - 3*a*c^2 - 9*a*f + 2*b^3 - 3*b^2*c - 3*b*c^2 + 18*b*f + 2*c^3 - 9*c*f)^2) - 3*a*b^2 + 12*a*b*c - 3*a*c^2 - 9*a*f + 2*b^3 - 3*b^2*c - 3*b*c^2 + 18*b*f + 2*c^3 - 9*c*f)^(1/3))/(6*2^(1/3))  + ((1 + I*sqrt(3))*(-a^2 + a*b + a*c - b^2 + b*c - c^2 - 6*f))/(3*2^(2/3)* (2*a^3 - 3*a^2*b - 3*a^2*c + sqrt(4*(-a^2 + a*b + a*c - b^2 + b*c - c^2 - 6*f)^3  + (2*a^3 - 3*a^2*b - 3*a^2*c - 3*a*b^2 + 12*a*b*c - 3*a*c^2 - 9*a*f + 2*b^3 - 3*b^2*c - 3*b*c^2 + 18*b*f + 2*c^3 - 9*c*f)^2) - 3*a*b^2 + 12*a*b*c  - 3*a*c^2 - 9*a*f + 2*b^3 - 3*b^2*c - 3*b*c^2 + 18*b*f + 2*c^3 - 9*c*f)^(1/3)) + 1/3*(a + b + c)

  Input:
  solve { x^3-(a+b+c)*x^2+(a*b+a*c+b*c-2*f)*x+f*(a+c)-a*b*c==0  && x>a && x<c && 0.1>f>0 && a>0 && b>0  && c>0 }; z=Re(x)


 or
 an example:
 solve {x^3-(60+100+110)*x^2 +(6000+6600+11000-2*0.01)*x+0.01*(170)-60*100*110=0 and x>60 and x<110}
       get x = 100 plus a minimal imaginary part.





'''



def cubicFunc(x, s1, s2, s3, f):
    a = 1.0
    b = -(s1 + s2 + s3)
    c = s1 * s2 + s1 * s3 + s2 * s3 - 2 * f
    d = f * (s1 + s3) - s1 * s2 * s3
    # where a, b,c are the coefficient of cubic equation.
    return a*(x**3)+b*(x**2)+c*x+d


def wolframeCubic(a, b, c, f):
    '''
    solve { x^3-(a+b+c)*x^2+(a*b+a*c+b*c-2*f)*x+f*(a+c)-a*b*c==0  && x>a && x<c && 0.1>f>0 && a>0 && b>0  && c>0 } over the reals

    :param a: x_{i-1}
    :param b: x_i
    :param c: x_{i+1}
    :param f: learningRate * barrierParameter(mu)
    :return:
    '''
    S = 2 * a ** 3 - 3 * a ** 2 * b - 3 * a ** 2 * c
    P = -a**2 + a*b + a*c - b**2 + b*c - c**2 - 6*f
    T = - 3*a*b**2 + 12*a*b*c - 3*a*c**2 - 9*a*f + 2*b**3 - 3*b**2*c - 3*b*c**2 + 18*b*f + 2*c**3 - 9*c*f
    Q = (S+(4*P**3+(S+T)**2)**(1/2)+T)**(1.0/3.0)

    x1 = 1.0 / (3 * 2 ** (1.0 / 3)) * Q - (2 ** (1.0 / 3) * P) / (3 * Q) + (a + b + c) / 3.0
    x2 = -Q * (1 - 1j * 3 ** (1 / 2)) / (6 * 2 ** (1.0 / 3)) + P * (1 + 1j * 3 ** (1 / 2)) / (
                Q * 3 * 2 ** (2.0 / 3)) + (a + b + c) / 3.0  # choice by wolframAlpha, as a<x<c
    x3 = -Q * (1 + 1j * 3 ** (1 / 2)) / (6 * 2 ** (1.0 / 3)) + P * (1 - 1j * 3 ** (1 / 2)) / (
                Q * 3 * 2 ** (2.0 / 3)) + (a + b + c) / 3.0
    # these are real and imaginary parts acccording explict i;
    x3Real = -Q / (6 * 2 ** (1.0 / 3)) + P / (Q * 3 * 2 ** (2.0 / 3)) + (a + b + c) / 3.0
    x3Imag = -Q * 3 ** (1 / 2) / (6 * 2 ** (1.0 / 3)) - P * 3 ** (1 / 2) / ( Q * 3 * 2 ** (2.0 / 3))

    # these are real and imaginary parts considering the Q hax implicit i;
    x3Real = x3Real + x3Imag*1j   # multiply 1j may be a problem.

    return x1,x2,x3, x3Real, x3Imag



    #return x




def cardanoCubic(s1, s2, s3, f):
    # ref: https://brilliant.org/wiki/cardano-method/
    #  solve { x^3-(a+b+c)*x^2+(a*b+a*c+b*c-2*f)*x+f*(a+c)-a*b*c==0  && x>a && x<c && 0.1>f>0 && a>0 && b>0  && c>0 } over the reals
    #  s1, s2, s3 are the k iteration surface location, f is learningRate*barrierParameter
    a = 1.0
    b = -(s1+s2+s3)
    c = s1*s2 + s1*s3+ s2*s3-2*f
    d = f*(s1+s3)-s1*s2*s3
    # where a, b,c are the coefficient of cubic equation.

    Q = (3*a*c-b*b)/(9*a*a)
    R = (9*a*b*c-27*a*a*d-2*b*b*b)/(54*a*a*a)
    S = (R+(Q*Q*Q+R*R)**(1/2))**(1/3)
    T = (R-(Q*Q*Q+R*R)**(1/2))**(1/3)
    x1 = S+T-b/(3*a)
    x2 = -(S+T)/2.0-b/(3*a)+ 1j*(S-T)*(3**(1/2))/2.0
    x3 = -(S+T)/2.0-b/(3*a)- 1j*(S-T)*(3**(1/2))/2.0
    return x1, x2, x3

def main():
    print("\n")
    print("Use WolframAlpha method")
    s1 =99.0   #surface1
    s2 =100
    s3 =101.0
    f = 0.01
    print(f"for a={s1}, b={s2}, c={s3}, f={f}")
    print("slove: solve { x^3-(a+b+c)*x^2+(a*b+a*c+b*c-2*f)*x+f*(a+c)-a*b*c==0  && x>a && x<c && 0.1>f>0 && a>0 && b>0  && c>0 } over the reals")
    x1, x2,x3, x3Real, x3Imag = wolframeCubic(s1,s2,s3,f)
    print (f"x1={x1}, x2={x2}, \n x3={x3},\n x3Real={x3Real}, x3Imag={x3Imag}")

    print("\n")
    print("Use Cardano Formula")
    x1, x2, x3 = cardanoCubic(s1, s2, s3, f)
    print(f"roots: x1={x1}, x2={x2}, x3={x3}")
    print("Verify:")
    print(f" x1={x1}, f(x1) = {cubicFunc(x1, s1,s2,s3,f)}")
    print(f" x2={x2}, f(x2) = {cubicFunc(x2, s1,s2,s3,f)}")
    print(f" x3={x3}, f(x3) = {cubicFunc(x3, s1,s2,s3,f)}")


    print ("======= The end ==========")

if __name__ == "__main__":
     main()


