#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: qpex1.py
# Version 12.10.0
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2009, 2019. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""
Entering and optimizing a quadratic programming problem.

To run from the command line, use

python qpex1.py

Maximize
 obj: x1 + 2 x2 + 3 x3 + [ - 33 x1 ^2 + 12 x1 * x2 - 22 x2 ^2 + 23 x2 * x3 - 11 x3 ^2 ] / 2
Subject To
 c1: - x1 + x2 + x3 <= 20
 c2: x1 - 3 x2 + x3 <= 30
Bounds
0 <= x1 <= 40
End

Explanation: 0.5 x^T Q x + c x

Q matrix: x^T Q x
-33   6   0
  6  -22  11.5
  0  11.5 -11
that results in:

         | -33   6   0    | x1
x1 x2 x3 |   6  -22  11.5 | x2 = - 33 x1 ^2 + 12 x1 * x2 - 22 x2 ^2 + 23 x2 * x3 - 11 x3 ^2
         |   0  11.5 -11  | x3



"""
import cplex


def setproblemdata_original(p):
    p.objective.set_sense(p.objective.sense.maximize)

    p.linear_constraints.add(rhs=[20.0, 30.0], senses="LL")

    obj = [1.0, 2.0, 3.0]
    ub = [40.0, cplex.infinity, cplex.infinity]

    # constrain matrix in column mode
    cols = [[[0, 1], [-1.0, 1.0 ]],
            [[0, 1], [1.0, -3.0]],
            [[0, 1], [1.0, 1.0]]]

    p.variables.add(obj=obj, ub=ub, columns=cols, names=["one", "two", "three"])

    qmat = [[[0, 1], [-33.0, 6.0 ]],
            [[0, 1, 2], [6.0, -22.0, 11.5]],
            [[1, 2], [11.5, -11.0]]]

    p.objective.set_quadratic(qmat)


# Hui modified version:
def setproblemdata(p):
    p.objective.set_sense(p.objective.sense.maximize)

    # object func: 0.5 x^T Q x + c x
    Q = [[[0, 1, 2], [-33.0, 6.0, 0]],
         [[0, 1, 2], [6.0, -22.0, 11.5]],
         [[0, 1, 2], [0, 11.5, -11.0]]]
    objCoef = [1.0, 2.0, 3.0]
    ub = [40.0, cplex.infinity, cplex.infinity]
    objName = ["x1", "x2", "x3"]
    p.variables.add(obj=objCoef, ub=ub, names=objName)
    p.objective.set_quadratic(Q)  # this line must be place after p.variable.add()

    # constraint matrix in row mode
    constraintMatrix = [[[0,1,2], [-1, 1, 1]],
                        [[0,1,2], [1 ,-3 ,1 ]]]
    rhs = [20, 30]
    p.linear_constraints.add(lin_expr=constraintMatrix, senses="LL",  rhs=rhs)


def qpex1():
    p = cplex.Cplex()
    setproblemdata(p)

    p.solve()

    # solution.get_status() returns an integer code
    print("Solution status = ", p.solution.get_status(), ":", end=' ')
    # the following line prints the corresponding string
    print(p.solution.status[p.solution.get_status()])
    print("Solution value  = ", p.solution.get_objective_value())

    numrows = p.linear_constraints.get_num()

    for i in range(numrows):
        print("Row ", i, ":  ", end=' ')
        print("Slack = %10f " % p.solution.get_linear_slacks(i), end=' ')
        print("Pi = %10f" % p.solution.get_dual_values(i))

    numcols = p.variables.get_num()

    for j in range(numcols):
        print("Column ", j, ":  ", end=' ')
        print("Value = %10f " % p.solution.get_values(j), end=' ')
        print("Reduced Cost = %10f" % p.solution.get_reduced_costs(j))


if __name__ == "__main__":
    qpex1()

'''
Original code execute result:
Connected to pydev debugger (build 202.6948.78)
Version identifier: 12.10.0.0 | 2019-11-26 | 843d4de
CPXPARAM_Read_DataCheck                          1
Number of nonzeros in lower triangle of Q = 2
Using Approximate Minimum Degree ordering
Total time for automatic ordering = 0.00 sec. (0.00 ticks)
Summary statistics for factor of Q:
  Rows in Factor            = 3
  Integer space required    = 3
  Total non-zeros in factor = 6
  Total FP ops to factor    = 14
Tried aggregator 1 time.
QP Presolve added 0 rows and 3 columns.
Reduced QP has 5 rows, 6 columns, and 14 nonzeros.
Reduced QP objective Q matrix has 3 nonzeros.
Presolve time = 0.00 sec. (0.00 ticks)
Parallel mode: using up to 32 threads for barrier.
Number of nonzeros in lower triangle of A*A' = 9
Using Approximate Minimum Degree ordering
Total time for automatic ordering = 0.00 sec. (0.00 ticks)
Summary statistics for Cholesky factor:
  Threads                   = 32
  Rows in Factor            = 5
  Integer space required    = 5
  Total non-zeros in factor = 15
  Total FP ops to factor    = 55
 Itn      Primal Obj        Dual Obj  Prim Inf Upper Inf  Dual Inf          
   0   1.0636929e+02   4.0119631e+04  3.00e+02  2.20e+01  4.02e+03
   1  -4.2647339e+03   1.3830857e+04  8.95e+00  6.58e-01  1.20e+02
   2  -5.7244038e+02   2.1778053e+03  4.31e-14  0.00e+00  1.57e-13
   3  -1.3359320e+02   2.1366540e+02  2.13e-14  7.11e-15  2.89e-14
   4  -1.5446636e+01   4.4803921e+01  1.11e-14  7.11e-15  1.88e-14
   5   3.4390904e-01   7.1112812e+00  5.77e-15  0.00e+00  4.52e-15
   6   1.9389541e+00   2.6972949e+00  2.44e-15  0.00e+00  1.66e-15
   7   2.0140768e+00   2.1563147e+00  2.22e-15  0.00e+00  8.86e-16
   8   2.0155916e+00   2.0388811e+00  1.13e-14  7.11e-15  7.93e-16
   9   2.0156158e+00   2.0194824e+00  7.55e-15  7.11e-15  8.75e-16
  10   2.0156165e+00   2.0162605e+00  1.21e-14  7.11e-15  1.71e-15
  11   2.0156165e+00   2.0157238e+00  8.99e-15  0.00e+00  2.50e-15
  12   2.0156165e+00   2.0156344e+00  8.55e-15  0.00e+00  2.23e-15
  13   2.0156165e+00   2.0156195e+00  4.44e-15  7.11e-15  2.29e-15
  14   2.0156165e+00   2.0156170e+00  7.66e-15  7.11e-15  1.32e-15
  15   2.0156165e+00   2.0156166e+00  7.99e-15  0.00e+00  5.56e-16
  16   2.0156165e+00   2.0156165e+00  5.33e-15  7.11e-15  2.01e-15
Barrier time = 0.01 sec. (0.04 ticks)

Total time on 32 threads = 0.01 sec. (0.04 ticks)
Solution status =  1 : optimal
Solution value  =  2.0156165232891565
Row  0 :   Slack =  18.642254  Pi =   0.000000
Row  1 :   Slack =  30.757886  Pi =   0.000000
Column  0 :   Value =   0.139115  Reduced Cost =  -0.000000
Column  1 :   Value =   0.598465  Reduced Cost =  -0.000000
Column  2 :   Value =   0.898396  Reduced Cost =  -0.000000

Process finished with exit code 0


'''