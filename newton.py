# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 10:59:47 2020

@author: bhj
"""

import numpy as np
from newton import *

# 求解    x1^2 + x2^2 - 5 = 0
#       (x1+1)*x2 - 3*x1 + 1 = 0

def f(x):
    fx = np.array([[x[0][0]**2 + x[1][0]**2 - 5],
                   [(x[0][0] + 1)*x[1][0] -3*x[0][0] - 1]])
    return fx

def jacobi(x):
    J = np.array([[ 2*x[0][0],2*x[1][0]],
                  [x[1][0] - 3,x[0][0] + 1]])
    return J

def newton(e,N,x):
    k = 0
    err = 1
    while k < N and err > e:
        d = np.linalg.solve(jacobi(x),-f(x))
        err = np.linalg.norm(d)
        x = x + d
        print(k,'x:',x,)

        k = k +1
    return x
    
if __name__ == "__main__":
    x0 = np.array([ [6],
                    [5] ])
    N = 10
    e = 0.001
    B=np.eye(2)
    ans=newton(e , N , x0 )
    print(ans)        