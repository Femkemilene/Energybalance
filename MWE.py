# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 12:41:10 2017

Minimal working example for using ergoNumAna

@author: Femke
"""

import numpy as np
import ergoNumAna as CC

import scipy.sparse.linalg as linalg

"Ornstein-Uhlerbeck process, 1D"
gamma=0.5
sigma=2
def drift(x):
    return -gamma*x
diffusion=np.ones((1,1))*sigma**2

"Domain over which to calculate"
dx = 0.25
domain = np.arange(-15,15,dx) 
xlen=len(domain)
domain = domain.reshape(1,xlen) #reshaping so that it's a 2D array.

"using Chang-Cooper script"
matrix1D=CC.ChangCooper(domain,[xlen],[dx],drift,diffusion).toarray()

print sum(sum(matrix1D))

"Ornstein-Uhlerbeck process, two-dimensional"
gamma1=0.3
gamma2=0.5
sigma2=2
def drift(x):
    driftx = -gamma1*x[0]
    #driftx = 0
    drifty = -gamma2*x[1]    
    return driftx,drifty
diffusion=np.diag([2,2])*sigma2**2

"Domain over which to calculate"
dx = np.array([0.25,0.25]) #seperate resolution for each dimension
xdomain = np.arange(-0.25,0.5,dx[0])
ydomain = np.arange(-15,15,dx[1])
xlen=len(xdomain)
ylen=len(ydomain)
domain = np.array([np.tile(xdomain,ylen),np.repeat(ydomain,xlen)])

"Using Chang-Cooper script"
matrix2D = CC.ChangCooper(domain,[xlen,ylen],dx,drift,diffusion)
print np.mean(matrix2D)
print np.sum(matrix2D)
hoi=matrix2D.toarray()

"Testing this 2D-array a bit"
initialDistribution=np.zeros((xlen,ylen))
initialDistribution[int(xlen/2),int(ylen/2)]=1
#matrixExp = ma