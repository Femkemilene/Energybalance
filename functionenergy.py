# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 11:05:33 2017

@author: user
"""

import numpy as np
import plotenergy as pltE
import scipy.sparse.linalg as linalg

def wiSuDifference(timeseries,sampleperyear):
    years = len(timeseries)/sampleperyear
    dT = np.zeros((years))
    for t in range(years):
        summert = t*sampleperyear + int(0.2*sampleperyear)
        wintert = t*sampleperyear + int(0.7*sampleperyear)
        dT[t]=timeseries[summert]-timeseries[wintert]
    return np.mean(dT)    
    
def oscDifference(timeseries,sampleperyear):
    years = len(timeseries)/sampleperyear
    period = 10 #yr
    dx=np.zeros((years/period))
    for t in range(0,years,period):
#        highx = t*sampleperyear +int(0.3*period*sampleperyear)
#        lowx = t*sampleperyear + int(0.8*period*sampleperyear)
#        dx[t/period]=max(timeseries[highx-5:highx+5])-\
#                      min(timeseries[lowx-5:lowx+5])
        timeslice =np.array([t,t+sampleperyear*period])
        dx[t/period]=max(timeseries[timeslice]-min(timeseries[timeslice]))
    return np.mean(dx)
    
    
def climateChange(timeseries,sampleperyear,OrnsteinorEnergy):
    preindustrialT = np.mean(timeseries[:50*sampleperyear])
    warmedworldT = np.mean(timeseries[-50*sampleperyear:])
    return warmedworldT - preindustrialT

def calculatebetal(drift,eigenvectors,ran,l):
    driftvector = np.array([drift(T) for T in ran])
    dx = ran[1]-ran[0]
    def observable(T):
        return T
    obsvector = observable(ran)
    potentialpart = eigenvectors[:,l].dot(driftvector)
    observablepart = eigenvectors[:,l].dot(obsvector)
    return potentialpart * observablepart * dx*dx
    
def responseRatio(drift,ran, eigenvalues,eigenvectors,w1,w2):
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    bl = [calculatebetal(drift,eigenvectors,ran,l) for \
            l in range(len(eigenvalues))]
    ll = eigenvalues
    sum1 = sum(bl*ll/(ll**2+w2**2))
    sum2 = sum(bl*ll/(ll**2+w1**2))
    return sum1/sum2
    
def computeEigenpairs(matrix,domain,OrnsteinorEnergy):
    eigenpairs = linalg.eigs(matrix,which='SM')
    eigenvalues = eigenpairs[0]
    eigenvectors = eigenpairs[1]
    
    #pltE.eigenvaluesplot(eigenvalues,last=True)
    #pltE.eigenvectorplot(eigenvectors[:,0],domain,OrnsteinorEnergy,last=True)
    #for x in range(0,800,50):
    #    eigenvectorplot(eigenvectors[x:x+3],)
    
    "Eigenvalues and eigenvectors linear"
    #matrixlin = CC.ChangCooper(np.makeMatrix(200,'lin') #TODO; complete this thingie
    
    #Eigenvalues and eigenvectors
#    eigenpairslin = linalg.eigs(matrixlin,which='SM')
#    eigenvalueslin = eigenpairslin[0]
#    eigenvectorslin = eigenpairslin[1]
    
    #eigenvaluesplot(eigenvalueslin,last=True)
    #for x in range(0,10,1):
    #    eigenvectorplot(eigenvectors[x:x+1],domain,OrnsteinOrEnergy)
    
    "Inverse shift method"
#    inversematrix = linalgnp.inv(matrix-np.identity(len(matrix)))
#    eigenvaluesshift = linalg.eigs(inversematrix,which='LM')[0]
#    eigenvaluesshiftl = 1./eigenvaluesshift +1
    return eigenvalues,eigenvectors
    
def makeTest(domain):
    length = len(domain)
    test = np.zeros(len(domain))
    locationdelta=int(length/2)
    test[locationdelta]=1
    return length,test
    
"-------------------------------"
def makeMatrix(matrixSize,model,reflectiveBC=True):
    """This matrix does not perserve probability, use ergoNum instead"""
    deltaT = 50./matrixSize #so that we have 200 data points
    matrix = np.zeros((matrixSize,matrixSize))
    
    #Drift-part
    if model=='nonlin':
        matrix[0,1]=nonlin(262,deltaT)
        for i in range(1,matrixSize-1):
            matrix[i,i+1]=nonlin(262+deltaT*i,deltaT)
            matrix[i,i-1]=-nonlin(262+deltaT*i,deltaT)
        matrix[-1,-2]=-nonlin(262+matrixSize*deltaT,deltaT)
    if model =='conalb':
        matrix[0,1]=conalb(262,deltaT)
        for i in range(1,matrixSize-1):
            matrix[i,i+1]=conalb(262+deltaT*i,deltaT)
            matrix[i,i-1]=-conalb(262+deltaT*i,deltaT)
        matrix[-1,-2]=-conalb(262+matrixSize*deltaT,deltaT)
    if model == 'lin':
        matrix[0,1]=lin(262,deltaT)
        for i in range(1,matrixSize-1):
            matrix[i,i+1]=lin(262+deltaT*i,deltaT)
            matrix[i,i-1]=-lin(262+deltaT*i,deltaT)
        matrix[-1,-2]=-lin(262+deltaT*i,deltaT)

    
    #Diffusion-part
    def addDiffusion(matrix):
        dif = diffusion(deltaT)
        matrix[0,0]+=-2*dif
        matrix[0,1]+=dif
        for i in range(1,matrixSize-1):
            matrix[i,i+1]+=dif
            matrix[i,i-1]+=dif
            matrix[i,i]=-2*dif
        matrix[-1,-2]+=dif
        matrix[-1,-1]=-2*dif
        return matrix
    matrix = addDiffusion(matrix)
        
    if reflectiveBC==True:
        matrix[0,0]=matrix[1,0]
        matrix[0,1]=matrix[1,1]
        matrix[0,2]=matrix[1,2]
        matrix[-1,-1]=matrix[-2,-1]
        matrix[-1,-2]=matrix[-2,-2]
        matrix[-1,-3]=matrix[-2,-3]
    return matrix
