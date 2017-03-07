# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 09:58:50 2017

@author: Femke Nijsse
"""

import matplotlib.pyplot as plt
import numpy as np
import coremodel as core
plt.rcParams['font.size']=14
secinyear = 3600*24*365


'''Plotting of physical parameters'''
def plotT(output,stochasticOutput,simpleoutput=0):
    time = stochasticOutput[0]/secinyear
    plt.plot(time,output,label='deter')
    if type(simpleoutput)==int:
        plt.plot(time,stochasticOutput[1],label='stoch')
    else:
        plt.plot(time,simpleoutput,label='simple')
    plt.legend()
    plt.ylabel("T")
    plt.xlabel("time (yr)")
    plt.show()

def plotAlb(output,stochasticOutput,amp,sigalph,constant=0):
    "Make a plot of the albedo, calculated from temperature"
    time=stochasticOutput[0]/secinyear
    plt.plot(time,sigalph(output,amp)) #TODO: check whether correct
    plt.plot(time,sigalph(stochasticOutput[1],amp))
    if constant != 0:
        plt.plot(time,constant*np.ones((len(output))))
    plt.ylabel("albedo")
    plt.xlabel("time (yr)")
    plt.show()
    
"Plotting of the emergent constraints that might be there"
def plotEmergent(emergent,p,OrnsteinorEnergy):
    if OrnsteinorEnergy ==  "Energy":
        sc=plt.scatter(emergent[0,0],emergent[0,1],c=p,label='constant albedo')
        plt.scatter(emergent[1,0],emergent[1,1],marker='d',c=p,label='deterministic')
        plt.scatter(emergent[2,0],emergent[2,1],marker='s',c=p,label='stochastic')
        cb=plt.colorbar(sc)
        cb.set_label('albedo')
        plt.xlabel('sensitivity season')
        plt.ylabel('sensitity century')
    else:
        sc =plt.scatter(emergent[0,0],emergent[0,1],c=p,label="Ornstein")
        plt.xlabel('sensitivity to sine forcing')
        plt.ylabel('sens. to linearly increas. forcing')
        cb = plt.colorbar(sc)
        cb.set_label('gamma')
    #plt.plot(emergent[2,0],emergent[2,1],label='stochastic')
    plt.legend()
    plt.show()
    
def plotRR(parameter,RR,emergentLoop):
    plt.plot(parameter,RR,'o')
    plt.xlabel(emergentLoop)
    plt.ylabel('susceptibility ratio')
    plt.show()
    

"Plotting of eigenfunctionspectrum"
def plotpdf(domain,matrixexpdottest,pasttime):
    plt.plot(domain,matrixexpdottest)
    plt.xlabel("T (K)")
    plt.title("pdf after %s yr" % pasttime)  
    plt.show()        
    
def eigenvaluesplot(a,last=False):
    plt.figure(10)
    plt.scatter(a.real,a.imag,s=10)    
    plt.ylabel('Imaginary')
    plt.xlabel('Real')
    if last==True:
        plt.grid()
        plt.show()
    
def eigenvectorplot(a,domain,OorE,last=False):
    plt.figure(20)
    plt.plot(domain,a.real)
#    for x in range(len(a)):
#        plt.plot(262+np.arange(0,deltaT*matrixSize,deltaT),abs(a[x]))
    plt.ylabel('eigenfunction')
    if OorE == "Ornstein":
        plt.xlabel('position')
    elif OorE == "Energy":
        plt.xlabel('temperature')
    if last==True:
        plt.grid()  
        plt.show()