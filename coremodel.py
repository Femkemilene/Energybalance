# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 10:05:17 2017

@author: user
"""

import numpy as np
from numba import jit

secinyear = 3600*24*365
secinday = 3600*24
cT = 5.0e8#5.0e8
Q0 = 342 #W/m^2
seasonalampl=100 #TODO: check whether valid
G = 1.5e2
A = 2.05e1
sigma = 5.67e-8
Cref = 295 #ppm
epsilon = 1.0

def C(t,Cref=295):
    if t<100*secinyear:
        C=Cref*1.002**(t/secinyear)
    else:
        tafter100=t/secinyear-100
        C = 360*1.01**(tafter100) #TODO: check the pace of increase 
    if C>720:
        C=720
    if t<200*secinyear: #TODO: remove this part
        C=Cref*(1.01**(t/secinyear))
    else:
        C=Cref*(1.01**100)
    return C

def makeQ(seasonalampl,Q0,seasonality=True):    
    if seasonality==True:
        def Q(t):
            "This starts in summer, on the 21st of June"
            Qt = Q0+seasonalampl*np.cos(2*np.pi*t/secinyear)
            return Qt
    if seasonality ==False:
        def Q(t):
            return Q0
    return Q
 
def makeAlbedoFunctions(albType,albmin,k,Te,amp):
    if albType == 'sigmoid':
        def sigalph(T,amp):
            return albmin + amp/(1+np.exp(k*(T-Te)))    
        return sigalph
    if albType == 'linsig':
        def sigalphlin(T):
            return amp/2.+albmin-1./4.*k*amp*(T-Te)
        return sigalphlin
    
def bollin(T):
    return epsilon*sigma*Te**4+4*epsilon*sigma*Te**3*(T-Te)
    
def initsimplemodel(Q,alpha=0.3,climateChange=False):
    if climateChange==False:    
        def simplemodel(T,t):
            "Energy balance model. The time is in seconds."
            dT = 1./cT*(Q(t)*(1-alpha)+G-sigma*epsilon*T**4)
            return dT
            #return alpha
    else: 
#        def simplemodel(T,t):
#            "Energy balance model. The time is in seconds."
#            dT = 1./cT*(Q(t)*(1-alpha)+G\
#            +A*np.log(C(t)/Cref)-sigma*epsilon*T**4)
#            return dT
        def simplemodel(T,t):
            "Energy balance model. The time is in seconds."
            QCC = (1+0.001*t/secinyear)
            dT = 1./cT*(QCC*Q(t)*(1-alpha)+G\
            -sigma*epsilon*T**4)
            return dT
    return simplemodel

def initsigmoidmodel(Q,sigalph,amp=0.2,albmin = 0.25,climateChange=False): 
    if climateChange==False:
        def sigmoidmodel(T,t):
            "Same as simple model, but now albedo is a function of temperature" 
            dT = 1./cT*(Q(t)*(1-sigalph(T,amp))+G-sigma*epsilon*T**4)
            #dT =sigalph(T,amp)
            return dT
    else:
#        def sigmoidmodel(T,t):
#            "Energy balance model. The time is in seconds."
#            dT = 1./cT*(Q(t)*(1-sigalph(T,amp))+G\
#            +A*np.log(C(t)/Cref)-sigma*epsilon*T**4)
#            return dT
        def sigmoidmodel(T,t):
            QCC = (1+0.001*t/secinyear)
            dT = 1./cT*(QCC*Q(t)*(1-sigalph(T,amp))+G\
            -sigma*epsilon*T**4)
            return dT
    return sigmoidmodel

def initNoise(beta):
    noise = lambda X,t: beta
    return noise

def initOrnstein(gamma,climateChange=False):
    if climateChange==False:
        @jit(nopython=True)
        def drift(x,t):
            return -gamma*x+np.sin(2*np.pi*t/10)
    else:
        @jit(nopython=True)
        def drift(x,t):
            return -gamma*x+0.01*t
    return drift
    
def initOrnsteinCos(gamma,climateChange=False):
    if climateChange==False:
        @jit(nopython=True)
        def drift(x,t):
            return -gamma*x+np.cos(2*np.pi*t/10)
    else:
        @jit(nopython=True)
        def drift(x,t):
            return -gamma*x+np.cos(2*np.pi*t/1000)
    return drift
        
    
'''--------------------------------------------------------------- '''

'''Equations used for own faulty matrix calculations'''
#TODO: make this match the intertrations"
def nonlin(T,deltaT):
    dT = 1/cT*(Q0*(1-sigalph(T,0.4))+G-sigma*epsilon*T**4)/(2*deltaT)    
    return dT * secinyear 
def lin(T,deltaT):
    dT = 1/cT*(Q0*(1-sigalphlin(T))+G-bollin(T))/(2.*deltaT)  
    return dT * secinyear
def conalb(T,deltaT):
    dT = 1/cT*(Q0*(1-alpha)+G-sigma*epsilon*T**4)/(2.*deltaT)  
    return dT * secinyear
def diffusion(deltaT):
    dif = beta**2/2/deltaT**2    
    return dif * secinyear

def model(state,t):
    "Energy balance model. The time is in seconds."
    T=state[0]
    f = state[1]
    dT = 1./cT*(Q0*(1-w*f)+G+A*np.log(C(t)/Cref)-sigma*epsilon*T**4)
    equif = (305-T)*1./60.0
    if equif <0.1:
        equif=0.1
    if equif>0.7:
        equif=0.7
    df = -1./(30*cT)*(f-equif)
    return dT,df
    
def model2(state,t):
    "Energy balance model. The time is in seconds."
    T=state[0]
    f = state[1]
    dT = 1./cT*(Q0*(1-w*f)+G+A*np.log(C(t)/Cref)-sigma*epsilon*T**4)
    equif = (305-T)*1./6.0
    df = 1./(50*cT)*(f-equif)
    return dT,df