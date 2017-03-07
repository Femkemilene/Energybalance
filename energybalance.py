# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 11:39:55 2017

@author: user
"""
import numpy as np
import scipy.integrate as integr
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
import solve_sde as sde
import time

import ergoNumAna as CC
import plotenergy as pltE
import coremodel as core
import functionenergy as fuE

starttime=time.time()

'adjustable parameters'
#w = 1 #whiteness of snow
#f0 = 0.3 
alpha0 = 0.7
T0 = 283.5 #The switch is around 284 degrees.
alpha=0.3
#beta = 2.5e7
ensembleSize=100
beta=1.0*10e-5 #default 5*10^-5
sigma2=2 #diffusion term Ornstein
matrixShit = True
model = "Ornstein"
emergentLoop = 'gamma'

"shape sigmoid"
k=0.5 #steepness logistic function
Te = 284 #for the logistic function
albmin = 0.25 #minimum value albedo (default 0.2)
amp = 0.3 #amplitude (default 0.3)

'constant parameters' 
secinyr = 3600*24*365
cT = 2.5e8#5.0e8
Q0 = 342 #W/m^2
seasonalampl=300 #TODO: check whether valid
G = 1.5e2
A = 2.05e1
sigma = 5.67e-8
Cref = 295 #ppm
epsilon = 1.0

"Initialisation"
Q = core.makeQ(seasonalampl,Q0,seasonality=True) #solar insolation
sigalphlin = core.makeAlbedoFunctions('linsig', albmin,k,Te,amp)

"Choice of loops"
loopdict = {"gamma":np.arange(0.05,1.0,0.15),\
            "albmin": np.arange(0.1,0.5,0.1),
            "amp": np.arange(0.1,0.6,0.05) }
looplist = loopdict[emergentLoop]

"Test for emergent constraints"
emergent = np.zeros((3,2,len(looplist)))
RR = np.zeros(len(looplist))

for x in np.arange(len(looplist)):
    if emergentLoop=="gamma":
        gamma = looplist[x]
    elif emergentLoop == "albmin":
        albmin = looplist[x]
        alpha = looplist[x]
    elif emergentLoop == 'amp':
        gamma=looplist[x]
        alpha = looplist[x]
    
    if model == "Energy":    
        noise = core.initNoise(beta)
        sigalph = core.makeAlbedoFunctions('sigmoid',albmin,k,Te,amp)
       
        years=300
        spY = 10 #samples per year
        dtime = 1.0/spY*secinyr
        simplemodel = core.initsimplemodel(Q,alpha)
        sigmoidmodel = core.initsigmoidmodel(Q,sigalph,amp=amp,albmin=albmin)
        
        T0sim = integr.odeint(simplemodel,T0,np.arange(0,100*secinyr,dtime))[-1,0]    
        T0sig = integr.odeint(sigmoidmodel,T0,np.arange(0,50*secinyr,dtime))[-1,0]  
        T0sto = sde.solve_sde(alfa = sigmoidmodel,beta = noise,X0=[T0], dt=dtime,N=1000)[1][-1,0]
        
        outputsimple = integr.odeint(simplemodel,T0sim, np.arange(0,years*secinyr,dtime))
        output = integr.odeint(sigmoidmodel,T0sig,np.arange(0,years*secinyr,dtime))
        stochasticOutput = np.mean(np.array([\
            sde.solve_sde(alfa = sigmoidmodel,beta = noise,X0=[T0sto], \
            dt=dtime,N=spY*years)[1] for e in range(ensembleSize)]),axis=0)
        emergent[0,0,x]=fuE.wiSuDifference(outputsimple,spY)
        emergent[1,0,x]=fuE.wiSuDifference(output,spY)
        emergent[2,0,x]=fuE.wiSuDifference(stochasticOutput,spY)
         
        simplemodel = core.initsimplemodel(Q,alpha,climateChange=True)
        sigmoidmodel = core.initsigmoidmodel(Q,sigalph,amp=amp,albmin=albmin,climateChange=True)
        
        outputsimple = integr.odeint(simplemodel,T0sim, np.arange(0,years*secinyr,dtime))
        output = integr.odeint(sigmoidmodel,T0sig,np.arange(0,years*secinyr,dtime))
        stochasticOutput = np.mean(np.array([\
            sde.solve_sde(alfa = sigmoidmodel,beta = noise,X0=[T0sto], \
            dt=dtime,N=spY*years)[1] for e in range(ensembleSize)]),axis=0)
        emergent[0,1,x]=fuE.climateChange(outputsimple,spY,model)     
        emergent[1,1,x]=fuE.climateChange(output,spY,model)  
        emergent[2,1,x]=fuE.climateChange(stochasticOutput,spY,model)  
        
        #pltE.plotT(outputsimple,stochasticOutput,outputsimple) #gaat verkeerd met tijdsreeks
        #pltE.plotAlb(output,stochasticOutput,amp,sigalph,constant=alpha)
    if model == "Ornstein":
        endTime=300
        spY=10
        dtime = 1.0/spY
        
        noise = core.initNoise(sigma2)
        Ornsteinmodel = core.initOrnstein(gamma)
        stochasticOutput1 = np.mean(np.array([\
            sde.solve_sde(alfa=Ornsteinmodel, beta=noise,X0=[0],dt=dtime,\
            N=spY*endTime)[1] for e in range(ensembleSize)]),axis=0)
        plt.plot(stochasticOutput1)
        plt.title('gamma is: %s' %gamma)
        plt.show()
        emergent[0,0,x]=fuE.oscDifference(stochasticOutput1,spY)
        
        Ornsteinmodel=core.initOrnstein(gamma,climateChange=True)        
        stochasticOutput2=np.mean(np.array([\
            sde.solve_sde(alfa=Ornsteinmodel, beta=noise,X0=[0],dt=dtime,\
            N=spY*endTime)[1] for e in range(ensembleSize)]),axis=0)
        emergent[0,1,x]=fuE.climateChange(stochasticOutput2,spY,model) 
        plt.plot(stochasticOutput2)
        plt.title('gamma is: %s' %gamma)
        plt.show()
        
    "Using Chang-Cooper scheme"
    if model == "Energy":
        def drift(T): 
            """Albedo sigmoid shape, nonlinear temperature"""
            dT = 1/cT*(Q0*(1-sigalph(T,amp))+G-sigma*epsilon*T**4)
            return dT*secinyr
        def nonlin3(T): 
            """Constant albedo, nonlinear temperature"""
            dT = 1/cT*(Q0*(1-alpha)+G-sigma*epsilon*T**4)
            return dT*secinyr
        diffusion2=np.ones((1,1))*beta**2/2*secinyr
    elif model == "Ornstein":
        def drift(x):
            return -gamma*x
        diffusion3=np.ones((1,1))*sigma2**2
    
    if model == "Energy":
        pasttime=5
        dx = 0.2
        domain=np.arange(242,302,dx)
        (Tlen,test) = fuE.makeTest(domain)      
        matrix=CC.ChangCooper(domain.reshape(1,Tlen),[Tlen],[dx],drift,diffusion2).toarray()     
    
    if model == "Ornstein":
        pasttime=8    
        domain = np.arange(-25,25,.25)  
        (xlen,test) = fuE.makeTest(domain)       
        matrix=CC.ChangCooper(domain.reshape(1,xlen),[xlen],[0.25],drift,diffusion3).toarray()
    
    matrixexp = linalg.expm(matrix*pasttime)
    #pltE.plotpdf(domain,matrixexp.dot(test),pasttime)
    print "Total probability is: ", sum(matrixexp.dot(test))
       
    "Eigenvalues and eigenvectors non-linear"
    if matrixShit ==True:
        "Calculating eigenvalues/eigenvectors"
        (eigenvalues,eigenvectors) = fuE.computeEigenpairs(matrix,domain,model)
        RR[x] = fuE.responseRatio(drift,domain, eigenvalues,eigenvectors,0.3,5)
    
pltE.plotEmergent(emergent,looplist,model)
pltE.plotRR(looplist,RR,emergentLoop)

endtime=time.time()
print "The total time elapsed is: ",endtime-starttime, "s"

"Some for easy testing"
#for gamma in np.arange(0,1,0.2):
#    Ornsteinmodel = core.initOrnstein(gamma)
#    hoi = sde.solve_sde(alfa=Ornsteinmodel, beta=noise,X0=[0],dt=dtime,\
#            N=spY*endTime)[1]
#    plt.plot(hoi)
#    plt.show()
          