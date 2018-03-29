#General on demand environment

import numpy as np
import GPy

class Environment():
    """
    s0 - the initial state
    """
    def __init__(self,s0):
        self.s0 = s0
        
        #State delta history and action history in our walk so far for each partition
        self.dxhist = [np.empty([0,np.prod(np.array(s0.sh[k]))]) for k in range(s0.nparts)]
        self.chist = [np.array([]) for k in range(s0.nparts)]
        self.ahist = []

        self.GP = []


#Set kernel hyperparameters to HMC output
        hmc1 = GPy.inference.mcmc.HMC(GPs1[i],stepsize = 5e-2)
        s1 = hmc.sample(num_samples = 200)
        s1 = s1[100:] #Burn in
        

