#Transition function for on demand MDPs
#Learns the probability model as we progress through the MDP
#The offline environment

import numpy as np
import GPy
import application

B = 1.2 #Bounding constant on environmental change

#Add xpredict (beta learning)

def cpredict(state,action,k,c_k):

    if len(state.chist) == 0:
        #If we have no data, sample from a normal with a mean of 0 and a variance of 1

        delta_c = np.random.normal(0,1,1)
        state.chist = np.append(state.chist, np.array([[delta_c]]), axis=0)
        return c_k+delta_c

    #Use GP-MCMC to model change in size (c)
    m = GPy.models.GPRegression(state.chist, state.ahist) #m = GPy.models.GPRegression(np.array([1,2,3,4]), np.array([2,5,7,9]))
    
    #Set kernel hyperparameters to HMC output
    hmc = GPy.inference.mcmc.HMC(m,stepsize = 5e-2)
    s = hmc.sample(num_samples = 1000)
    s = s[300:] #Burn in
    m.kern.variance[:] = s[:,0].mean()
    m.kern.lengthscale[:] = s[:,1].mean()
    m.likelihood.variance[:] = s[:,2].mean()

    #Sample from GP
    mean, variance = m.predict(np.array([c_k]), np.array([action]))
    delta_c = np.random.normal(mean,variance,1)

    state.chist = np.append(state.chist, np.array([[delta_c]]), axis=0)

    return c_k+delta_c

def model(state,action):
    #Return a sample from our learned probability model
    for k in range(state.nparts):
        x_k, c_k = state.decompose(k)
        ahist = state.ahist

        #Find change in probability tensor and size
        dx_k = xpredict(state,action,k,x_k)
        dc_k = cpredict(state,action,k,c_k)

        #Add action to history
        state.ahist = np.append(state.ahist, [action], axis=0)
        
        #Reconstruct state
        state.reconstruct(x_k+dx_k,c_k+dc_k,k)
    state.done = application.checkTermination(state)
    return state

class OnDemandEnvironment:
    """
    r - the reward function (inputted by user): <r(s, a, s2) = R_a(s,s')> (should be the same as in transition)
    s0 - the initial state
    """
    def __init__(self):
        self.r = application.r
        self.state = application.s0
    def transition(action):
        s2 = model(self.state,action)

        self.state = s2
        return self.r(self.state, action)