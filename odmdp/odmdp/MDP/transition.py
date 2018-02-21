#Transition function for on demand MDPs
#Learns the probability model as we progress through the MDP
#The offline environment

import numpy as np
import GPy
import application
from math import sqrt

B = 1.2 #Bounding constant on environmental change

def CON(alpha1,alpha2):
    #Given two dirichlet hyperparameters, sample from their convolution
    x1 = np.random.dirichlet(alpha1,1)
    x2 = np.random.dirichlet(alpha2,1)

    dx = x1-x2
    return dx
        
def objective(eps,state,k):
    #Return sum of lengthscales
    data = alpha(state.xhist[k])


def alpha(eps, mu):
    #returns the alpha vectors for a given epsilon vector
    sh = mu.shape
    mu = mu.reshape(sh[0],np.prod(sh[1:]))
    print(mu.shape)
    print(mu)
    #Compute row reduction closed form + alpha computation
    s = np.sum(mu[:,:-1],axis=1)
    S = s - s**2
    print(s)
    print(S)
    

    print(mu)
    print(np.array([mu[i][:-1]*S[i] for i in range(sh[0])]))
    #Compute alpha values
    alpha = np.empty(mu.shape)
    alpha[:,:-1] = np.array([mu[i][:-1]*S[i]/eps[i] for i in range(sh[0])]) - mu[:,:-1]
    alpha[:,-1] = mu[:,-1]*(1-mu[:,-1])/eps - mu[:,-1]

    alpha.reshape(sh)

    return alpha

    

#def xpredict(state,action,k,x_k):
    #If we have no data, sample from a CON distribution

    #Use beta learning to find delta
    
    

def cpredict(state,action,k,c_k):

    if len(state.chist) == 0:
        #If we have no data, sample from a normal with a mean of 0 and a variance of 1

        delta_c = np.random.normal(0,1,1)
        state.chist[k] = np.append(state.chist[k], np.array([[delta_c]]), axis=0)
        return c_k+delta_c

    #Use GP-MCMC to model change in size (c)
    m = GPy.models.GPRegression(state.chist[k], state.ahist)
    
    #set prior
    m.kern.set_prior(GPy.priors.Gamma.from_EV(2.,4.))
    m.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(2.,4.))

    #Set kernel hyperparameters to HMC output
    hmc = GPy.inference.mcmc.HMC(m,stepsize = 5e-2)
    s = hmc.sample(num_samples = 200)
    s = s[100:] #Burn in
    m.kern.variance[:] = s[:,0].mean()
    m.kern.lengthscale[:] = s[:,1].mean()
    m.likelihood.variance[:] = s[:,2].mean()

    #Sample from GP
    mean, variance = m.predict(np.array([c_k]), np.array([action]))
    delta_c = np.random.normal(mean,sqrt(variance),1)

    state.chist[k] = np.append(state.chist[k], np.array([[delta_c]]), axis=0)

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


mu = np.array([[[.1,.2],[.3,.4]],[[.1,.2],[.3,.4]]])
print(mu)
print(alpha(np.array([.1,.1]),mu))

