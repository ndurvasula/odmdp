#Transition function for on demand MDPs
#Learns the probability model as we progress through the MDP
#The offline environment

import numpy as np
import GPy
import application
from math import sqrt
from scipy import optimize

def CON(alpha1,alpha2):
    #Given two dirichlet hyperparameters, sample from their convolution
    x1 = np.random.dirichlet(alpha1,1)
    x2 = np.random.dirichlet(alpha2,1)

    dx = x1-x2
    return dx
        
def objective(eps,state,k,x_k):
    eps1 = eps[0]
    eps2 = eps[1]
    
    #Return sum of lengthscales
    data_raw1 = alpha(eps1,state.xhist[k])
    data_cols1 = [data_raw1[:,i].reshape(1,data_raw1.shape[0],1) for i in range(data_raw1.shape[1])]

    data_raw2=alpha(eps2,np.append(state.xhist[k][1:],np.array([x_k]),axis=0))
    data_cols2 = [data_raw2[:,i].reshape(1,data_raw2.shape[0],1) for i in range(data_raw2.shape[1])]

    GPs1 = [GPy.models.GPRegression(state.ahist,data_cols1[i]) for i in range(data_raw1.shape[1])]
    GPs2 = [GPy.models.GPRegression(state.ahist,data_cols2[i]) for i in range(data_raw2.shape[1])]

    #Perform HMC on every GP
    kvar1 = np.empty([1,data_raw1.shape[1]])
    klen1 = np.empty([1,data_raw1.shape[1]])
    lvar1 = np.empty([1,data_raw1.shape[1]])

    kvar2 = np.empty([1,data_raw2.shape[1]])
    klen2 = np.empty([1,data_raw2.shape[1]])
    lvar2 = np.empty([1,data_raw2.shape[1]])

    for i in range(data_raw.shape[1]):
        
        #set prior
        GPs1[i].kern.set_prior(GPy.priors.Gamma.from_EV(2.,4.))
        GPs1[i].likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(2.,4.))

        GPs2[i].kern.set_prior(GPy.priors.Gamma.from_EV(2.,4.))
        GPs2[i].likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(2.,4.))

        #Set kernel hyperparameters to HMC output
        hmc1 = GPy.inference.mcmc.HMC(GPs1[i],stepsize = 5e-2)
        s1 = hmc.sample(num_samples = 200)
        s1 = s1[100:] #Burn in
        kvar1[i] = s1[:,0].mean()
        klen1[i] = s1[:,1].mean()
        lvar1[i] = s1[:,2].mean()

        hmc2 = GPy.inference.mcmc.HMC(GPs2[i],stepsize = 5e-2)
        s2 = hmc.sample(num_samples = 200)
        s2 = s2[100:] #Burn in
        kvar2[i] = s2[:,0].mean()
        klen2[i] = s2[:,1].mean()
        lvar2[i] = s2[:,2].mean()

    return np.sum(kvar1*klen1*lvar1*kvar2*klen2*lvar2)


def alpha(eps, mu):
    #Returns the alpha vectors for a given epsilon vector
    eps = abs(eps)
    
    #Compute alpha values
    return np.array([eps[i]*(mu[i]*(1-mu[i])) for i in range(mu.shape[0])])

    

def xpredict(state,action,k,x_k):
    #If we have no data, sample from a CON distribution
    if state.xhist.shape[0] == 0:
        #Normally distribute alpha vectors with mean 1, std 1, truncate to positive
        alpha1 = np.random.normal(1,1,state.xhist.shape[1])
        while not np.all(np.greater(alpha1,np.zeros(alpha1.shape))):
            alpha1 = np.random.normal(1,1,state.xhist.shape[1])
    
        alpha2 = np.random.normal(1,1,state.xhist.shape[1])
        while not np.all(np.greater(alpha2,np.zeros(alpha2.shape))):
            alpha2 = np.random.normal(1,1,state.xhist.shape[1])
        
        #Compute convolution and reshape
        deltax_k = CON(alpha1,alpha2).reshape(state.sh[k])

        #Compute new x_k
        nx_k = x_k+deltax_k
        state.xhist[k] = np.append(state.xhist[k], np.array([nx_k]), axis=0)

    #Find eps that minimizes our objective
    #Use Nelder Mead to minimize the lengthscale*variance*likelihood_variance (find best data points and hyperparameters)

    #Start Nelder-Mead with epsilon array of all 1s
    eps_0 = np.empty([2,state.xhist[k].shape[0]])
    eps_0.fill(1)
    
    opt = optimize.minimize(objective, eps_0, args=(state,k,x_k,), method='Nelder-Mead')
    eps1, eps2 = opt.x

    data_raw1 = alpha(eps1,state.xhist[k])
    data_cols1 = [data_raw1[:,i].reshape(1,data_raw1.shape[0],1) for i in range(data_raw1.shape[1])]

    data_raw2=alpha(eps2,np.append(state.xhist[k][1:],np.array([x_k]),axis=0))
    data_cols2 = [data_raw2[:,i].reshape(1,data_raw2.shape[0],1) for i in range(data_raw2.shape[1])]

    GPs1 = [GPy.models.GPRegression(state.ahist,data_cols1[i]) for i in range(data_raw1.shape[1])]
    GPs2 = [GPy.models.GPRegression(state.ahist,data_cols2[i]) for i in range(data_raw2.shape[1])]

    #Perform HMC on every GP
    for i in range(data_raw.shape[1]):
        
        #set prior
        GPs1[i].kern.set_prior(GPy.priors.Gamma.from_EV(2.,4.))
        GPs1[i].likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(2.,4.))

        GPs2[i].kern.set_prior(GPy.priors.Gamma.from_EV(2.,4.))
        GPs2[i].likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(2.,4.))

        #Set kernel hyperparameters to HMC output
        hmc1 = GPy.inference.mcmc.HMC(GPs1[i],stepsize = 5e-2)
        s1 = hmc.sample(num_samples = 200)
        s1 = s1[100:] #Burn in
        GPs1[i].kern.variance[:] = s1[:,0].mean()
        GPs1[i].kern.lengthscale[:] = s1[:,1].mean()
        GPs1[i].likelihood.variance[:] = s1[:,2].mean()

        hmc2 = GPy.inference.mcmc.HMC(GPs2[i],stepsize = 5e-2)
        s2 = hmc.sample(num_samples = 200)
        s2 = s2[100:] #Burn in
        GPs2[i].kern.variance[:] = s2[:,0].mean()
        GPs2[i].kern.lengthscale[:] = s2[:,1].mean()
        GPs2[i].likelihood.variance[:] = s2[:,2].mean()

    norms1 = [GPs1[i].predict(np.array([action]))[0] for i in range(data_raw1.shape[1])]
    norms2 = [GPs2[i].predict(np.array([action]))[0] for i in range(data_raw2.shape[1])]

    #Compute alpha vectors using truncated normal output from GP
    alpha1 = np.array([np.random.normal(norms1[i][0],sqrt(norms1[i][1]),1) for i in range(data_raw1.shape[1])])
    while not np.all(np.greater(alpha1,np.zeros(alpha1.shape))):
        alpha1 = np.array([np.random.normal(norms1[i][0],sqrt(norms1[i][1]),1) for i in range(data_raw1.shape[1])])
    
    alpha2 = np.array([np.random.normal(norms2[i][0],sqrt(norms2[i][1]),1) for i in range(data_raw2.shape[1])])
    while not np.all(np.greater(alpha2,np.zeros(alpha2.shape))):
        alpha2 = np.array([np.random.normal(norms2[i][0],sqrt(norms2[i][1]),1) for i in range(data_raw2.shape[1])])

    #Compute convolution and reshape
    deltax_k = CON(alpha1,alpha2).reshape(state.sh[k])

    #Compute new x_k
    nx_k = x_k+deltax_k
    state.xhist[k] = np.append(state.xhist[k], np.array([x_k]), axis=0)

    return nx_k

    
def cpredict(state,action,k,c_k):

    if state.chist.shape[0] == 0:
        #If we have no data, sample from a normal with a mean of 0 and a variance of 1

        delta_c = np.random.normal(0,1,1)
        state.chist[k] = np.append(state.chist[k], np.array([[delta_c]]), axis=0)
        return c_k+delta_c

    #Use GP-MCMC to model change in size (c)
    m = GPy.models.GPRegression(state.ahist,state.chist[k])
    
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
    mean, variance = m.predict(np.array([action]))
    delta_c = np.random.normal(mean,sqrt(variance),1)

    state.chist[k] = np.append(state.chist[k], np.array([[c_k]]), axis=0)

    return c_k+delta_c

def model(state,action):
    #Return a sample from our learned probability model
    for k in range(state.nparts):
        x_k, c_k = state.decompose(k)
        ahist = state.ahist

        #Find change in probability tensor and size
        nx_k = xpredict(state,action,k,x_k)
        nc_k = cpredict(state,action,k,c_k)

        #Add action to history
        state.ahist = np.append(state.ahist, [action], axis=0)
        
        #Reconstruct state
        state.reconstruct(nx_k,nc_k,k)
    state.done = application.checkTermination(state)
    return state

class OnDemandEnvironment:
    """
    r - the reward function (inputted by user): <r(s, a, s2) = R_a(s,s')> (should be the same as in transition)
    s0 - the initial state
    """
    def __init__(self,state_0):
        self.r = application.r
        self.state = state_0
    def transition(action):
        s2 = model(self.state,action)

        self.state = s2
        return self.r(self.state, action)


