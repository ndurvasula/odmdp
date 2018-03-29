#On demand environment solver

import numpy as np
import GPy
import space
"""
application contains all app specific code

ACTION_SIZE - length of array denoting each action (all actions have shape [ACTION_SIZE,])
SIZE_CHANGE - np array of booleans, if SIZE_CHANGE[k] is True, that means that taking an action can change the size of the kth partition

class Application
> subsolver(transition) - solves the RL problem given that transition(state,action) performs a state transition and returns the *first* action
> explore(dxhist,chist,ahist) - generates a random action by user-chosen distribution, may utilize dxhist, chist, ahist
> > dxhist - delta x history for all k
> > chist - delta c history for all k
> > ahist - action history
"""
from application import Application

class Solver():
    """
    s0 - the initial state
    e - the exploration constant, at time t, the probability that we explore instead of subsolve is e^t (t starts at 0)
    """
    def __init__(self,s0,e):
        self.state = s0
        self.t = 0
        
        #State delta history and action history in our walk so far for each partition
        self.dxhist = [np.empty([0,np.prod(np.array(self.state.sh[k]))]) for k in range(self.state.nparts)]
        self.chist = [np.empty([0,1]) for k in range(self.state.nparts)]
        self.ahist = np.empty([0,ACTION_SIZE])

        #Set of all GPs we have
        self.XGP = [None for k in range(self.state.nparts)]
        self.CGP = [None for k in range(self.state.nparts)]

    """
    Returns the next action that we should take and update the action history
    """
    def step(self):
        #Do we explore?
        if np.random.uniform(0,1) < self.e**self.t:
            t+= 1
            act = Application.explore(self.dxhist,self.chist,self.ahist)
            self.ahist = np.append([act],self.ahist,axis=0)
            return act

        #Use the subsolver to solve the simulated environment
        t += 1
        act = Application.subsolver(self.sample)
        self.ahist = np.append(self.ahist,[act],axis=0)
        return act

    """
    Updates our GP model using new state transition

    parts - new state data
    """
    def update(self, parts):
        #Update history
        x_old = self.state.x
        c_old = self.state.c
        self.state.transition(parts)
        x_new = self.state.x
        c_new = self.state.c

        dX = [x_new[k]-x_old[k] for k in range(self.state.nparts)]
        dc = [c_new[k]-c_old[k] for k in range(self.state.nparts)]
        
        self.dxhist = [np.append(self.dxhist[k],[dX[k]],axis=0) for k in range(self.state.nparts)]
        self.chist = [np.append(self.chist[k],[dc[k]],axis=0) for k in range(self.state.nparts)]

        #Convert difference data to bounded R^n
        bounded = [np.array([space.DB(dxhist[i][j]) for j in range(dxhist[i].shape[0])]) for i in range(self.state.nparts)]

        #Construct the GPs
        self.XGP = [GPy.models.GPRegression(self.ahist,bounded[i]) for i in range(self.state.nparts)]
        self.CGP = [GPy.models.GPRegression(self.ahist,self.chist) for i in range(self.state.nparts)]

        #Update transition model
        for k in range(self.state.nparts):
            
            #Set X kernel hyperparameters to HMC output
            hmcX = GPy.inference.mcmc.HMC(self.XGP[k])
            sX = hmcX.sample()
            sX = sX[100:] #Burn in
            self.XGP[k].kern.variance = sX[:,0].mean()
            self.XGP[k].kern.lengthscale = sX[:,1].mean()
            self.XGP[k].likelikhood.variance = sX[:,2].mean()

            #Set C kernel hyperparameters to HMC output
            hmcC = GPy.inference.mcmc.HMC(self.CGP[k])
            sC = hmcC.sample()
            sC = sC[100:] #Burn in
            self.CGP[k].kern.variance = sC[:,0].mean()
            self.CGP[k].kern.lengthscale = sC[:,1].mean()
            self.CGP[k].likelikhood.variance = sC[:,2].mean()

    """
    Sample from our model

    state - input state that we will transition from
    action - the action that we take in <state>
    """
    def sample(state,action):
        #Get estimated state deltas from GPs
        for k in range(state.nparts):
            s = self.XGP[k].posterior_samples_f(np.array([action]))
            bounded = np.array([s[i][0] for i in range(s.shape[0])])
            
            diffX = space.BD(bounded,state.x[k].shape)
            diffC = self.CGP[k].posterior_samples_f(np.array([action]))[0]

            state.reconstruct(state.x[k]+diffX,state.c[k]+diffC,k)

        return state
        
            
        
            
            
        

