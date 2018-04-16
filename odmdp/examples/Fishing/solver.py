#On demand environment solver

import numpy as np
import GPy
import space
import application
from state import State
import pylab as pb
import pickle
pb.ion()

class Solver():
    """
    s0 - the initial state
    e - the exploration constant, at time t, the probability that we explore instead of subsolve is e^t (t starts at 0)
    """
    def __init__(self,s0,e):
        self.state = s0
        self.e = e
        self.t = 0
        
        #State delta history and action history in our walk so far for each partition
        self.dxhist = [np.empty([0,np.prod(np.array(self.state.sh[k]))]) for k in range(self.state.nparts)]
        self.chist = [np.empty([0,1]) for k in range(self.state.nparts)]
        self.ahist = np.empty([0,application.ACTION_SIZE])

        #Set of all GPs we have
        self.XGP = [None for k in range(self.state.nparts)]
        self.CGP = [None for k in range(self.state.nparts)]

    """
    Returns the next action that we should take and update the action history
    """
    def step(self):
        #Do we explore?
        if np.random.uniform(0,1) < self.e**self.t:
            print("Explored on time",self.t)
            self.t+= 1
            act = application.explore(self.dxhist,self.chist,self.ahist)
            self.ahist = np.append(self.ahist,[act],axis=0)
            return act

        #Use the subsolver to solve the simulated environment
        print("Subsolved on time",self.t)
        act = application.subsolver(self.state,self.t,self.sample)
        self.t += 1
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

        #dX = [x_new[k]-x_old[k] for k in range(self.state.nparts)]
        #dc = [c_new[k]-c_old[k] for k in range(self.state.nparts)]
        dState = State([np.array([])])
        dX = [self.state.x[k]-dState.x[k] for k in range(self.state.nparts)]
        dc = [self.state.c[k]-dState.c[k] for k in range(self.state.nparts)]
        
        self.dxhist = [np.append(self.dxhist[k],[dX[k]],axis=0) for k in range(self.state.nparts)]
        self.chist = [np.append(self.chist[k],[dc[k]],axis=0) for k in range(self.state.nparts)]

        #Convert difference data to bounded R^n
        bounded = [np.array([space.DB(self.dxhist[k][j]) for j in range(self.dxhist[k].shape[0])]) for k in range(self.state.nparts)]

        #Construct the GPs
        self.XGP = []
        for k in range(self.state.nparts):
            arr = [np.array([[j] for j in bounded[k][:,i]]) for i in range(bounded[k][0].shape[0])]
            add = [GPy.models.GPRegression(self.ahist,i) for i in arr]
            self.XGP.append(add)
            
        self.CGP = [GPy.models.GPRegression(self.ahist,self.chist[i]) for i in range(self.state.nparts)]

        print("Updating model")

        #Update transition model
        for k in range(self.state.nparts):
            print("X HMC")
            #Set X kernel hyperparameters to HMC output

            """
            hmcX = GPy.inference.mcmc.HMC(self.XGP[k])
            sX = hmcX.sample(num_samples=300)
            sX = sX[100:] #Burn in
            self.XGP[k].kern.variance = sX[:,0].mean()
            self.XGP[k].kern.lengthscale = sX[:,1].mean()
            self.XGP[k].likelihood.variance = sX[:,2].mean()
            """
            for i in range(len(self.XGP[k])):
                self.XGP[k][i].optimize()
            
            pickle.dump(self.XGP[k],open("xgp.bin",'wb'))

            print("C HMC")
            #Set C kernel hyperparameters to HMC output

            """
            hmcC = GPy.inference.mcmc.HMC(self.CGP[k])
            sC = hmcC.sample(num_samples=300)
            sC = sC[100:] #Burn in
            self.CGP[k].kern.variance = sC[:,0].mean()
            self.CGP[k].kern.lengthscale = sC[:,1].mean()
            self.CGP[k].likelihood.variance = sC[:,2].mean()
            """
            self.CGP[k].optimize()

            pickle.dump(self.CGP[k],open("cgp.bin","wb"))


    """
    Sample from our model

    state - input state that we will transition from
    action - the action that we take in <state>
    """
    def sample(self, state,action):
        #Get estimated state deltas from GPs
        for k in range(state.nparts):
            #s = self.XGP[k].posterior_samples_f(np.array([action]))

            #Get data from GPs and ensure that it falls in bounded space
            #bounded = np.array([s[i][0][0] for i in range(s.shape[0])])
            bounded = np.array([self.XGP[k][i].predict(np.array([action]))[0][0][0] for i in range(len(self.XGP[k]))])

            bounded[bounded>1] = 1
            bounded[bounded<-1] = -1
            
            diffX = space.BD(bounded,state.x[k].shape)
            diffC = self.CGP[k].posterior_samples_f(np.array([action]))[0][0]

            #state.reconstruct(state.x[k]+diffX,state.c[k]+diffC,k)
            dState = State([np.array([])])
            state.reconstruct(np.array(dState.x[k]+diffX),dState.c[k]+diffC,k)

        return state
    
