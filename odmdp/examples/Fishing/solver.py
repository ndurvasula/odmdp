#On demand environment solver

import numpy as np
import GPy
import space
import application
from state import State
import pylab as pb
import pickle
import os, shutil

CHANGE = False
X_STOCHASTIC = False
C_STOCHASTIC = True
LOG = True
LOG_STEP = 5
HMC = False
HMC_SAMPLES = 300
DEBUG = True
PLOT = True
DNAME = ""

switch = True
remaining = 0

class Solver():
    """
    s0 - the initial state
    e - the exploration constant, at time t, the probability that we explore instead of subsolve is e^t (t starts at 0)
    trust - the rate at which we trust the model
    """
    def __init__(self,s0,e,trust,dname):
        global DNAME
        DNAME = dname

        application.init(DNAME)

        if LOG:
            if os.path.exists(DNAME+"logs/"):
                shutil.rmtree(DNAME+"logs/")
                
            os.makedirs(DNAME+"logs/")
            pickle.dump([],open(DNAME+"logs/xgp.bin",'wb'))
            pickle.dump([],open(DNAME+"logs/cgp.bin","wb"))
        
        self.state = s0
        self.e = e
        self.t = 0
        self.trust = trust
        
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
        global switch, remaining
        
        #Do we explore?
        if switch and np.random.uniform(0,1) < self.e**self.t:
            if DEBUG:
                print("Explored on time",self.t)
                
            self.t+= 1
            act = application.explore(self.dxhist,self.chist,self.ahist)
            self.ahist = np.append(self.ahist,[act],axis=0)
            return act

        #Use the subsolver to solve the simulated environment
        if remaining == 0:
            remaining = np.floor(self.trust**self.t)
            switch = False
            if DEBUG:
                print("Subsolved on time",self.t,"for",remaining,"iterations")
            
        remaining -= 1

        if remaining == 0:
            switch = True
            
        act = application.subsolver(self.state,self.t,np.floor(self.trust**self.t),self.sample)
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
            
        if CHANGE:
            dX = [x_new[k]-x_old[k] for k in range(self.state.nparts)]
            dc = [c_new[k]-c_old[k] for k in range(self.state.nparts)]

        else:
            dState = State([np.array([]) for i in range(self.state.nparts)])
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

        if DEBUG:
            print("Updating model")

        #Update transition model
        for k in range(self.state.nparts):
            if DEBUG:
                print("Tuning X GPs")
                
            #Set X kernel hyperparameters to HMC output
            
            for i in range(len(self.XGP[k])):
                if not HMC:
                    self.XGP[k][i].optimize()

                else:
                    hmcX = GPy.inference.mcmc.HMC(self.XGP[k][i])
                    sX = hmcX.sample(num_samples=HMC_SAMPLES)
                    sX = sX[100:] #Burn in
                    self.XGP[k][i].kern.variance = sX[:,0].mean()
                    self.XGP[k][i].kern.lengthscale = sX[:,1].mean()
                    self.XGP[k][i].likelihood.variance = sX[:,2].mean()

            if DEBUG:
                print("Tuning C GP")
                
            #Set C kernel hyperparameters to HMC output

            if HMC:
                hmcC = GPy.inference.mcmc.HMC(self.CGP[k])
                sC = hmcC.sample(num_samples=300)
                sC = sC[100:] #Burn in
                self.CGP[k].kern.variance = sC[:,0].mean()
                self.CGP[k].kern.lengthscale = sC[:,1].mean()
                self.CGP[k].likelihood.variance = sC[:,2].mean()
            
            else:
                self.CGP[k].optimize()

        if LOG and self.t % LOG_STEP == 0:
            Xs = pickle.load(open(DNAME+"logs/XGP.bin","rb"))
            Cs = pickle.load(open(DNAME+"logs/CGP.bin","rb"))

            Xs.append(self.XGP)
            Cs.append(self.CGP)
            
            pickle.dump(Xs,open(DNAME+"logs/XGP.bin",'wb'))
            pickle.dump(Cs,open(DNAME+"logs/CGP.bin","wb"))

            if PLOT:
                if not os.path.exists(DNAME+"logs/plots/XGP"):
                    os.makedirs(DNAME+"logs/plots/XGP")
                if not os.path.exists(DNAME+"logs/plots/CGP"):
                    os.makedirs(DNAME+"logs/plots/CGP")

                for k in range(self.state.nparts):
                    if not os.path.exists(DNAME+"logs/plots/XGP/Partition_"+str(k)):
                        os.makedirs(DNAME+"logs/plots/XGP/Partition_"+str(k))
                    if not os.path.exists(DNAME+"logs/plots/CGP/Partition_"+str(k)):
                        os.makedirs(DNAME+"logs/plots/CGP/Partition_"+str(k))

                    cp = self.CGP[k].plot().figure
                    cp.savefig(DNAME+"logs/plots/CGP/Partition_"+str(k)+"/t="+str(self.t)+".png")
                    pb.close(cp)

                    for i in range(len(self.XGP[k])):
                        if not os.path.exists(DNAME+"logs/plots/XGP/Partition_"+str(k)+"/Variable_"+str(i)):
                            os.makedirs(DNAME+"logs/plots/XGP/Partition_"+str(k)+"/Variable_"+str(i))

                        xp = self.XGP[k][i].plot().figure
                        xp.savefig(DNAME+"logs/plots/XGP/Partition_"+str(k)+"/Variable_"+str(i)+"/t="+str(self.t)+".png")
                        pb.close(xp)


    """
    Sample from our model

    state - input state that we will transition from
    action - the action that we take in <state>
    """
    def sample(self, state, action):
        #Get estimated state deltas from GPs
        for k in range(state.nparts):
            #Get data from GPs and ensure that it falls in bounded space

            if X_STOCHASTIC:
                bounded = np.array([self.XGP[k][i].posterior_samples_f(np.array([action]),size=1)[0][0] for i in range(len(self.XGP[k]))])

            else:
                bounded = np.array([self.XGP[k][i].predict(np.array([action]))[0][0][0] for i in range(len(self.XGP[k]))])

            bounded[bounded>1] = 1
            bounded[bounded<-1] = -1
            
            diffX = space.BD(bounded,state.x[k].shape)

            if C_STOCHASTIC:
                diffC = self.CGP[k].posterior_samples_f(np.array([action]))[0][0]
            else:
                diffC = self.CGP[k].predict(np.array([action]))[0][0][0]

            if CHANGE:
                state.reconstruct(state.x[k]+diffX,state.c[k]+diffC,k)

            else:
                dState = State([np.array([]) for i in range(self.state.nparts)])
                state.reconstruct(np.array(dState.x[k]+diffX),dState.c[k]+diffC,k)

        return state
    
