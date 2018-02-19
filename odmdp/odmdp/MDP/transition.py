#Transition function for on demand MDPs
#Learns the probability model as we progress through the MDP
#The offline environment

import numpy as np

def model(state,action):
    #Learn the probability model using beta learning
    xmodel = xlearn(state, action)
    cmodel = clearn(state, action)

class OnDemandEnvironment:
    """
    r - the reward function (inputted by user): <r(s, a, s2) = R_a(s,s')> (should be the same as in transition)
    s0 - the initial state
    """
    def __init__(self, s0, r):
        self.r = r
        self.state = s0
    def transition(action):
        xhist = self.state.xhist
        ahist = self.state.ahist
        chist = self.state.chist

        xt = []
        ct = []
        for i in range(self.state.nparts):
            x,c = self.state.decompose(i)
            xt.append(x)
            ct.append(c)

        xhist.append(xt)
        chist.append(ct)
        ahist.append(action)

        s2 = model(self.state,action)
        s2.xhist = xhist
        s2.chist = chist
        s2.ahist = ahist

        self.state = s2
        return r(self.state, action)