from transition import *
import numpy as np

class OnDemandEnvironment:
    """
    s0 - the initial state
    """
    def __init__(self,state_0):
        self.state = state_0
    def transition(self,action):
        s2 = model(self.state,action)

        self.state = s2

class Environment:
    """
    s0 - the initial state
    """
    def __init__(self,s0):
        self.state = s0
    def transition(new_state):
        #Once the true environment gives us a new state (with just <parts>), we have to update the rest of the state
        xhist = self.state.xhist
        ahist = self.state.ahist
        chist = self.state.chist

        for k in range(self.state.nparts):
            x = self.state.x[k]
            c = self.c[k]


            xhist[k] = np.append(xhist[k], np.array([x]), axis=0)
            chist[k] = np.append(chist[k], np.array([[c]]), axis=0)
            

        s2 = new_state
        s2.xhist = xhist
        s2.chist = chist
        s2.ahist = ahist

        self.state = s2