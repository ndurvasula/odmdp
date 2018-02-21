#The on demand MDP environment
import numpy as np
import application

class Environment:
    """
    model(state,action) - a user defined function that takes in the current state and action, and returns the next state
    r - the reward function (inputted by user): <r(s, a, s2) = R_a(s,s')> (should be the same as in transition)
    s0 - the initial state

    Place all of the above in application.py
    """
    def __init__(self):
        self.model = application.model
        self.r = application.r
        self.state = application.s0
        self.score = 0
    def transition(action):
        xhist = self.state.xhist
        ahist = self.state.ahist
        chist = self.state.chist

        for k in range(self.state.nparts):
            x,c = self.state.decompose(k)

            xhist[k] = np.append(xhist[k], np.array([x]), axis=0)
            chist[k] = np.append(chist[k], np.array([[c]]), axis=0)
            

        s2 = self.model(self.state,action)
        s2.xhist = xhist
        s2.chist = chist
        s2.ahist = ahist

        self.state = s2
        self.score += self.r(self.state, action)
