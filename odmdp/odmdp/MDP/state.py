import numpy as np

class State():
    """
    nparts - number of partitions
    dparts - dimension of partition (array of length nparts e.g. [2,1,3,4,5])
    P - joint probability density function over all inaccesible information
    bounds - a multidimensional array denoting the set of values that the partitioned part of the state can take on
             for example, if we have a state space with dparts = [1,2,1], we could have bounds = [[[1,2,3]],[[.1,.2],[3,4,5,6]],[["a","b","c"]]]
    shape - a multidimensional array denoting the number of values that the partioned part of the state can take on
            if we take the above example for <bounds>, we get shape = [[3],[2,4],[3]]
    transition - a function that takes in an action and returns the next state <transition(state, action)>
    data - the current state as an array of partitions (e.g. [np.array(~),np.array(~),...])
    """
    def __init__(self, nparts, dparts, P, bounds, shape, transition, data):
        self.n = nparts
        self.d = dparts
        self.P = P
        self.parts = data
        self.sh = shape
        self.bounds = bounds
        
        #State-action history in our walk so far for each partition
        self.xhist = []
        self.chist = []
        self.ahist = []

    """
    Genrates the joint probability mass tensor and size for the kth partition
    k - the partition number
    """
    def decompose(k):
        jpmf = np.zeros(self.sh[k])
        objs = []
        freq = []
        diff = 0

        for i in parts[k]:
            if i in objs:
                freq[objs.index(i)] += 1
            else:
                diff += 1
                objs.append(i)
                freq.append(1)

        freq = [x/len(self.parts[k]) for x in freq]
        inds = convert(objs,k)
        for i in range(diff):
            jpmf[tuple(inds[i])] = freq[i]

        return jpmf, self.d[k]

    """
    Reconstructs the state given a joint probability mass tensor and size
    k - the partition number
    """
    def reconstruct(jpmf, c, k):
        self.parts[k] = np.array([])
        for index, x in np.ndenumerate(jpmf):
            if round(x*c) > 0:
                obj = []
                for i in self.d[k]:
                    obj.append(self.bounds[index[i]])
                for i in range(round(x*c)):
                    self.parts[k] = np.append(self.parts[k],np.array(obj))

    """
    Converts objects in the kth partition from their original format to Z^n based on <bounds>
    """
    def convert(objs,k):
        ret = []
        for i in objs:
            next = []
            for j in range(self.d[k]):
                next.append(self.bounds[k][j].index(objs[j]))
            ret.append(next)
        return ret

    """
    Returns the state size
    """
    def size():
        size = 1
        for k in range(nparts):
            for p in sh[k]:
                size *= p

        return size





