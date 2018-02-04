import numpy as np

class State():
    """
    nparts - number of partitions
    dparts - dimension of partition (array of length nparts e.g. [2,1,3,4,5])
    P - joint probability distribution function over all inaccesible information
    bounds - a multidimensional array denoting the set of values that the partitioned part of the state can take on
             for example, if we have a state space with dparts = [1,2,1], we could have bounds = [[[1,2,3]],[[.1,.2],[3,4,5,6]],[["a","b","c"]]]
    shape - a multidimensional array denoting the number of values that the partioned part of the state can take on
            if we take the above example for <bounds>, we get shape = [[3],[2,4],[3]]
    """
    def __init__(self, nparts, dparts, dist, bounds):
        self.n = nparts
        self.d = dparts
        self.P = dist
        self.parts = None
        self.info = None

    """
    parts - the current state as an array of partitions (e.g. [np.array(~),np.array(~),...])
    info - any other information present within the state - can be in any format as long as it is accepted by <P>
    """
    def setState(self, parts, info):
        self.parts =  parts
        self.info = info

    """
    Genrates the joint probability mass tensor and size for the kth partition
    k - the partition number
    """
    def decompose(k):
        jpmf = np.zeros(sh[k])
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

        freq = [x/len(parts[k]) for x in freq]
        inds = convert(objs,k)
        for i in range(diff):
            jpmf[tuple(inds[i])] = freq[i]

        return jpmf, len(parts[k])

    """
    Converts objects in the kth partition from their original format to Z^n based on <bounds>
    """
    def convert(objs,k):
        ret = []
        for i in objs:
            next = []
            for j in range(self.d[k]):
                next.append(bounds[k][j].index(objs[j]))
            ret.append(next)
        return ret





