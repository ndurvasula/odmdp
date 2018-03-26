import numpy as np

class State():
    """
    nparts - number of partitions
    dparts - dimension of partition (array of length nparts e.g. [2,1,3,4,5]), note that elements of a partition must be 1d vectors
    bounds - a multidimensional array denoting the set of values that the partitioned part of the state can take on
             for example, if we have a state space with dparts = [1,2,1], we could have bounds = [[[1,2,3]],[[.1,.2],[3,4,5,6]],[["a","b","c"]]]
    shape - a multidimensional array denoting the number of values that the partioned part of the state can take on
            if we take the above example for <bounds>, we get shape = [[3],[2,4],[3]]
    parts - the current state as an array of partitions (e.g. [np.array(~),np.array(~),...])
    """
    def __init__(self, nparts, dparts, bounds, shape, parts):
        self.nparts = nparts
        self.dparts = dparts
        self.parts = parts
        self.sh = shape
        self.bounds = bounds

        self.x = []
        self.c = []
        for k in range(nparts):
            x_k,c_k = self.decompose(k)
            self.x.append(x_k)
            self.c.append(c_k)


    """
    Genrates the joint probability mass tensor and size for the kth partition
    k - the partition number
    """
    def decompose(self,k):
        jpmf = np.zeros(self.sh[k])
        objs = []
        freq = []
        diff = 0

        for i in self.parts[k]:
            if any((i == x).all() for x in objs):
                freq[objs.index(i)] += 1
            else:
                diff += 1
                objs.append(i)
                freq.append(1)

        c = len(self.parts[k])
        freq = [x/c for x in freq]
        inds = self.convert(objs,k)
        for i in range(diff):
            jpmf[tuple(inds[i])] = freq[i]

        return jpmf, c

    """
    Reconstructs the state given a joint probability mass tensor and size back into original vector space
    k - the partition number
    """
    def reconstruct(self,jpmf, c, k):
        self.x[k] = jpmf
        self.c[k] = c
        self.parts[k] = np.empty([0,self.dparts[k]])
        for index, x in np.ndenumerate(jpmf):
            if round(x*c) > 0:
                obj = []
                for i in range(self.dparts[k]):
                    obj.append(self.bounds[k][i][index[i]])
                for i in range(int(np.round(x*c))):
                    self.parts[k] = np.append(self.parts[k],np.array([obj]),axis=0)

    """
    Converts objects in the kth partition from their original format to Z^n based on <bounds>
    """
    def convert(self,objs,k):
        ret = []
        for i in objs:
            next = []
            for j in range(self.dparts[k]):
                next.append(self.bounds[k][j].index(i[j]))
            ret.append(next)
        return ret

    """
    Returns the state size
    """
    def size(self):
        size = 0
        for k in range(self.nparts):
            size+=np.prod(self.parts[k].shape)

        return size

    """
    Changes state data
    """
    def transition(parts):
        self.parts = parts

        self.x = []
        self.c = []
        for k in range(self.nparts):
            x_k,c_k = self.decompose(k)
            self.x.append(x_k)
            self.c.append(c_k)





