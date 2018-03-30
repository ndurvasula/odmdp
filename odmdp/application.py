"""
application contains all app specific code

ACTION_SIZE - length of array denoting each action (all actions have shape [ACTION_SIZE,])
SIZE_CHANGE - np array of booleans, if SIZE_CHANGE[k] is True, that means that taking an action can change the size of the kth partition
NPARTS - number of partitions
DPARTS - dimension of partition (array of length nparts e.g. [2,1,3,4,5]), note that elements of a partition must be 1d vectors
BOUNDS - a multidimensional array denoting the set of values that the partitioned part of the state can take on
         for example, if we have a state space with dparts = [1,2,1], we could have bounds = [[[1,2,3]],[[.1,.2],[3,4,5,6]],[["a","b","c"]]]
SHAPE - a multidimensional array denoting the number of values that the partioned part of the state can take on
        if we take the above example for <bounds>, we get shape = [[3],[2,4],[3]]

subsolver(transition) - solves the RL problem given that transition(state,action) performs a state transition and returns the *first* action
explore(dxhist,chist,ahist) - generates a random action by user-chosen distribution, may utilize dxhist, chist, ahist
> dxhist - delta x history for all k
> chist - delta c history for all k
> ahist - action history
"""
