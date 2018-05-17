import gym, gym_fish
import sys
import solver, state, space
import numpy as np
import pickle

def parts(obs):
    arr = []
    for i in range(len(obs)):
        for j in range(obs[i]):
            arr.append([i])
    return np.array(arr)

typ,rand,sbd,d = sys.argv[1:]
dname = str(typ)+"_"+str(rand)+"_"+str(sbd)+"_"+str(d)
pickle.dump(([[[i for i in range(typ)]]],[[typ]]), open(dname+".bounds","wb"))
pickle.dump([],open(dname+"true.state","wb"))
pickle.dump([],open(dname+"true.reward","wb"))

MEANS = np.array([i*1.0/(typ-1) for i in range(typ)])
if rand:
    MEANS = np.array([np.random.uniform() for i in range(typ)])

pickle.dump((typ,MEANS), open(dname+"subsolve.dat","wb"))

env = gym.make('fish-v0')
env.initialize(types=typ,sbdepth=sbd,days=d,means=MEANS)

reward = 0
s0 = state.State([parts(env.reset())])
sol = solver.Solver(s0,.95,1.02,dname)
for i in range(365):
    a = sol.step()
    s, r, done, _ = env.step(a)
    sol.update([parts(s)])
    reward += r
    #input("Continue:")
    S = pickle.load(open(dname+"true.state","rb"))
    R= pickle.load(open(dname+"true.reward","rb"))
    S.append(s)
    R.append(r)
    pickle.dump(S,open(dname+"true.state","wb"))
    pickle.dump(R,open(dname+"true.reward","wb"))
    
    print("ACTUAL FISH:",s)
    print("ACTUAL REWARD",r)
 
print("FINAL REWARD: ",reward)
pickle.dump(reward,open(dname+".final","wb"))



