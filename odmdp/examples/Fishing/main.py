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

def experiment(exp):
    if exp <= 240:
        typ = 5
    elif exp <= 480:
        typ = 10
        exp -= 240
    else:
        typ = 2
        exp -= 480

    if exp <= 120:
        rand = False
    else:
        rand = True
        exp -= 120

    if exp <= 60:
        sbd = False
    else:
        sbd = True
        exp -= 60

    if exp <= 20:
        d = 75
    elif exp <= 40:
        d = 365
        exp -= 20
    else:
        d = 20
        exp -= 40

    return typ, rand, sbd, d, exp

exp = sys.argv[1]
typ,rand,sbd,d,tnumber = experiment(exp)


dname = str(typ)+"_"+str(rand)+"_"+str(sbd)+"_"+str(d)+"_"+str(tnumber)
pickle.dump(([[[i for i in range(typ)]]],[[typ]]), open(dname+".bounds","wb"))
pickle.dump([],open(dname+"true.state","wb"))
pickle.dump([],open(dname+"true.reward","wb"))

MEANS = np.array([i*1.0/(typ-1) for i in range(typ)])
if rand:
    MEANS = np.array([np.random.uniform() for i in range(typ)])

pickle.dump((typ,MEANS,d), open(dname+"subsolve.dat","wb"))

env = gym.make('fish-v0')
env.initialize(types=typ,sbdepth=sbd,days=d,means=MEANS)

reward = 0
s0 = state.State([parts(env.reset())],dname)
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



