import gym, gym_fish
import solver, state, space
import numpy as np

def log():
    import GPy, pickle
    import pylab as pb
    XGP = pickle.load(open("xgp.bin","rb"))
    for i in XGP:
        i.plot()
    CGP = pickle.load(open("cgp.bin","rb"))
    CGP.plot()

def parts(obs):
    arr = []
    for i in range(len(obs)):
        for j in range(obs[i]):
            arr.append([i])
    return np.array(arr)


env = gym.make('fish-v0')
reward = 0
s0 = state.State([parts(env.reset())])
sol = solver.Solver(s0,.99)
for i in range(365):
    a = sol.step()
    s, r, done, _ = env.step(a)
    sol.update([parts(s)])
    reward += r
    #input("Continue:")
    print("ACTUAL FISH:",s)
    print("ACTUAL REWARD",r)
 
print("FINAL REWARD: ",reward)



