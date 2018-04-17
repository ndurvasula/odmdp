import gym, gym_fish
import solver, state, space
import numpy as np

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



