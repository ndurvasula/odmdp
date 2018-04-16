import gym, gym_fish
import solver, state
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
sol = solver.Solver(s0,.9)
for i in range(365):
    a = sol.step()
    s, r, done, _ = env.step(a)
    sol.update([parts(s)])
    reward += r
    print("ACTUAL REWARD",r)
print("FINAL REWARD: ",reward)


