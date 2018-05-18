import gym, gym_fish
import sys
import solver, state, space
import numpy as np
import pickle
import multiprocessing

def parts(obs):
    arr = []
    for i in range(len(obs)):
        for j in range(obs[i]):
            arr.append([i])
    return np.array(arr)

def run(typ,rand,sbd,d,tnumber):

    try:

        dname = str(typ)+"_"+str(rand)+"_"+str(sbd)+"_"+str(d)+"_"+str(tnumber)
        pickle.dump(([[[i for i in range(typ)]]],[[typ]]), open(dname+".bounds","wb"))
        pickle.dump([],open(dname+"true.state","wb"))
        pickle.dump([],open(dname+"true.reward","wb"))

        MEANS = np.array([i*1.0/(typ-1) for i in range(typ)])
        if rand:
            MEANS = np.array([np.random.uniform() for i in range(typ)])

        pickle.dump((typ,MEANS,d), open(dname+"subsolve.dat","wb"))

        env = gym.make('fish-v0')
        env.initialize(types=typ,sbdepth=sbd,days=d,means=MEANS,discretize=False)

        reward = 0
        s0 = state.State([parts(env.reset())],dname)
        sol = solver.Solver(s0,.95,1.02,dname)
        for i in range(d):
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

    except Exception as e:
        pickle.dump(str(e), open(dname+".error","wb"))

def experiment(exp):
    if exp <= 15:
        typ = 5
    elif exp <= 30:
        typ = 10
        exp -= 15
    else:
        typ = 2
        exp -= 30
        
    if exp <= 5:
        d = 75
    elif exp <= 10:
        d = 365
        exp -= 5
    else:
        d = 20
        exp -= 10

    return typ, d, exp

if __name__ == '__main__':

    exp = int(sys.argv[1])
    typ,d,tp = experiment(exp)
    jobs = []
    for rand in range(2):
        for sbd in range(2):
            for tnumber in range(4*(tp-1)+1, 4*(tp-1)+5):
                thr = multiprocessing.Process(target=run, args = (typ,rand,sbd,d,tnumber,))
                jobs.append(thr)
                thr.start()





