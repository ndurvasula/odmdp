import gym, gym_fish
import sys, os
import solver, state, space
import numpy as np
import pickle
#import multiprocessing
import traceback

def parts(obs):
    arr = []
    for i in range(len(obs)):
        for j in range(obs[i]):
            arr.append([i])
    return np.array(arr)

def run(typ,rand,sbd,d,tnumber):

    try:

        dname = str(typ)+"_"+str(rand)+"_"+str(sbd)+"_"+str(d)+"_"+str(tnumber)
        pickle.dump("Success",open(dname,"wb"))
        pickle.dump(([[[i for i in range(typ)]]],[[typ]]), open(dname+".bounds","wb"),-1)
        pickle.dump([],open(dname+"true.state","wb"),-1)
        pickle.dump([],open(dname+"true.reward","wb"),-1)

        MEANS = np.array([i*1.0/(typ-1) for i in range(typ)])
        STDS = np.array([1.0/(6*(typ-1)) for i in range(typ)])
        if rand:
            MEANS = np.array([np.random.uniform() for i in range(typ)])

        pickle.dump((typ,MEANS,STDS,d), open(dname+"subsolve.dat","wb"),-1)
        

        env = gym.make('fish-v0')
        env.initialize(types=typ,sbdepth=sbd,days=d,means=MEANS,discretize=False,stds=STDS)

        pickle.dump([],open(dname+"true.state","wb"),-1)
        pickle.dump([],open(dname+"true.reward","wb"),-1)
        
        reward = 0
        s0 = state.State([parts(env.reset())],dname)
        sol = solver.Solver(s0,.95,1.02,dname)
        for i in range(d):
            a = sol.step()
            s, r, done, _ = env.step(a)
            sol.update([parts(s)])
            reward += r
            #input("Continue:")
            if os.path.getsize(dname+"true.state") > 0 and os.path.getsize(dname+"true.reward") > 0:
                S = pickle.load(open(dname+"true.state","rb"))
                R= pickle.load(open(dname+"true.reward","rb"))
            else:
                S = []
                R = []
            S.append(s)
            R.append(r)
            pickle.dump(S,open(dname+"true.state","wb"),-1)
            pickle.dump(R,open(dname+"true.reward","wb"),-1)
            
            print("ACTUAL FISH:",s)
            print("ACTUAL REWARD",r)
         
        print("FINAL REWARD: ",reward)
        pickle.dump(reward,open(dname+".final","wb"),-1)

    except:
        print(traceback.print_exc(file=open(dname+".error","w")))

def experiment(exp):
    if exp <= 600:
        typ = 5
    elif exp <= 1200:
        typ = 10
        exp -= 600
    else:
        typ = 2
        exp -= 1200
        
    if exp <= 200:
        d = 75
    elif exp <= 400:
        d = 365
        exp -= 200
    else:
        d = 25
        exp -= 400

    if exp <= 100:
        rand = True
    else:
        rand = False
        exp -= 100

    if exp <= 50:
        sbd = True
    else:
        sbd = False
        exp -= 50

    return typ,rand,sbd,d,exp

if __name__ == '__main__':

    exp = int(sys.argv[1])
    typ,rand,sbd,d,tn = experiment(exp)
    run(typ,rand,sbd,d,tn)
    





