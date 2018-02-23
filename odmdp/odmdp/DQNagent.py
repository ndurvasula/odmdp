import random
import tensorflow as tf
import gym
import numpy as np
from state import *
import math
from odmdp import *
print("Finished imports")

def parts(obs):
    return [np.array([obs % 4, math.floor(obs/4.0)])]

def obs(state):
    row = state.parts[0][0]
    col = state.parts[0][1]

    #Make sure we didn't accidentally move past the grid
    if row < 0:
        row = 0
    if row > 3:
        row = 3
    if col < 0:
        col = 0
    if col > 3:
        col = 3

    state.parts[0] = np.array([row,col])
    state.x[0], state.c[0] = state.decompose(0)

    ret = 4*row + col

    return ret, 1 if ret == 15 else 0, ret == 5 or ret == 7 or ret == 11 or ret == 12 or ret == 15

def OnDemandDQNSubSolver(trial,nparts,dparts,bounds,shape,parts,aspace):
    tf.reset_default_graph()
    inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)
    W = tf.Variable(tf.random_uniform([16,4],0,0.01))
    Qout = tf.matmul(inputs1,W)
    predict = tf.argmax(Qout,1)
    nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)

    loss = tf.reduce_sum(tf.square(nextQ-Qout))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    updateModel = trainer.minimize(loss)

    init = tf.global_variables_initializer()

    gamma = .99
    e = 0.1
    num_episodes = 1000
    jList = []
    rList = []

    with tf.Session() as sess:
        sess.run(init)

        for i in range(num_episodes):
            st = State(nparts,dparts,bounds,shape,parts)
            oEnv = OnDemandEnvironment(st)
            s,_,__,___ = obs(st)
            rAll = 0
            d = False
            j = 0
            while j < 99:
                j+= 1
                a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})

                if np.random.rand(1) < e:
                    a[0] = aspace.sample()
                s1,r,d,_ = obs(oEnv.transition(a[0]))

                Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0,a[0]] = r + gamma*maxQ1
                _,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})

                rAll += r
                s = s1
                if d == True:
                    e = 1./((i/50)+10)
                    break
            jList.append(j)
            rList.append(rAll)
            print("Trial "+str(trial)+", episode "+str(i)+", reward "+str(rAll)+", time "+str(j), ", average "+str(sum(rList[len(rList)-101:])/100))
        print("Percent of succesful episodes: "+str(sum(rList)/num_episodes) + "%")

        #Now that we've trained, let's output the best action
        st = State(nparts,dparts,bounds,shape,parts)
        oEnv = OnDemandEnvironment(st)
        s,_,__,___ = obs(st)
        a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})

        return a[0]
    

num_trials = 100

nparts = 1
dparts = [2]
bounds = [[[0,1,2,3],[0,1,2,3]]]
sh = [[4,4]]

rList = []
jList = []

for T in range(num_trials):

    print("Trial "+str(T))

    env = gym.make('FrozenLake-v0')
    
    rEnv = Environment(State(nparts,dparts,bounds,shape,parts(env.reset())))
    done = False

    rTotal = 0

    while not done:
        
        #Act based on the simulated transition system and deep Q learning
        action = OnDemandDQNSubSolver(T,rEnv.state.nparts,rEnv.state.dparts,rEnv.state.bounds,rEnv.state.shape,rEnv.state.parts,env.action_space)
        s1, r, done, _ = env.step(action)

        rTotal += r

        #Update our state history
        rEnv.transition(State(rEnv.state.nparts,rEnv.state.dparts,rEnv.state.bounds,rEnv.state.shape,parts(s1)))


    rList.append(rTotal)

print("Percentage correct: "+str(sum(rList)/num_trials))
    

    
