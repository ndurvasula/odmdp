print("Importing...")
import tensorflow as tf
import numpy as np
from state import *
from scipy.stats import norm
import pickle
print("Starting...")

class QNetwork(object):
    """
    Class representing an implementation of a Q-Learning method using Neural Networks
    """

    def __init__(self, in_dimension, out_dimension, discount_factor=0.99, start_epsilon=0.1,
                 decay_rate=0.91, decay_step=10, learning_rate=0.1):
        self.discount_factor = discount_factor
        self.start_epsilon = start_epsilon
        self.learning_rate = learning_rate

        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
        self.cur_epsilon = start_epsilon

        # Tensorflow Objects
        self.Q_out = None
        self.prediction_op = None
        self.train_op = None
        self.next_Q = None
        self.states = None
        self.weights = None

    def create_network_graph(self):
        """
        Creates tensorflow computational graph
        :return:
        """
        tf.reset_default_graph()
        # These lines establish the feed-forward part of the network used to choose actions
        self.states = tf.placeholder(shape=[1, self.in_dimension], dtype=tf.float32)
        self.weights = tf.get_variable("weights", shape=[self.in_dimension, self.out_dimension],
                                       initializer=tf.contrib.layers.xavier_initializer())
        # Probability of each action
        self.Q_out = tf.matmul(self.states, self.weights)

        # We choose the one having the highest outcome
        self.prediction_op = tf.argmax(self.Q_out, 1)[0]

        self.next_Q = tf.placeholder(shape=[1, self.out_dimension], dtype=tf.float32)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        loss = tf.reduce_sum(tf.square(self.next_Q - self.Q_out))

        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)
        # self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        return self.states, self.weights, self.Q_out, self.next_Q, self.train_op, self.prediction_op

    def get_current_epsilon(self):
        return self.cur_epsilon

    def end_episode(self, current_episode):
        # Reduce chance of random action as we train the model.
        # TODO change this by a proper decay
        """
        if self.cur_epsilon > self.min_epsilon and self.cur_epsilon - self.stepping_epsilon > self.min_epsilon:
            self.cur_epsilon -= self.stepping_epsilon
        """
        #print((current_episode / self.decay_step))
        self.cur_epsilon *= self.decay_rate ** (current_episode / self.decay_step)

    def train(self, session, observation, targetQ):
        """
        Update model:
        Train the network after adding information corresponding to the observation
        :param session: Tensorflow session
        :param observation: gym observation
        :param targetQ: Score of each action available for this observation with updated value from the previously choosen action
        :return:
        """

        _, _ = session.run([self.train_op, self.weights],
                           feed_dict={self.states: np.identity(self.in_dimension)[observation:observation + 1],self.next_Q: targetQ})

#Fish pricing
BASE = 1 #Worst possible fish price
MAX = 3 #Best possible fish price
PERIOD = 365.0 #Time for pricing cycle to repeat itself
K = .2 #Spread factor
TYPES = 5
DEBUG = True

#Fish locations
MEANS = np.array([i*1.0/(TYPES-1) for i in range(TYPES)])
STDS = np.array([1.0/(6*(TYPES-1)) for i in range(TYPES)])

def prices(day):
    inp = .5 - .5*np.cos((2*np.pi)*day/PERIOD)
    raw = np.array([norm.cdf(inp+K, loc=MEANS[i], scale=STDS[i]) - norm.cdf(inp-K, loc=MEANS[i], scale=STDS[i]) for i in range(TYPES)])
    return BASE + (MAX-BASE)*raw/np.sum(raw)

def _reward(fish,day):
    p = prices(day)
    return sum([p[i]*fish[i] for i in range(TYPES)])

def conv(parts):
    arr = [0 for i in range(TYPES)]
    for i in parts:
        arr[int(i[0])] += 1
    return np.array(arr)

def solve(s0, t0, iters, transition, dname):
    global TYPES, MEANS
    TYPES, MEANS = pickle.load(open(dname+"subsolve.dat","rb"))

    pickle.dump([],open(dname+"pred.state", "wb"))
    pickle.dump([],open(dname+"pred.reward", "wb"))
    
    MAX_EPISODES = 200
    MAX_STEPS = 365 #How many days we fish for
    DELTA=100 #Action space subdivision
    QN = QNetwork(in_dimension=MAX_STEPS,out_dimension=DELTA,discount_factor=.99,start_epsilon=1,decay_rate=.99,decay_step=100,learning_rate=.075)
    QN.create_network_graph()

    C_Reward = 0
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for i in range(MAX_EPISODES):
            if DEBUG:
                print("Episode",i)
        
            obs = t0
            parts0 = np.array([np.copy(s0.parts[i]) for i in range(s0.nparts)])
            current = State(parts0)

            for step in range(0,MAX_STEPS):
                pred_a, allQ = sess.run([QN.prediction_op, QN.Q_out],
                                        feed_dict={QN.states:np.identity(QN.in_dimension)[obs:obs+1]})
                if np.random.rand(1) < QN.get_current_epsilon():
                    pred_a = np.random.randint(DELTA)

                n_obs = obs+1
                current = transition(current,np.array([pred_a*1.0/DELTA]))
                reward = _reward(conv(current.parts[0]), obs)

                all_next = sess.run(QN.Q_out, feed_dict={
                    QN.states: np.identity(QN.in_dimension)[n_obs:n_obs+1]})

                best = np.max(all_next)
                tQ = allQ

                tQ[0, pred_a] = reward + QN.discount_factor * best
                QN.train(session=sess,observation=obs,targetQ=tQ)

                obs = n_obs
            
                C_Reward += reward

                if obs == MAX_STEPS-1:
                    QN.end_episode(current_episode=obs)
                    break
            if DEBUG:
                print("Episode reward:",C_Reward)
                
            C_Reward=0

        #Return the string of actions
        actions = []
        for step in range(0,int(min(iters,MAX_STEPS-t0))):
            pred_a, allQ = sess.run([QN.prediction_op, QN.Q_out],
                                        feed_dict={QN.states:np.identity(QN.in_dimension)[t0+step:t0+step+1]})
            actions.append(np.array([pred_a*1.0/DELTA]))
            current = transition(current,np.array([pred_a*1.0/DELTA]))

            if step == 0:
                reward = _reward(conv(current.parts[0]), t0)
                fi = conv(current.parts[0])
        S = pickle.load(open(dname+"pred.state", "rb"))
        R = pickle.load(open(dname+"pred.reward", "rb"))
        S.append(fi)
        R.append(reward)
        pickle.dump(S,open(dname+"pred.state", "wb"))
        pickle.dump(R,open(dname+"pred.reward", "wb"))
        if DEBUG:
            print("Predicted fish:",fi)
            print("Predicted reward:",reward)

        return actions, C_Reward


    
