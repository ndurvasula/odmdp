#Modified from https://github.com/keon/deep-q-learning/blob/master/ddqn.py

import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import application #place model(state,action), r(state,action), and s0 in application.py
from environment import *
from transition import *

EPISODES = 5000

TESTS = 1

class DQNAgent:
    def __init__(self, state_size, action_size, buildModel):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 1    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    for iteration in range(TESTS):
        env = Environment()
        state_size = env.state.size()
        agent = DQNAgent(state_size, action_size)
        batch_size = 32

        while not env.done:
        
            for e in range(EPISODES):
                oenv = OnDemandEnvironment(env.state)
                state = np.reshape(oenv.state, [1, state_size])
                while not oenv.done:
                    action = agent.act(state)
                    reward = oenv.transition(action)
                    next_state = np.reshape(oenv.state, [1, state_size])
                    agent.remember(state, action, reward, next_state, oenv.done)
                    state = next_state
                    if oenv.done:
                        agent.update_target_model()
                        print("episode: {}/{}, score: {}, e: {:.2}"
                              .format(e, EPISODES, time, agent.epsilon))
                        break
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)

        action = agent.act(env.state)
        env.transition(action)
            
