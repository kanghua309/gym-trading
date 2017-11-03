# -*- coding: utf-8 -*-

"""
In this example we demonstrate how to implement a DQN agent and
train it to trade optimally on a periodic price signal.
Training time is short and results are unstable.
Do not hesitate to run several times and/or tweak parameters to get better results.
Inspired from https://github.com/keon/deep-q-learning
"""
import random

import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import gym
import gym_trading  #必须引入才自动注册
import logging

log = logging.getLogger(__name__)
logging.basicConfig()
log.setLevel(logging.INFO)
log.info('%s logger started.',__name__)
class DQNAgent:
    def __init__(self,
                 state_size,
                 action_size,
                 episodes,
                 episode_length,
                 memory_size=2000,
                 train_interval=100,
                 gamma=0.95,
                 learning_rate=0.00001,
                 batch_size=64,# 64 TODO
                 epsilon_min=0.01
                 ):
        self.state_size = state_size
        self.action_size = action_size
        self.memory_size = memory_size
        self.memory = [None] * memory_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decrement = (self.epsilon - epsilon_min)\
            * train_interval / (episodes * episode_length)  # linear decrease rate
        self.learning_rate = learning_rate
        self.train_interval = train_interval
        self.batch_size = batch_size
        self.brain = self._build_brain()
        self.i = 0

    def _build_brain(self):
        """Build the agent's brain
        """
        brain = Sequential()
        neurons_per_layer = 24
        activation = "relu"
        brain.add(Dense(neurons_per_layer,
                        input_dim=self.state_size,
                        activation=activation))
        brain.add(Dense(neurons_per_layer, activation=activation))
        brain.add(Dense(self.action_size, activation='linear'))
        brain.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return brain
    '''
    def act(self, state):
        """Acting Policy of the DQNAgent
        """
        action = np.zeros(self.action_size)
        print "action:",action
        if np.random.rand() <= self.epsilon:
            action[random.randrange(self.action_size)] = 1
        else:
            state = state.reshape(1, self.state_size)
            act_values = self.brain.predict(state)
            action[np.argmax(act_values[0])] = 1
        return action
    '''
    def act(self, state):
        """Acting Policy of the DQNAgent
        """
        #action = np.zeros(self.action_size)
        #print "action:", action
        if np.random.rand() <= self.epsilon:
            #action[random.randrange(self.action_size)] = 1
            action = random.randrange(self.action_size)
        else:
            state = state.reshape(1, self.state_size)
            act_values = self.brain.predict(state)
            #action[np.argmax(act_values[0])] = 1
            #print "x:",np.argmax(act_values[0])
            action = np.argmax(act_values[0])
            #print "------------------------ qval:", np.argmax(act_values),act_values[0],state

        #print "action:", action
        return action

    def observe(self, state, action, reward, next_state, done, warming_up=False):
        """Memory Management and training of the agent
        """
        #print "0 DEBuG .....:", self.i

        self.i = (self.i + 1) % self.memory_size
        self.memory[self.i] = (state, action, reward, next_state, done)
        if (not warming_up) and (self.i % self.train_interval) == 0:
            #print "1 DEBuG .....:",self.i
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decrement
            state, action, reward, next_state, done = self._get_batches()
            reward += (self.gamma
                       * np.logical_not(done)
                       * np.amax(self.brain.predict(next_state),
                                 axis=1))
            q_target = self.brain.predict(state)
            #print "state :",state,np.shape(state)
            #print "reward :",reward,np.shape(reward)
            #print "action :",action,np.shape(action)
            #print "q_target :",q_target,np.shape(q_target)
            #q_target[action[0], action[1]] = reward
            #where?
            _ = pd.Series(action)
            one_hot = pd.get_dummies(_).as_matrix()
            action_batch = np.where(one_hot == 1)
            #print "batch:",action_batch
            q_target[action_batch] = reward
            #print "----q target:",reward
            return self.brain.fit(state, q_target,
                                  batch_size=self.batch_size,
                                  epochs=1,
                                  verbose=False)

    def _get_batches(self):
        """Selecting a batch of memory
           Split it into categorical subbatches
           Process action_batch into a position vector
        """
        batch = np.array(random.sample(self.memory, self.batch_size))
        #print "batch :",batch
        #print "---------------------",self.batch_size,self.state_size
        #state, action, reward, next_state, done
        #print batch[:, 0]
        #print batch[:, 1]
        #print batch[:, 2]
        #print batch[:, 3]
        #print batch[:, 4]
        state_batch = np.concatenate(batch[:, 0])\
            .reshape(self.batch_size, self.state_size)
        #action_batch = np.concatenate(batch[:, 1])\
        #    .reshape(self.batch_size, self.action_size)
        action_batch = batch[:, 1]

        reward_batch = batch[:, 2]
        next_state_batch = np.concatenate(batch[:, 3])\
            .reshape(self.batch_size, self.state_size)
        done_batch = batch[:, 4]
        # action processing
        #action_batch = np.where(action_batch == 1)

        #print "action_batch:",action_batch
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    #from tgym.envs import SpreadTrading
    #from tgym.gens.deterministic import WavySignal
    # Instantiating the environmnent
    #generator = WavySignal(period_1=25, period_2=50, epsilon=-0.5)
    episodes = 500
    episode_length = 400
    trading_fee = .2
    time_fee = 0
    history_length = 2
    '''
    environment = SpreadTrading(spread_coefficients=[1],
                                data_generator=generator,
                                trading_fee=trading_fee,
                                time_fee=time_fee,
                                history_length=history_length,
                                episode_length=episode_length)
    '''
    environment = gym.make('trading-v0').env
    environment.initialise(symbol='000001', start='2015-09-01', end='2017-09-01', days=252)

    state = environment.reset()
    # Instantiating the agent
    memory_size = 1000
    state_size = len(state)
    gamma = 0.96
    epsilon_min = 0.01
    batch_size = 64
    #action_size = len(environment._actions)
    action_size = 3 #FIXIT
    train_interval = 10
    learning_rate = 0.001
    agent = DQNAgent(state_size=state_size,
                     action_size=action_size,
                     memory_size=memory_size,
                     episodes=episodes,
                     episode_length=episode_length,
                     train_interval=train_interval,
                     gamma=gamma,
                     learning_rate=learning_rate,
                     batch_size=batch_size,
                     epsilon_min=epsilon_min)
    # Warming up the agent
    for _i in range(memory_size):
        action = agent.act(state)
        #print "action:", action
        next_state, reward, done, info = environment.step(action)
        #print _i, next_state, reward, done, info
        if done == True:
            environment.reset()
            continue
        agent.observe(state, action, reward, next_state, done, warming_up=True)
    print "Warming up over .............."
     # Training the agent
    for ep in range(episodes):
        state = environment.reset()
        rew = 0
        for _ in range(episode_length):
            action = agent.act(state)
            next_state, reward, done, _ = environment.step(action)
            loss = agent.observe(state, action, reward, next_state, done)
            #print "1 DEBUG loss :",loss
            state = next_state
            rew += reward
            if done == True:
                environment.reset()
                continue
        #print("Ep:" + str(ep)
        #      + "| rew:" + str(round(rew, 2))
        #      + "| eps:" + str(round(agent.epsilon, 2))
        #      + "| loss:" + str(round(loss.history["loss"][0], 4)))

    # Running the agent
    print "Train  over .............."

    done = False
    state = environment.reset()
    while not done:
        action = agent.act(state)
        state, _, done, info = environment.step(action)
        print info
        if 'status' in info and info['status'] == 'Closed plot':
            print "done  True .............."
            done = True
        else:
            print "render  True .............."
            #environment.render()