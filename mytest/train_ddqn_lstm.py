# -*- coding: utf-8 -*-


import gym_trading  #必须引入才自动注册
import gym
import numpy as np
import pandas as pd
import random
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, Adam

from collections import deque
import matplotlib.pyplot as plt

class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.tau = .125
        self.batch_size = 64 # 64 TODO
        self.state_size = self.env.observation_space.shape[0]

        self.model = self.create_model()
        self.target_model = self.create_model()

    '''
    def create_model(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape
        print 'state_shape:',self.env.observation_space.shape
        model.add(Dense(24, input_dim=self.state_size, activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))
        return model
    '''

    def create_model(self):
        model = Sequential()
        model.add(LSTM(64,
                       input_shape=(1, self.state_size),
                       return_sequences=True,
                       stateful=False))
        model.add(Dropout(0.5))
        model.add(LSTM(64,
                       input_shape=(1, self.state_size),
                       return_sequences=False,
                       stateful=False))
        model.add(Dropout(0.5))

        model.add(Dense(self.env.action_space.n, kernel_initializer='lecun_uniform'))
        model.add(Activation('linear'))  # linear output so we can have range of real-valued outputs

        rms = RMSprop()
        adam = Adam()
        model.compile(loss='mse', optimizer=adam)
        return model


    def act(self, state):
        """Acting Policy of the DQNAgent
        """
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        #print "------------------------ epsilon 0:",self.epsilon

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        #return np.argmax(self.model.predict(state,batch_size=1))
        qval = self.model.predict(state, batch_size=1)
        #print "------------------------ qval 0:", np.argmax(qval),qval,state
        assert np.any(np.isnan(qval)) == False
        #print "b",np.shape(state),state,np.shape(qval),qval
        return np.argmax(qval)
        #return np.argmax(self.model.predict(state,batch_size=1))

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        """Memory Management and training of the agent
        """
        if len(self.memory) < self.batch_size:
            return

        state, action, reward, next_state, done = self._get_batches()
        #print (np.shape(state))
        reward += (self.gamma
                   * np.logical_not(done)
                   * np.amax(self.model.predict(next_state.reshape(self.batch_size,1,self.state_size)),axis=1))
        q_target = self.target_model.predict(state.reshape(self.batch_size,1,self.state_size))
        #reward += (self.gamma
        #           * np.logical_not(done)
        #           * np.amax(self.model.predict(next_state,batch_size=1), axis=1))
        #q_target = self.target_model.predict(state,batch_size=1)
        # print "state :",state,np.shape(state)
        # print "reward :",reward,np.shape(reward)
        # print "action :",action,np.shape(action)
        # print "q_target :",q_target,np.shape(q_target)
        # q_target[action[0], action[1]] = reward
        # where?
        _ = pd.Series(action)
        one_hot = pd.get_dummies(_).as_matrix()
        action_batch = np.where(one_hot == 1)
        # print "batch:",action_batch
        q_target[action_batch] = reward
        return self.model.fit(state.reshape(self.batch_size,1,self.state_size), q_target,
                              batch_size=self.batch_size,
                              epochs=1,
                              verbose=False)


    def _get_batches(self):
        """Selecting a batch of memory
           Split it into categorical subbatches
           Process action_batch into a position vector
        """
        batch = np.array(random.sample(self.memory, self.batch_size))
        state_batch = np.concatenate(batch[:, 0]) \
            .reshape(self.batch_size, self.state_size)
        # action_batch = np.concatenate(batch[:, 1])\
        #    .reshape(self.batch_size, self.action_size)
        action_batch = batch[:, 1]

        reward_batch = batch[:, 2]
        next_state_batch = np.concatenate(batch[:, 3]) \
            .reshape(self.batch_size, self.state_size)
        done_batch = batch[:, 4]
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


    def target_train(self):
        weights = self.model.get_weights()
        #print "target train weight:",weights
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)


def main():
    env = gym.make('trading-v0').env
    env.initialise(symbol='000001', start='2015-01-01', end='2017-01-01', days=252)

    trials = 500
    trial_len = 500

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    #steps = []
    simrors = np.zeros(trials)
    mktrors = np.zeros(trials)
    victory = False
    i = 0
    for trial in range(trials):
        if victory == True:
            break;
        cur_state = env.reset().reshape(1,1,dqn_agent.state_size) #FIX IT
        #print("cur_state :", np.shape(cur_state),cur_state)
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            #print "action;",action
            new_state, reward, done, _ = env.step(action)
            #print  new_state, reward, done, _
            # reward = reward if not done else -20
            new_state = new_state.reshape(1,1,dqn_agent.state_size)
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()  # internally iterates default (prediction) model
            dqn_agent.target_train()  # iterates target model

            cur_state = new_state
            i += 1
            if trial >= 3000:
                #####################################################################################
                #print "trail is :",trial
                env.render()

            ####################################################################################

            if done:
                print "done ",trial,step
                #break
        #if step >= 199:
        #    print("Failed to complete in trial {}".format(trial))
        #    if step % 10 == 0:
        #        dqn_agent.save_model("trial-{}.model".format(trial))
        #else:
                df = env.sim.to_df()
                #print df.tail()
                #print df.bod_nav.values[-1]
                # pdb.set_trace()
                #print df.bod_nav.values
                #print df.bod_nav.values[-1]
                simrors[trial] = df.bod_nav.values[-1] - 1  # compound returns
                mktrors[trial] = df.mkt_nav.values[-1] - 1

                print('year #%6d, sim ret: %8.4f, mkt ret: %8.4f, net: %8.4f', i,
                        simrors[trial], mktrors[trial], simrors[trial] - mktrors[trial])
                #save_path = self._saver.save(self._sess, model_dir + 'model.ckpt',
                #                             global_step=episode + 1)

                #print simrors[i - 100:i]
                #print mktrors[i - 100:i]
                if trial > 5:
                    vict = pd.DataFrame({'sim': simrors[trial - 5:trial],
                                         'mkt': mktrors[trial - 5:trial]})
                    vict['net'] = vict.sim - vict.mkt
                    print('vict:',vict.net.mean())
                    if vict.net.mean() > 30.0:
                        victory = True
                        print('Congratulations, Warren Buffet!  You won the trading game.')
                break


    print("Completed in {} trials".format(trial))
    dqn_agent.save_model("success.model")
    print "model.get_weights():",dqn_agent.model.get_weights()
    #break


if __name__ == "__main__":
    main()
