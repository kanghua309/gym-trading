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

from keras.models import load_model


if __name__ == "__main__":
    print "__________________________________________________________"
    model = load_model('success.model')

    env = gym.make('trading-v0').env
    env.initialise(symbol='000001', start='2015-01-01', end='2017-01-01', days=252)

    state = env.reset()
    while not done:
        #action = agent.act(state)
        state = state.reshape(1, 8)
        act_values = model.predict(state)
        # action[np.argmax(act_values[0])] = 1
        # print "x:",np.argmax(act_values[0])
        action = np.argmax(act_values[0])
        # print "action:", action
        state, _, done, info = env.step(action)
        print info

        print "render  True .............."
        env.render()