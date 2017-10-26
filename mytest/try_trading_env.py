# -*- coding: utf-8 -*-

import gym

import gym_trading
import pandas as pd
import numpy as np

pd.set_option('display.width',500)

env = gym.make('trading-v0').env
env.initialise(symbol='000001',start='2015-01-01',end='2017-01-01',days=256)
#env.time_cost_bps = 0

Episodes=1

obs = []

for _ in range(Episodes):
    observation = env.reset()
    print "observation:",observation
    done = False
    count = 0
    bod_posn = 0.0
    info = None
    while not done:
        action = env.action_space.sample() # random
        #if info != None:
        #    bod_posn = info['pos']
        #print "----------------------:",action,bod_posn
        #if (action == 0 and bod_posn != 1) or (action == 2 and bod_posn != 0):
        #    continue
        print "sample action:",action
        observation, reward, done, info = env.step(action)
        obs = obs + [observation]
        count += 1
        print observation,reward,done,info,count
        if done:
            print "-------------------done------------------"
            print reward
            print count
        
df = env.sim.to_df()

df.head()
#df.tail()

#buyhold = lambda x,y : 2
#df = env.run_strat( buyhold )

#print "---------df-------:",df
#df10 = env.run_strats( buyhold, Episodes )
