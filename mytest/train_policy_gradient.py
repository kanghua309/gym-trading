import gym
import tensorflow as tf
from matplotlib import interactive
interactive(True)
import logging

log = logging.getLogger()
#log.addHandler(logging.StreamHandler())
import policy_gradient 
# create gym
env = gym.make('trading-v0').env
env.initialise(symbol='000001',start='2015-01-01',end='2017-01-01',days=252)

sess = tf.InteractiveSession()

# create policygradient
pg = policy_gradient.PolicyGradient(sess, obs_dim=8, num_actions=3, learning_rate=1e-2)

# train model, loading if possible
alldf,summrzed = pg.train_model( env,episodes=1001, log_freq=100)#, load_model=True)
print summrzed
#pd.DataFrame(sharpes).expanding().mean().plot()

