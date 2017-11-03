import gym
import click

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.optimizers import Adam
import gym_trading
import logging
import pandas as pd


log = logging.getLogger(__name__)
logging.basicConfig()
log.setLevel(logging.INFO)
log.info('%s logger started.', __name__)

class PGAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.model = self._build_model()
        self.model.summary()
    '''
    def _build_model(self):
        model = Sequential()
        model.add(Reshape((1, 80, 80), input_shape=(self.state_size,)))
        model.add(Convolution2D(32, 6, 6, subsample=(3, 3), border_mode='same',
                                activation='relu', init='he_uniform'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu', init='he_uniform'))
        model.add(Dense(32, activation='relu', init='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model
    '''

    def _build_model(self):
        model = Sequential()
        neurons_per_layer = 32
        activation = "relu"
        model.add(Dense(neurons_per_layer,
                        input_dim=self.state_size,
                        activation=activation))
        model.add(Dense(neurons_per_layer, activation=activation))
        model.add(Dense(self.action_size,  activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, prob, reward):
        #print "state:",np.shape(state)
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)

    def act(self, state):
        state = state.reshape(1, self.state_size)
        #print "act:",np.shape(state)
        aprob = self.model.predict(state, batch_size=1).flatten()
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        #print "prop:",prob
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        #print "action:",action
        return action, prob

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards = rewards / np.std(rewards - np.mean(rewards))
        gradients *= rewards
        X = np.squeeze(([self.states]))
        #X = self.states
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        #print "X:", np.shape(X) , " Y:",np.shape(Y)
        self.model.train_on_batch(X, Y)
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []

    def load_model(self, name):
        self.model.load_weights(name)

    def save_model(self, name):
        self.model.save_weights(name)

@click.command()
@click.option(
    '-s',
    '--symbol',
    default='000001',
    show_default=True,
    help='given stock code ',
)
@click.option(
    '-b',
    '--begin',
    default='2016-09-01',
    show_default=True,
    help='The begin date of the train.',
)

@click.option(
    '-e',
    '--end',
    default='2017-09-01',
    show_default=True,
    help='The end date of the train.',
)

@click.option(
    '-d',
    '--days',
    type=int,
    default=100,
    help='train days',
)

@click.option(
    '-t',
    '--train_round',
    type=int,
    default=10000,
    help='train round',
)


@click.option(
     '--plot/--no-plot',
     #default=os.name != "nt",
     is_flag = True,
     default=False,
     help="render when training"
)

@click.option(
    '-m',
    '--model_path',
    default='.',
    show_default=True,
    help='trained model save path.',
)

def execute(symbol,begin,end,days,train_round,plot,model_path):
    trial = train_round

    env = gym.make('trading-v0').env
    env.initialise(symbol=symbol, start=begin, end=end, days=days)
    state = env.reset()

    agent = PGAgent(env.observation_space.shape[0], env.action_space.n)
    episode = 0
    score = 0
    victory = False
    simrors = np.zeros(trial)
    mktrors = np.zeros(trial)
    while episode < trial and not victory :
        action,prob = agent.act(state)
        new_state,reward,done,info = env.step(action)
        new_state = state.reshape(1, env.observation_space.shape[0])
        agent.remember(new_state,action,prob,reward)
        score += reward
        #print('Episode: %d - Score: %f.' % (episode, score))
        if done:
            #print('Episode: %d - Score: %f.' % (episode, score))            # pdb.set_trace()
            df = env.sim.to_df()
            simrors[episode] = df.bod_nav.values[-1] - 1  # compound returns
            mktrors[episode] = df.mkt_nav.values[-1] - 1
            #alldf = df if alldf is None else pd.concat([alldf, df], axis=0)
            if episode % 10 == 0:
                log.info('year #%6d, sim ret: %8.4f, mkt ret: %8.4f, net: %8.4f', episode,
                         simrors[episode], mktrors[episode], simrors[episode] - mktrors[episode])

                if episode > 10:
                    vict = pd.DataFrame({'sim': simrors[episode - 10:episode],
                                         'mkt': mktrors[episode - 10:episode]})
                    vict['net'] = vict.sim - vict.mkt
                    log.info('vict:%f', vict.net.mean())
                    if vict.net.mean() > 0.1:
                        victory = True
                        log.info('Congratulations, Warren Buffet!  You won the trading game ',)
                        break
            agent.train()
            state = env.reset()
            state = state.reshape(1, env.observation_space.shape[0])

            episode += 1
            score = 0
    import os
    log.info("Completed in %d trials , save it as %s", trial,
             os.path.join(model_path, env.src.symbol + ".model"))
    agent.save_model(os.path.join(model_path, env.src.symbol + ".model"))



if __name__ == "__main__":
    execute()


