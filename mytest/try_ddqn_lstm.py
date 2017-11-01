# -*- coding: utf-8 -*-

import gym_trading  #必须引入才自动注册
import gym
import numpy as np


from keras.models import load_model
import logging
import click


log = logging.getLogger(__name__)
logging.basicConfig()
log.setLevel(logging.INFO)
log.info('%s logger started.',__name__)


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
    default='2015-09-01',
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
    default=252,
    help='train days',
)


@click.option(
     '--plot/--no-plot',
     #default=os.name != "nt",
     is_flag = True,
     default=False,
     help="render when training"
)

def execute(symbol,begin,end,days,plot):
    model = load_model('success.model')
    env = gym.make('trading-v0').env
    env.initialise(symbol=symbol, start=begin, end=end, days=252)

    state = env.reset()
    done = False
    while not done:
        state = state.reshape(1, 1, 8)
        qval = model.predict(state, batch_size=1)
        action = (np.argmax(qval))
        state, _, done, info = env.step(action)
        log.info("%s,%s,%s,%s",state, _, done, info)
        if plot :
            env.render()

if __name__ == "__main__":
    execute()