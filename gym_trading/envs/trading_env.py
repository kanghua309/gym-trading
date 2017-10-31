# -*- coding: utf-8 -*-

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from collections import Counter

import quandl
import numpy as np
from numpy import random
import pandas as pd
import logging
import pdb

import numpy as np
import pandas as pd
from me.helper.research_env import Research
import matplotlib.pyplot as plt

import tempfile

log = logging.getLogger(__name__)
log.info('%s logger started.',__name__)


def _sharpe(Returns, freq=252) :
  """Given a set of returns, calculates naive (rfr=0) sharpe """
  return (np.sqrt(freq) * np.mean(Returns))/np.std(Returns)

def _prices2returns(prices):
  px = pd.DataFrame(prices)
  nl = px.shift().fillna(0)
  R = ((px - nl)/nl).fillna(0).replace([np.inf, -np.inf], np.nan).dropna()
  R = np.append( R[0].values, 0)
  return R


'''
class QuandlEnvSrc(object):

  #Quandl-based implementation of a TradingEnv's data source.
  #Pulls data from Quandl, preps for use by TradingEnv and then
  #acts as data provider for each new episode.
 

  MinPercentileDays = 100 
  QuandlAuthToken = ""  # not necessary, but can be used if desired
  Name = "GOOG/NYSE_SPY" #"GOOG/NYSE_IBM"

  def __init__(self, days=252, name=Name, auth=QuandlAuthToken, scale=True ):
    self.name = name
    self.auth = auth
    self.days = days+1
    log.info('getting data for %s from quandl...',QuandlEnvSrc.Name)
    df = quandl.get(self.name) if self.auth=='' else quandl.get(self.name, authtoken=self.auth)
    log.info('got data for %s from quandl...',QuandlEnvSrc.Name)
    
    df = df[ ~np.isnan(df.Volume)][['Close','Volume']]
    # we calculate returns and percentiles, then kill nans
    df = df[['Close','Volume']]   
    df.Volume.replace(0,1,inplace=True) # days shouldn't have zero volume..
    df['Return'] = (df.Close-df.Close.shift())/df.Close.shift()
    pctrank = lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    df['ClosePctl'] = df.Close.expanding(self.MinPercentileDays).apply(pctrank)
    df['VolumePctl'] = df.Volume.expanding(self.MinPercentileDays).apply(pctrank)
    df.dropna(axis=0,inplace=True)
    R = df.Return
    if scale:
      mean_values = df.mean(axis=0)
      std_values = df.std(axis=0)
      df = (df - np.array(mean_values))/ np.array(std_values)
    df['Return'] = R # we don't want our returns scaled
    self.min_values = df.min(axis=0)
    self.max_values = df.max(axis=0)
    self.data = df
    self.step = 0
    
  def reset(self):
    # we want contiguous data
    self.idx = np.random.randint( low = 0, high=len(self.data.index)-self.days )
    self.step = 0

  def _step(self):    
    obs = self.data.iloc[self.idx].as_matrix()
    self.idx += 1
    self.step += 1
    done = self.step >= self.days
    return obs,done
'''

ret = lambda x,y: np.log(y/x) #Log return
zscore = lambda x:(x -x.mean())/x.std() # zscore

class ZiplineEnvSrc(object):
  # Quandl-based implementation of a TradingEnv's data source.
  # Pulls data from Quandl, preps for use by TradingEnv and then
  # acts as data provider for each new episode.

  MinPercentileDays = 100

  def __init__(self,symbol,start,end,days=252, scale=True):
    self.symbol = symbol
    self.days = days + 1

    log.info('getting data for %s from zipline bundle...', symbol)
    research = Research()

    #log.info('got data for %s from quandl...', QuandlEnvSrc.Name)
    print "debug :", self.symbol,start,end
    panel = research.get_pricing([self.symbol],start,end,'1d',['close','low','high','open','volume'])
    _df = panel.transpose(2, 1, 0).iloc[0]


    #df = df[~np.isnan(df.volume)][['close', 'volume']]
    # we calculate returns and percentiles, then kill nans
    #df = df[['close', 'volume']]
    #_df.volume.replace(0, 1, inplace=True)  # days shouldn't have zero volume..
    _df.dropna(axis=0, inplace=True)

    df = pd.DataFrame()
    df['Return'] = (_df.close-_df.close.shift())/_df.close.shift() # today return
    df['H20'] = zscore(ret(_df.high,_df.open))
    df['L20'] = zscore(ret(_df.low,_df.open))
    df['C2O'] = zscore(ret(_df.close,_df.open))
    df['H2C'] = zscore(ret(_df.high,_df.close))
    df['L2C'] = zscore(ret(_df.low,_df.close))
    df['H2L'] = zscore(ret(_df.high,_df.low))
    df['VOL'] = zscore(_df.volume)
    #df['Price'] = _df.close

    #pctrank = lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    #df['ClosePctl'] = df.close.expanding(self.MinPercentileDays).apply(pctrank)
    #df['VolumePctl'] = df.volume.expanding(self.MinPercentileDays).apply(pctrank)
    #df.dropna(axis=0, inplace=True)
    #R = df.Return
    #if scale:
    #  mean_values = df.mean(axis=0)

    #  std_values = df.std(axis=0)
    #  df = (df - np.array(mean_values)) / np.array(std_values)
    #df['Return'] = R  # we don't want our returns scaled
    self.min_values = df.min(axis=0)
    self.max_values = df.max(axis=0)
    self.data = df
    self.step = 0
    self.orgin_idx = 0
    self.prices = _df.close

  def reset(self):
    # we want contiguous data
    #print self.data.index
    #print len(self.data.index)
    self.idx = np.random.randint(low=0, high=len(self.data.index) - self.days)
    #self.idx = 0
    self.step = 0
    self.orgin_idx = self.idx  #for render , so record it

  def _step(self):
    obs = self.data.iloc[self.idx].as_matrix()
    self.idx  += 1
    self.step += 1
    done = self.step >= self.days
    return obs, done




class TradingSim(object) :
  """ Implements core trading simulator for single-instrument univ """

  def __init__(self, steps, trading_cost_bps = 1e-3, time_cost_bps = 1e-4):
    # invariant for object life
    self.trading_cost_bps = trading_cost_bps
    self.time_cost_bps    = time_cost_bps
    self.steps            = steps
    # change every step
    self.step = 0
    self.actions = np.zeros(self.steps)
    self.navs = np.ones(self.steps)
    self.mkt_nav = np.ones(self.steps)
    self.strat_retrns = np.ones(self.steps)
    self.posns = np.zeros(self.steps)
    self.costs = np.zeros(self.steps)
    self.trades = np.zeros(self.steps)
    self.mkt_retrns = np.zeros(self.steps)




  def reset(self, train=True):
    #self.total_reward = 0
    #self.total_trades = 0
    #self.average_profit_per_trade = 0
    #self.count_open_trades = 0

    #if train:
    #  self.current_time = 1
    #else:
    #  self.current_time = self.train_end + 1

    #self.curr_trade = {'Entry Price': 0, 'Exit Price': 0, 'Entry Time': None, 'Exit Time': None, 'Profit': 0,'Trade Duration': 0, 'Type': None, 'reward': 0}
    #self.journal = []
    #self.open_trade = False
    self.step = 0
    self.actions.fill(0)
    self.navs.fill(1)
    self.mkt_nav.fill(1)
    self.strat_retrns.fill(0)
    self.posns.fill(0)
    self.costs.fill(0)
    self.trades.fill(0)
    self.mkt_retrns.fill(0)

  def _step(self, action, retrn ):
    """ Given an action and return for prior period, calculates costs, navs,
        etc and returns the reward and a  summary of the day's activity. """

    bod_posn = 0.0 if self.step == 0 else self.posns[self.step-1]
    bod_nav  = 1.0 if self.step == 0 else self.navs[self.step-1]
    mkt_nav  = 1.0 if self.step == 0 else self.mkt_nav[self.step-1]

    #  action # 1 flat ; 0 short ; 2 long
    #  pos    # 0  -1  1
    #if (action == 0 and bod_posn != 1) or (action == 2 and  bod_posn != 0) : #我们没有-1状态；允许开仓情况下，再发出买入信号？ 或 平仓情况下，还发出卖出信号？ —— 是否是中强化
    #    msg = "Pre Pos is %s , can not do action %s" % (bod_posn,action)
    #    _thorwErr(msg)

    self.mkt_retrns[self.step] = retrn
    self.actions[self.step] = action
    
    self.posns[self.step] = action - 1
    self.trades[self.step] = self.posns[self.step] - bod_posn

    trade_costs_pct = abs(self.trades[self.step]) * self.trading_cost_bps 
    self.costs[self.step] = trade_costs_pct +  self.time_cost_bps
    reward = ( (bod_posn * retrn) - self.costs[self.step] )
    self.strat_retrns[self.step] = reward
    #print (self.step,retrn)
    #print ("debug ----- :action:%d,bod_posn,%d,posn:%d,trades:%d,trade_costs_pct:%f,costs:%f,reward:%f" % (action,
    #                                                                                            bod_posn,
    #                                                                                             self.posns[self.step],
    #                                                                                             self.trades[self.step],
    #                                                                                             trade_costs_pct,
    #                                                                                             self.costs[self.step],
    #                                                                                             reward))

    if self.step != 0 :
      self.navs[self.step] =  bod_nav * (1 + self.strat_retrns[self.step-1])
      self.mkt_nav[self.step] =  mkt_nav * (1 + self.mkt_retrns[self.step-1])
    
    info = { 'reward': reward, 'nav':self.navs[self.step], 'costs':self.costs[self.step] ,'pos': self.posns[self.step]}
    self.step += 1      
    return reward, info

  def to_df(self):
    """returns internal state in new dataframe """
    cols = ['action', 'bod_nav', 'mkt_nav','mkt_return','sim_return',
            'position','costs', 'trade' ]
    #rets = _prices2returns(self.navs)
    #pdb.set_trace()
    df = pd.DataFrame( {  'action':     self.actions, # today's action (from agent)
                          'bod_nav':    self.navs,    # BOD Net Asset Value (NAV)
                          'mkt_nav':    self.mkt_nav, #
                          'mkt_return': self.mkt_retrns,
                          'sim_return': self.strat_retrns,
                          'position':   self.posns,   # EOD position
                          'costs':      self.costs,   # eod costs
                          'trade':      self.trades },# eod trade
                         columns=cols)
    return df

class TradingEnv(gym.Env):
  """This gym implements a simple trading environment for reinforcement learning.

  The gym provides daily observations based on real market data pulled
  from Quandl on, by default, the SPY etf. An episode is defined as 252
  contiguous days sampled from the overall dataset. Each day is one
  'step' within the gym and for each step, the algo has a choice:

  SHORT (0)
  FLAT (1)
  LONG (2)

  If you trade, you will be charged, by default, 10 BPS of the size of
  your trade. Thus, going from short to long costs twice as much as
  going from short to/from flat. Not trading also has a default cost of
  1 BPS per step. Nobody said it would be easy!

  At the beginning of your episode, you are allocated 1 unit of
  cash. This is your starting Net Asset Value (NAV). If your NAV drops
  to 0, your episode is over and you lose. If your NAV hits 2.0, then
  you win.

  The trading env will track a buy-and-hold strategy which will act as
  the benchmark for the game.

  """
  metadata = {'render.modes': ['human']}

  def __init__(self):
      self.inited = False;
      pass

  def initialise(self,symbol,start,end,days):
      self.days = days
      self.src = ZiplineEnvSrc(symbol=symbol, start=start, end=end, days=self.days)
      self.sim = TradingSim(steps=self.days, trading_cost_bps=1e-3, time_cost_bps=1e-4)

      self.action_space = spaces.Discrete(3)
      self.observation_space = spaces.Box(self.src.min_values,
                                          self.src.max_values)
      self.reset()
      self.inited = True
      self.render_on = 0
      self.reset_count = 0

  def _configure(self, display=None):
    self.display = display

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _step(self, action):
    if self.inited == False : return
    assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
    observation, done = self.src._step()
    yret = observation[0] #RETURN
    #print "yret:",yret ," action:",action
    reward, info = self.sim._step( action, yret )
    return observation, reward, done, info
  
  def _reset(self):
    if self.inited == False : return
    self.reset_count += 1
    self.src.reset()
    self.sim.reset()
    return self.src._step()[0]

  def _plot_trades(self):

    ####################################################################
    plt.subplot(3, 1, 1)
    p = self.src.prices[self.src.orgin_idx:]  #TODO
    p = p.reset_index(drop=True).head(self.days)
    p.plot(style='kx-', label = 'price')
    l = ['price']
    # --- plot trades
    # colored line for long positions
    idx = (pd.Series(self.sim.trades) > 0)
    if idx.any():
      p[idx].plot(style='go')
      l.append('long')
    # colored line for short positions
    idx = (pd.Series(self.sim.trades) < 0)
    if idx.any():
      p[idx].plot(style='ro')
      l.append('short')

    plt.xlim([p.index[0], p.index[-1]])  # show full axis
    plt.legend(l ,loc='upper right')
    plt.title('trades')
    plt.draw()

    ####################################################################
    plt.subplot(3, 1, 2)
    pd.Series(self.sim.mkt_nav).plot(style='g')
    plt.title('market net value')
    plt.draw()

    plt.subplot(3, 1, 3)
    pd.Series(self.sim.navs).plot(style='r')
    plt.title('simulate net value')
    plt.draw()

    #print self.sim.to_df()
    return plt

  def _render(self, mode='human', close=False):
    #plt.figure(figsize=(3, 4))
    if self.inited == False : return

    if self.render_on == 0:
       #self.fig = plt.figure(figsize=(10, 4))
       self.fig = plt.figure(figsize=(12, 9))
       self.render_on = 1
       plt.ion()
       #plt.show()

    plt.clf()
    self._plot_trades()
    plt.suptitle(str("round:" + str(self.reset_count) + "-" + str("step:" + str(self.src.idx - self.src.orgin_idx))))
    #plt.draw()
    plt.pause(0.001)
    return self.fig

    #plt.axvline(x=400, color='black', linestyle='--')
    #plt.text(250, 400, 'training data')
    #plt.text(450, 400, 'test data')
    #plt.suptitle(str(epoch))
    #plt.savefig('./' + str(epoch) + '.png', bbox_inches='tight', pad_inches=1, dpi=72)
    #plt.close('all')
    #pass
  # some convenience functions:
  
  def run_strat(self,  strategy, return_df=True):
    if self.inited == False : return
    """run provided strategy, returns dataframe with all steps"""
    observation = self.reset()
    done = False
    #bod_posn = 0.0
    count =0
    while not done:
      action = strategy( observation, self ) # call strategy
      #if (action == 0 and bod_posn != 1) or (action == 2 and bod_posn != 0):
      #    continue
      observation, reward, done, info = self.step(action)
      #bod_posn = info['pos']
      count += 1
      print observation, reward, done, info, count

    return self.sim.to_df() if return_df else None
      
  def run_strats( self, strategy, episodes=1, write_log=True, return_df=True):
    if self.inited == False : return

    """ run provided strategy the specified # of times, possibly
        writing a log and possibly returning a dataframe summarizing activity.
    
        Note that writing the log is expensive and returning the df is moreso.  
        For training purposes, you might not want to set both.
    """
    logfile = None
    if write_log:
      logfile = tempfile.NamedTemporaryFile(delete=False)
      log.info('writing log to %s',logfile.name)
      need_df = write_log or return_df

    alldf = None
        
    for i in range(episodes):
      df = self.run_strat(strategy, return_df=need_df)
      if write_log:
        df.to_csv(logfile, mode='a')
        if return_df:
          alldf = df if alldf is None else pd.concat([alldf,df], axis=0)
            
    return alldf
