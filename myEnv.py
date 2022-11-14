
import numpy as np
import pandas as pd
import math
import random
from util import *
from config import *

from stable_baselines3.common.vec_env import DummyVecEnv
import statistics

import sys
sys.path.append("../FinRL-Library")

from gym.utils import seeding
import gym
from gym import spaces

class StockPortfolioEnv(gym.Env):
    """A single stock trading environment for OpenAI gym

    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        hmax : int
            maximum number of shares to trade
        initial_amount : int
            start money
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        turbulence_threshold: int
            a threshold to control risk aversion
        day: int
            an increment number to control date

    Methods
    -------
    _sell_stock()
        perform sell action based on the sign of the action
    _buy_stock()
        perform buy action based on the sign of the action
    step()
        at each step the agent will return actions, then 
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step
        

    """
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                df,
                stock_dim,
                hmax,
                initial_amount,
                transaction_cost_pct,
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                stdev_tolerance_range,
                reward_by = 'reward',
                reward_by_value_list = [0,0,0],
                reward_fuction_type = 1, # 1:bestABC, 2:closestABC, 3:bestA/B/C+closestABC ,4:bestA/B/C+bestABC
                turbulence_threshold=None,
                lookback=252,
                day = 0):
        #super(StockEnv, self).__init__()
        #money = 10 , scope = 1
        self.day = day
        self.lookback=lookback
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct =transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.max_portfolio_value = 0        
        self.mdd = 0.0001
        self.reward_fuction_type = reward_fuction_type
        self.reward_by = reward_by
        self.reward_by_value_list = reward_by_value_list
        self.return_list = []
        self.week_return_list=[]
        self.pre_value = initial_amount
        self.week_std = 0.00001
        self.stdev_tolerance_range = stdev_tolerance_range


        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low = 0, high = 1,shape = (self.action_space,)) 
        # Shape = (34, 30)
        # covariance matrix + technical indicators
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_space+len(self.tech_indicator_list),self.state_space))

        # load data from a pandas dataframe
        self.data = self.df.loc[self.day,:]
        self.covs = self.data['cov_list'].values[0]
        self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
        self.terminal = False     
        self.turbulence_threshold = turbulence_threshold        
        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim +[0]*self.stock_dim ]
        self.weights_memory=[ [1/self.stock_dim]*self.stock_dim ]
        self.date_memory=[self.data.date.unique()[0]]

        
    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique())-1
#         print(actions)

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ['daily_return']
            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ['daily_return']
            if df_daily_return['daily_return'].std() !=0:
                sharpe = (252**0.5)*df_daily_return['daily_return'].mean()/ \
                       df_daily_return['daily_return'].std()
            return self.state, self.reward, self.terminal,{}

        else:
            len_port = int(len(actions)/2)
#             weights = self.softmax_normalization(actions) 
            weights_mean = actions[:len_port] 
            weights_var = actions[len_port:]
            weights = []
            for i in range(len_port):
                w = random.gauss(weights_mean[i], weights_var[i])
                weights.append(w)
            weights = self.softmax_normalization(weights) 
            self.actions_memory.append(actions)
            self.weights_memory.append(weights)
            
            last_day_memory = self.data

            #load next state
            self.day += 1
            self.data = self.df.loc[self.day,:]
            self.covs = self.data['cov_list'].values[0]
            self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
            #print(self.state)
            # calcualte portfolio return=individual stocks' return * weight
            portfolio_return = sum(((self.data.close.values / last_day_memory.close.values)-1)*weights)
            # update portfolio value
            # print(self.portfolio_value,portfolio_return)
            new_portfolio_value = self.portfolio_value*(1+portfolio_return)
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])            
            self.asset_memory.append(new_portfolio_value)
            
            # cal uncertainty
            sharpness = np.sqrt(np.mean(weights_var ** 2))

            # cal ABC & reward function
            now_return = (new_portfolio_value-self.initial_amount)/self.initial_amount #reward
            #mdd
            ann_return = (1+now_return) ** (252/self.day) -1
            if self.max_portfolio_value <  new_portfolio_value :
                self.max_portfolio_value = new_portfolio_value
            now_dd = 1-(new_portfolio_value/self.max_portfolio_value)
            if self.mdd < now_dd:
                self.mdd = now_dd
            #stdev
            rrr = (new_portfolio_value-self.pre_value)/self.pre_value
            self.return_list.append(rrr)
            
            if len(self.return_list)%5==0:
                tmp_return = np.prod(np.array(self.return_list[-5:].copy())+1)-1
#                 print('week return',tmp_return)
                self.week_return_list.append(tmp_return)
                if len(self.week_return_list)>1:
                    self.week_std = statistics.stdev(self.week_return_list)* math.pow( 50.4, 0.5 )

            if len(self.return_list)<21:
                stdev = 0.0000001
            else:
                self.return_list.pop(0)
                stdev = statistics.stdev(self.return_list)* math.pow( 252, 0.5 )
                
            if self.reward_fuction_type==1: # best A/B/C
                if self.reward_by=='reward':
                    self.reward = ann_return

                elif self.reward_by=='mdd':
                    self.reward = self.mdd*(-1)

                elif self.reward_by=='1mstdev':
                    self.reward = stdev*(-1)
                    
            elif self.reward_fuction_type==2: # best A/B/C + closest ABC
                if self.reward_by=='reward':
                    self.reward = ann_return + (abs(self.mdd-self.reward_by_value_list[1])+abs(stdev-self.reward_by_value_list[2]))*(-1) 

                elif self.reward_by=='mdd':
                    self.reward = self.mdd*(-1) + (abs(ann_return-self.reward_by_value_list[0])+abs(stdev-self.reward_by_value_list[2]))*(-1) 

                elif self.reward_by=='1mstdev':
                    self.reward = stdev*(-1) + (abs(ann_return-self.reward_by_value_list[0])+abs(self.mdd-self.reward_by_value_list[1]))*(-1) 
                    
                if ann_return<(self.reward_by_value_list[0]*(1-TOLERANCE_RANGE)):
                    self.reward -= abs(ann_return-self.reward_by_value_list[0]*(1-TOLERANCE_RANGE))
                if self.mdd>(self.reward_by_value_list[1]):
                    self.reward -= abs(self.mdd-self.reward_by_value_list[1])
                if stdev>(self.stdev_tolerance_range):
                    self.reward -= abs(self.week_std-self.stdev_tolerance_range)

            elif self.reward_fuction_type==3: # best A/B/C + closest ABC + uncertainty
                if self.reward_by=='reward':
                    self.reward = ann_return + (abs(self.mdd-self.reward_by_value_list[1])+abs(stdev-self.reward_by_value_list[2]))*(-1) - sharpness #+ ann_return/stdev

                elif self.reward_by=='mdd':
                    self.reward = self.mdd*(-1) + (abs(ann_return-self.reward_by_value_list[0])+abs(stdev-self.reward_by_value_list[2]))*(-1) - sharpness #+ ann_return/stdev

                elif self.reward_by=='1mstdev':
                    self.reward = stdev*(-1) + (abs(ann_return-self.reward_by_value_list[0])+abs(self.mdd-self.reward_by_value_list[1]))*(-1) - sharpness #+ ann_return/stdev
                    
                if ann_return<(self.reward_by_value_list[0]*(1-TOLERANCE_RANGE)):
                    self.reward -= abs(ann_return-self.reward_by_value_list[0]*(1-TOLERANCE_RANGE))
                if self.mdd>(self.reward_by_value_list[1]):
                    self.reward -= abs(self.mdd-self.reward_by_value_list[1])
                if stdev>(self.stdev_tolerance_range):
                    self.reward -= abs(self.week_std-self.stdev_tolerance_range)

            elif self.reward_fuction_type==4: # best A/B/C  + uncertainty
                if self.reward_by=='reward':
                    self.reward = ann_return  - sharpness 

                elif self.reward_by=='mdd':
                    self.reward = self.mdd*(-1)  - sharpness 

                elif self.reward_by=='1mstdev':
                    self.reward = stdev*(-1) - sharpness 
            else:
                self.reward = ann_return

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        # load states
        self.covs = self.data['cov_list'].values[0]
        self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
        self.portfolio_value = self.initial_amount
        #self.cost = 0
        #self.trades = 0
        self.terminal = False 
        self.portfolio_return_memory = [0]
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        self.date_memory=[self.data.date.unique()[0]] 
        return self.state
    
    def render(self, mode='human'):
        return self.state
        
    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator/denominator
        return softmax_output

    
    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        #print(len(date_list))
        #print(len(asset_list))
        df_account_value = pd.DataFrame({'date':date_list,'daily_return':portfolio_return})
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']
        
        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
#         df_actions.columns = self.data.tic.values+self.data.tic.values
        df_actions.index = df_date.date
        #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions
    
    def save_weight_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']
        
        action_list = self.weights_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        seed = 100
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs