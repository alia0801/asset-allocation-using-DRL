import pandas as pd
import numpy as np
from config import *
from util import *

from finrl.apps import config
from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.finrl_meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.drl_agents.stablebaselines3.models import DRLAgent
from finrl.finrl_meta.data_processors.processor_yahoofinance import YahooFinanceProcessor
import sys
import torch
import random
sys.path.append("../FinRL-Library")
import statistics
import os


from gym.utils import seeding
import gym
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

import argparse

if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)

def set_seed(seed_value):
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

set_seed(100)


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


        self.return_list = []
        self.pre_value = initial_amount
        self.reward=0


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
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
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
            weights = self.softmax_normalization(actions) 
            self.actions_memory.append(weights)
            last_day_memory = self.data
            #load next state
            self.day += 1
            self.data = self.df.loc[self.day,:]
            self.covs = self.data['cov_list'].values[0]
            self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
            #print(self.state)
            # calcualte portfolio return
            # individual stocks' return * weight
            portfolio_return = sum(((self.data.close.values / last_day_memory.close.values)-1)*weights)
            # update portfolio value
            new_portfolio_value = self.portfolio_value*(1+portfolio_return)
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])            
            self.asset_memory.append(new_portfolio_value)

            # the reward is the new portfolio value or end portfolo value
            # self.reward = new_portfolio_value 
            # self.reward = (new_portfolio_value-self.initial_amount)/self.initial_amount
            now_return = (new_portfolio_value-self.initial_amount)/self.initial_amount
            ann_return = (1+now_return) ** (252/self.day) -1
            if self.max_portfolio_value <  new_portfolio_value :
                self.max_portfolio_value = new_portfolio_value
            now_dd = 1-(new_portfolio_value/self.max_portfolio_value)
            if self.mdd < now_dd:
                self.mdd = now_dd    
            rrr = (new_portfolio_value-self.pre_value)/self.pre_value
            self.return_list.append(rrr)
            if len(self.return_list)<21:
                stdev = 0.00001
            else:
                self.return_list.pop(0)
                stdev = statistics.stdev(self.return_list)

            
            if REWARD_FUNCTION_TYPE ==1: # best A/B
                self.reward = ann_return/stdev-stdev
            elif REWARD_FUNCTION_TYPE ==2: # closest A/B/C
                self.reward = (abs(ann_return-TARGET_A) + abs(stdev-TARGET_B) + abs(self.mdd-TARGET_C)) * (-1)
            elif REWARD_FUNCTION_TYPE ==3: # closest A/B/C +  best A/B
                self.reward = ann_return/stdev-stdev + (abs(ann_return-TARGET_A) + abs(stdev-TARGET_B) + abs(self.mdd-TARGET_C)) * (-1)

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
    
    def save_weight_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']
        
        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']
        
        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--comb', type=int, default=0, help='comb code')
    parser.add_argument('--rft', type=int, default=0, help='reward_fuction_type')
    parser.add_argument('--arg_from', type=bool, default=False, help='wheather arg from command')
    args = parser.parse_args()
    
    if args.arg_from == False: # get arg from parm_escpr.txt
        print('get arg from parm_drl.txt')

        filename = './parm/parm_drl.txt'
        parm = []  
        f = open(filename)
        for line in f:
            parm.append(line[:-1])
        print(parm)
        comb_num = int(parm[0])
        reward_fuction_type = int(parm[1])
    else:
        comb_num=args.comb
        reward_fuction_type=args.rft

    print('comb_num=',comb_num)
    print('reward_fuction_type=',reward_fuction_type)


    save_path = 'result_baseline_rl/'+str(comb_num)+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path+'trained_models/'):
        os.makedirs(save_path+'trained_models/')

    REWARD_FUNCTION_TYPE = reward_fuction_type
    comb_key=list(COMBS.keys())[comb_num]
    org_etfs=COMBS[comb_key]['etfs']
    org_comb_weight=COMBS[comb_key]['weights']
    print('comb name:',comb_key)
    print('comb etfs:',org_etfs)
    print('comb weights:',org_comb_weight)

    start_date = ORG_TRAIN_START#'2009-01-01'
    trade_date = ORG_TRADE_START#'2016-01-01'
    end_date = ORG_TRADE_END#'2021-12-31'

    TARGET_A,_,TARGET_C,TARGET_B = get_comb_ABC(org_etfs,start_date,trade_date, org_comb_weight)

    dp = YahooFinanceProcessor()
    df = dp.download_data(start_date = start_date,
                         end_date = end_date,
                         ticker_list = org_etfs, time_interval='1D')

    fe = FeatureEngineer(
                        use_technical_indicator=True,
                        use_vix=False,
                        use_turbulence=False,
                        user_defined_feature = False)

    df = fe.preprocess_data(df)
    # df = state_augmentation(df)

    """
    Add covariance matrix as states
    """
    # add covariance matrix as states
    df=df.sort_values(['date','tic'],ignore_index=True)
    df.index = df.date.factorize()[0]

    cov_list = []
    return_list = []

    # look back is one year
    lookback=252
    for i in range(lookback,len(df.index.unique())):
      data_lookback = df.loc[i-lookback:i,:]
      price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
      return_lookback = price_lookback.pct_change().dropna()
      return_list.append(return_lookback)

      covs = return_lookback.cov().values 
      cov_list.append(covs)

    
    df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
    df = df.merge(df_cov, on='date')
    df = df.sort_values(['date','tic']).reset_index(drop=True)

    train = data_split(df, start_date ,trade_date)

    stock_dimension = len(train.tic.unique())
    state_space = stock_dimension
    print("Stock Dimension: {stock_dimension}, State Space: {state_space}")
    env_kwargs = {
        "hmax": 1000000, 
        "initial_amount": 1000000, 
        "transaction_cost_pct": 0.001, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
        "action_space": stock_dimension, 
        "reward_scaling": 1

    }

    e_train_gym = StockPortfolioEnv(df = train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    print(type(env_train))


    # initialize

    ######################################################################
    agent = DRLAgent(env = env_train)
    PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.005,
        "learning_rate": 0.0001,
        "batch_size": 128,
    }
    model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)
    trained_ppo = agent.train_model(model=model_ppo, 
                                 tb_log_name='ppo',
                                 total_timesteps=100000)
    trained_ppo.save(save_path+'trained_models/trained_ppo.zip')

    ######################################################################

    agent = DRLAgent(env = env_train)
    TD3_PARAMS = {"batch_size": 100, 
                  "buffer_size": 1000000, 
                  "learning_rate": 0.001}

    model_td3 = agent.get_model("td3",model_kwargs = TD3_PARAMS)
    trained_td3 = agent.train_model(model=model_td3, 
                                 tb_log_name='td3',
                                 total_timesteps=50000)

    trained_td3.save(save_path+'trained_models/trained_td3.zip')

    ######################################################################

    trade = data_split(df,trade_date, end_date)
    e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs)

    df_daily_return, df_actions,_ = DRLAgent.DRL_prediction(model=trained_ppo,environment = e_trade_gym)
    if REWARD_FUNCTION_TYPE==1:
        filename_return = 'df_daily_return_ppo.csv'
        filename_action = 'df_actions_ppo.csv'
    elif REWARD_FUNCTION_TYPE==2:
        filename_return = 'df_daily_return_ppo_close.csv'
        filename_action = 'df_actions_ppo_close.csv'
    elif REWARD_FUNCTION_TYPE==3:
        filename_return = 'df_daily_return_ppo_opt_close.csv'
        filename_action = 'df_actions_ppo_opt_close.csv'
    df_daily_return.to_csv(save_path+filename_return)
    df_actions.to_csv(save_path+filename_action)

    df_daily_return, df_actions,_ = DRLAgent.DRL_prediction(model=trained_td3,environment = e_trade_gym)
    if REWARD_FUNCTION_TYPE==1:
        filename_return = 'df_daily_return_td3.csv'
        filename_action = 'df_actions_td3.csv'
    elif REWARD_FUNCTION_TYPE==2:
        filename_return = 'df_daily_return_td3_close.csv'
        filename_action = 'df_actions_td3_close.csv'
    elif REWARD_FUNCTION_TYPE==3:
        filename_return = 'df_daily_return_td3_opt_close.csv'
        filename_action = 'df_actions_td3_opt_close.csv'
    df_daily_return.to_csv(save_path+filename_return)
    df_actions.to_csv(save_path+filename_action)

    ######################################################################s