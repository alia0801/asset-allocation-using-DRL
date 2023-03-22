
import numpy as np
import pandas as pd
import math
import random
import torch
from config import *
from myEnv import StockPortfolioEnv

from finrl.apps import config
from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer, data_split
# from finrl.finrl_meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.drl_agents.stablebaselines3.models import DRLAgent
from finrl.finrl_meta.data_processors.processor_yahoofinance import YahooFinanceProcessor

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import datetime
import statistics

import sys
sys.path.append("../FinRL-Library")

# reward_by_list=['reward','mdd','1mstdev']
# tolerance_range = 0.1 

# get_data_train_start = '2008-01-01' # org_train_start - 1year
# org_train_start = '2009-01-01'
# org_train_end = '2016-01-01'
# get_data_trade_start = '2015-01-01' # org_trade_start - 1year
# org_trade_start = '2016-01-01' 
# org_trade_end = '2021-12-31'
# org_initial_amount = 1000000



def set_seed(seed_value):
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

set_seed(100)

def state_augmentation(df):

    usecols = ['Date', 'Close']
    VIX = pd.read_csv('./input_features/VIX.csv', usecols=usecols)
    VIX.columns = ['Date', 'Value']
    
    usecols = ['Date', 'Close']
    SPY = pd.read_csv('./input_features/SPY.csv', usecols=usecols)
    SPY.columns = ['Date', 'Value']

    usecols = ['Date', 'Close']
    MOV = pd.read_csv('./input_features/MOVE.csv', usecols=usecols)
    MOV.columns = ['Date', 'Value']

    usecols = ['Date', 'ra_bex']
    RAI = pd.read_csv('./input_features/ra_bex.csv', usecols=usecols)
    RAI.columns = ['Date', 'Value']
    RAI['Date'] = pd.to_datetime(RAI['Date'], format='%Y%m%d')

    usecols = ['Date', 'unc_bex_A']
    UNC = pd.read_csv('./input_features/unc_bex_A.csv', usecols=usecols)
    UNC.columns = ['Date', 'Value']
    UNC['Date'] = pd.to_datetime(UNC['Date'], format='%Y%m%d')

    col_name=df.columns.tolist()
    col_name.insert(df.shape[1],'MOV')
    col_name.insert(df.shape[1]+1,'RAI')
    col_name.insert(df.shape[1]+2,'UNC')
    col_name.insert(df.shape[1]+3,'SPY')
    col_name.insert(df.shape[1]+4,'VIX')
    df=df.reindex(columns=col_name)

    for index, row in df.iterrows():
        finding_date = df.loc[index, 'date']
        try:
            VIX_value = VIX[VIX['Date'] == finding_date].iloc[0]['Value']
        except Exception as e:
            pass
            # print(0, finding_date, e)
        try:
            SPY_value = SPY[SPY['Date'] == finding_date].iloc[0]['Value']
        except Exception as e:
            pass
            # print(0, finding_date, e)
        try:
            Mov_value = MOV[MOV['Date'] == finding_date].iloc[0]['Value']
        except Exception as e:
            pass
            # print(1, finding_date, e)
        try:
            RAI_value = RAI[RAI['Date'] == finding_date].iloc[0]['Value']
        except Exception as e:
            pass
            # print(2, finding_date, e)
        try:
            UNC_value = UNC[UNC['Date'] == finding_date].iloc[0]['Value']
        except Exception as e:
            pass
            # print(3, finding_date, e)
        
        df.loc[index, 'SPY'] = SPY_value
        df.loc[index, 'MOV'] = Mov_value
        df.loc[index, 'RAI'] = RAI_value
        df.loc[index, 'UNC'] = UNC_value
        df.loc[index, 'VIX'] = VIX_value
    
    df.to_csv('./input_features/testing_augmentation.csv')
    return df

def get_data(etfs,start,end):#org_trade_start,org_trade_end
    dp = YahooFinanceProcessor()
    df = dp.download_data(start_date = start,#'2009-01-01', #'2001-01-01'
                     end_date = end,#'2021-12-31',
                     ticker_list = etfs, time_interval='1D')
    fe = FeatureEngineer(
                    use_technical_indicator=True,
                    use_vix=False,
                    use_turbulence=False,
                    user_defined_feature = False)

    df = fe.preprocess_data(df)
    df = state_augmentation(df)
    print(df.head())
    
    # add covariance matrix as states
    df=df.sort_values(['date','tic'],ignore_index=True)
    df.index = df.date.factorize()[0]
    
    print(df.head())

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
    print('targets:',df.tic.unique())
    print(df.head())
    return df

def train_rl(df,start_date,end_date,stdev_tolerance_range,initial_amount=ORG_INIT_AMOUNT,reward_by='reward',reward_by_value_list=[0,0,0],reward_fuction_type=3):

    
    train = data_split(df, start_date,end_date)
    stock_dimension = len(train.tic.unique())
    state_space = stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    
    env_kwargs = {
        "hmax": initial_amount, 
        "initial_amount": initial_amount, 
        "transaction_cost_pct": 0.001, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
        "action_space": stock_dimension*2, 
        "reward_scaling": 1e-4,
        "reward_by":reward_by,
        "stdev_tolerance_range":stdev_tolerance_range,
        "reward_by_value_list":reward_by_value_list,
        "reward_fuction_type":reward_fuction_type
    }
    
    e_train_gym = StockPortfolioEnv(df = train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    
    agent = DRLAgent(env = env_train)
    PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.005,
        "learning_rate": 0.0001,
        "batch_size": 128,
    }
    model_ppo = agent.get_model("ppo",seed=100,model_kwargs = PPO_PARAMS)
#     enable_dropout(model_ppo)
    trained_ppo = agent.train_model(model=model_ppo, 
                             tb_log_name='ppo',
                             total_timesteps=100000)
#     trained_ppo.save('trained_models/trained_ppo.zip')
    return trained_ppo

def trade_rl(df,start_date,end_date,trained_model,stdev_tolerance_range,initial_amount=ORG_INIT_AMOUNT,reward_by='reward',reward_by_value_list=[0,0,0],reward_fuction_type=3):
    trade = data_split(df,start_date,end_date)
    stock_dimension = len(trade.tic.unique())
    state_space = stock_dimension
    env_kwargs = {
        "hmax": initial_amount, 
        "initial_amount": initial_amount, 
        "transaction_cost_pct": 0.001, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
        "action_space": stock_dimension*2, 
        "reward_scaling": 1e-4,
        "stdev_tolerance_range":stdev_tolerance_range,
        "reward_by":reward_by,
        "reward_by_value_list":reward_by_value_list,
        "reward_fuction_type":reward_fuction_type
    }
    e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs)
    df_daily_return, df_actions, df_weights = DRLAgent.DRL_prediction(model=trained_model, environment = e_trade_gym)
#     print(df_actions)
#     print(df_weights)
#     print(df_daily_return)
    return df_daily_return, df_actions, df_weights #df_actions

def get_ABC(input_df):
    input_df = input_df.reset_index()

    # cal week ann std
    input_df_cp = input_df.copy(deep=True)
    sub_df = input_df_cp[input_df_cp['index']%5==0].reset_index(drop=True)
    sub_df['w_return'] = 0 
    for i in range(len(sub_df)-1):
        sub_df.loc[i+1,'w_return'] = (sub_df['Close'][i+1] - sub_df['Close'][i])/sub_df['Close'][i]
    week_ann_stdev = statistics.stdev(sub_df['w_return'])* math.pow( 52, 0.5 )
    
    input_df['day_return'] = 0 
    for i in range(len(input_df)-1):
        input_df.loc[i+1,'day_return'] = (input_df['Close'][i+1] - input_df['Close'][i])/input_df['Close'][i]
    
    # cal mdd
    input_df = input_df.fillna(0)
    input_df['max']=0
    s1 = input_df['Close']
    for i in range(len(input_df)):
        input_df.loc[i,'max'] = s1[0:i+1].max() 
    input_df['dd'] = 0
    input_df['dd'] = 1-(input_df['Close']/input_df['max'])
    mdd = input_df['dd'].max()

    # cal ann stdev
    input_df['total_value'] = ORG_INIT_AMOUNT
    for i in range(1,len(input_df)):
        input_df.loc[i,'total_value'] = input_df['total_value'][i-1]*(input_df['day_return'][i]+1)
    ann_stdev = statistics.stdev(input_df['day_return'])* math.pow( 252, 0.5 )

    # cal ann reward
    ann_reward = (input_df['total_value'][len(input_df)-1]/input_df['total_value'][0])**(252/len(input_df))-1

    return ann_reward,ann_stdev,mdd,week_ann_stdev

def get_comb_ABC(comb,start,end,w=None):
    if w is None:
        w = [1/len(comb)]*len(comb)
    close_data = pd.read_csv('./price_indicator_data/all_etf_close_2001_2022_all.csv')
    close_data.drop('Unnamed: 0', inplace=True, axis=1)
    col = comb.copy()
    col.insert(0,'Date')
    sub_df = close_data[col]
    
    start_list = start.split('-')
    start_list2 = [str(int(i)) for i in start_list]
    start = '/'.join(start_list2)
    
    end_list = end.split('-')
    end_list2 = [str(int(i)) for i in end_list]
    end = '/'.join(end_list2)
     
    
    sub_df = sub_df[(sub_df['Date']>=start)& (sub_df['Date']<=end) ]
    
    sub_df = sub_df.reset_index(drop=True)
    for i in range(len(comb)):
        name = comb[i]
        sub_df[name+'_d_return'] = (sub_df[name] - sub_df[name].shift(1))/sub_df[name].shift(1)
        sub_df[name+'_money'] = ORG_INIT_AMOUNT*w[i]
        for i in range(1,len(sub_df)):
            sub_df.loc[i,name+'_money'] = sub_df[name+'_money'][i-1]*(1+sub_df[name+'_d_return'][i])
    sub_df['Close'] = 0
    for name in comb:
        sub_df['Close']+=sub_df[name+'_money']

    sub_df = sub_df.reset_index(drop=True)
    
    ann_reward,ann_stdev,mdd,week_ann_stdev = get_ABC(sub_df)
    return ann_reward,ann_stdev,mdd,week_ann_stdev

def get_avg_ABC(comb,start,end):
    ann_reward,ann_stdev,mdd,week_ann_stdev = get_comb_ABC(comb,start,end)
    return ann_reward,ann_stdev,mdd,week_ann_stdev

def get_all_A(start,end,all_etf):
    
    close_data = pd.read_csv('./price_indicator_data/all_etf_close_2001_2022_all.csv')
    close_data.drop('Unnamed: 0', inplace=True, axis=1)
    col = all_etf.copy()
    col.insert(0,'Date')
    sub_df = close_data[col]
    
    start_list = start.split('-')
    start_list2 = [str(int(i)) for i in start_list]
    start = '/'.join(start_list2)
    
    end_list = end.split('-')
    end_list2 = [str(int(i)) for i in end_list]
    end = '/'.join(end_list2)
     
    sub_df = sub_df[(sub_df['Date']>=start)& (sub_df['Date']<=end)] 
    
    sub_df = sub_df.reset_index(drop=True)
    
    sub_df.drop('Date', inplace=True, axis=1)
    
    sub_df_T = sub_df.T
    sub_df_T_col = list(sub_df_T.columns)    
    sub_df_T['reward'] = (sub_df_T[sub_df_T_col[-1]]/sub_df_T[sub_df_T_col[0]])**(252/len(sub_df_T_col))-1

    return sub_df_T[ ['reward']]

def get_uncertainty_thresh(org_etfs,trained_model,org_df,reward_by,reward_by_value_list,reward_fuction_type,stdev_tolerance_range):
    _, train_df_actions,_ = trade_rl(org_df,ORG_TRAIN_START,ORG_TRAIN_END,trained_model,stdev_tolerance_range,ORG_INIT_AMOUNT,reward_by,reward_by_value_list,reward_fuction_type)
    
    std_col_name = []
    mean_col_name = []
    for i in range(len(org_etfs)):
        mean_col_name.append(i)
        std_col_name.append(i+len(org_etfs))
    stdev_df = train_df_actions[std_col_name]

    sharpness_record = []
    for d in stdev_df.T:
        sharpness = np.sqrt(np.mean(stdev_df.T[d].values ** 2))
        if np.isnan(sharpness):
            continue
        else:
            sharpness_record.append(sharpness)
    sig = statistics.stdev(sharpness_record)
    mu = sum(sharpness_record)/len(sharpness_record)
    thresh = mu+sig*2
    # print(thresh) 
    return thresh

def detect_uncertainty(org_etfs,trained_model,org_df,reward_by,reward_by_value_list,thresh,reward_fuction_type,stdev_tolerance_range):
    now_etfs = org_df.tic.unique()
    print(org_df.head())
    
    std_col_name = []
    mean_col_name = []
    for i in range(len(org_etfs)):
        mean_col_name.append(i)
        std_col_name.append(i+len(org_etfs))
    
    trade_df_daily_return, trade_df_actions,_ = trade_rl(org_df,ORG_TRADE_START,ORG_TRADE_END,trained_model,stdev_tolerance_range,ORG_INIT_AMOUNT,reward_by,reward_by_value_list,reward_fuction_type)
    
    print(trade_df_actions.head())
    
    mean_df = trade_df_actions[mean_col_name]
    stdev_df = trade_df_actions[std_col_name]
    
    detect = []
    sharpness_record = []
    for d in stdev_df.T:
        sharpness = np.sqrt(np.mean(stdev_df.T[d].values ** 2))
        if np.isnan(sharpness):
            continue
        sharpness_record.append(sharpness)
        if sharpness>thresh:
            max_stdev=-1
            idx=-1
            count=0
            for etf in stdev_df.columns:
                stdev = stdev_df[etf][d]
                mean = mean_df[etf-len(org_etfs)][d]
                if stdev>max_stdev:
                    max_stdev = stdev
                    idx = etf
                    count=0
                elif stdev==max_stdev:
                    count+=1
            if count>0:
                min_mean = 100
                idx=-1
                for etf in stdev_df.columns:
                    stdev = stdev_df[etf][d]
                    mean = mean_df[etf-len(org_etfs)][d]
                    if mean<min_mean and stdev==max_stdev:
                        min_mean = mean
                        idx = etf
                print(min_mean,max_stdev)
            detect.append([d,now_etfs[idx-len(org_etfs)],stdev_df[idx][d],mean_df[idx-len(org_etfs)][d]])
    print(detect)
    plt.plot(sharpness_record)
    plt.show()
    print(sum(sharpness_record)/len(sharpness_record),statistics.stdev(sharpness_record))
    
    return detect,trade_df_actions,trade_df_daily_return

def org_detect_func(txt_path,org_etfs,reward_by_i,reward_by_value_list,reward_fuction_type,stdev_tolerance_range):
    org_df = get_data(org_etfs,GET_DATA_TRAIN_START,ORG_TRADE_END)
    now_etfs = org_df.tic.unique()
    models = []
    detects = []
    thresh_record = []
    i=reward_by_i
    print(i)
    trained_model = train_rl(org_df,ORG_TRAIN_START,ORG_TRAIN_END,stdev_tolerance_range,ORG_INIT_AMOUNT,REWARD_BY_LIST[i],reward_by_value_list,reward_fuction_type)
    models.append(trained_model)

    uncertainty_thresh = get_uncertainty_thresh(org_etfs,trained_model,org_df,REWARD_BY_LIST[i],reward_by_value_list,reward_fuction_type,stdev_tolerance_range)
    detect,df_concat,trade_df_daily_return = detect_uncertainty(org_etfs,trained_model,org_df,REWARD_BY_LIST[i],reward_by_value_list,uncertainty_thresh,reward_fuction_type,stdev_tolerance_range)
    detects.append(detect)
    thresh_record.append(uncertainty_thresh)
    print("detect in org_detect_func")
    print(detect)
    print(detects)
    
    if len(detect)>0:

        ddate = detect[0][0]
        dtarget = detect[0][1]
    #        print(df_concat)
        textfile = open(txt_path+"detect_record/"+REWARD_BY_LIST[i]+"/mean_stdev.txt", "w")
        textfile.write('abnormal'+'\t'+ddate+'\t'+dtarget+"\n")

        df_concat_new = df_concat.reset_index()
        df_concat_cut = df_concat_new[df_concat_new['date']==detect[0][0]]
    #        print(df_concat_cut)
        for col in range(len(org_etfs)):
            stdev = df_concat_cut[col+len(org_etfs)].values[0]
            mean = df_concat_cut[col].values[0]
            print(col,now_etfs[col],stdev,mean)
            textfile.write(now_etfs[col]+'\t'+str(stdev)+'\t'+str(mean)+"\n")
        textfile.close()
    else:
        textfile = open(txt_path+"detect_record/"+REWARD_BY_LIST[i]+"/mean_stdev.txt", "w")
        textfile.write('all dates are normal ! \n')

    return models,detects,trade_df_daily_return,thresh_record   
  
def find_new_target(txt_path,stdev_tolerance_range,trade_dates,org_etfs,old_etfs,detect,detect_latest,trained_model,last_detect_date,reward_by,reward_by_value_list,org_trade_reward,org_trade_stdev,org_trade_mdd,uncertainty_thresh,all_etf,reward_fuction_type):

    change_date = detect_latest[0]
    change_etf = detect_latest[1]
    change_etf_idx = old_etfs.index(change_etf)
    change_date_org = str(change_date)
    
    change2org=False
    done=False
    if old_etfs!=org_etfs:
        for i in range(-1,len(org_etfs)):
            etf = org_etfs[i]
            now_etfs = old_etfs.copy()
            if etf not in now_etfs or i==-1:#org誰被換掉
                if i==-1:
                    now_etfs= org_etfs.copy()
                    print('change2org comb')
                else:
                    print(etf+'not in comb')
                    for j in range(len(now_etfs)):
                        if now_etfs[j] not in org_etfs:
                            now_etfs[j] = etf
                            break
                print(now_etfs)
                dt_tmp = last_detect_date.split('-')
                dt = datetime.date(int(dt_tmp[0]),int(dt_tmp[1]),int(dt_tmp[2]))
                time_del = datetime.timedelta(days=8) 
                dt = dt+time_del
                dt_tmp = change_date.split('-')
                dt_stop = datetime.date(int(dt_tmp[0]),int(dt_tmp[1]),int(dt_tmp[2]))
                print(dt,dt_stop)
                while True:
                    if dt==dt_stop:
                        break
                    if dt>dt_stop:
                        break
                    if str(dt) not in trade_dates:
                        dt = dt+time_del
                        continue
                    avg_reward,avg_std,avg_mdd,avg_stdev_week = get_avg_ABC(now_etfs,ORG_TRADE_START,str(dt))
                    if avg_reward>org_trade_reward and avg_std<org_trade_stdev and avg_mdd<org_trade_mdd:    
                        print(dt,'abc ok')
                        change2org=True
                        break
                    dt = dt+time_del
                    print(dt)
                    if dt>datetime.date(2022,1,1):
                        break
            if change2org:
                change_date = str(dt)
                now_df = get_data(now_etfs,GET_DATA_TRADE_START,ORG_TRADE_END)
                if len(now_df.tic.unique())!=len(org_etfs):
                    print(now_df.tic.unique())
                    continue
                now_etfs = list(now_df.tic.unique())
                detect_new,df_concat,_ = detect_uncertainty(org_etfs,trained_model,now_df,reward_by,reward_by_value_list,uncertainty_thresh,reward_fuction_type,stdev_tolerance_range)
                
                std_col_name = []
                for i in range(len(org_etfs)):
                    std_col_name.append(i+len(org_etfs))
    
                stdev_df = df_concat[std_col_name]
                sharpness = np.sqrt(np.mean(stdev_df.T[change_date].values ** 2))
                
                if sharpness<uncertainty_thresh:
                    print('change2org ok!')
                    done=True
                    
                    detect_latest = []
                    dt_tmp = change_date.split('-')
                    dt = datetime.date(int(dt_tmp[0]),int(dt_tmp[1]),int(dt_tmp[2]))
                    for i in range(len(detect_new)):
                        dt_tmp = detect_new[i][0].split('-')
                        dt_new = datetime.date(int(dt_tmp[0]),int(dt_tmp[1]),int(dt_tmp[2]))
            
                        if dt_new>dt: #dt_new較晚
                            detect_latest = detect_new[i]
                            print('detect_latest',detect_latest)
                            break
                    break
    if done:
        print(now_etfs,'successed','normal@',change_date)
    
        textfile = open(txt_path+"detect_record/"+reward_by+"/mean_stdev.txt", "a")
        textfile.write('normal'+'\t'+change_date+"\n")
    
        df_concat_new = df_concat.reset_index()
        df_concat_cut = df_concat_new[df_concat_new['date']==change_date]
        for col in range(len(org_etfs)):
            stdev = df_concat_cut[col+len(org_etfs)].values[0]
            mean = df_concat_cut[col].values[0]
            print(col,now_etfs[col],stdev,mean)
            textfile.write(now_etfs[col]+'\t'+str(stdev)+'\t'+str(mean)+"\n")
    
        try:
            textfile.write('abnormal'+'\t'+detect_latest[0]+'\t'+detect_latest[1]+"\n")
            
            df_concat_new = df_concat.reset_index()
            df_concat_cut = df_concat_new[df_concat_new['date']==detect_latest[0]]
                
            for col in range(len(org_etfs)):
                stdev = df_concat_cut[col+len(org_etfs)].values[0]
                mean = df_concat_cut[col].values[0]
                print(col,now_etfs[col],stdev,mean)
                textfile.write(now_etfs[col]+'\t'+str(stdev)+'\t'+str(mean)+"\n")
            textfile.close()
        except:
            textfile.close()
        
        return now_etfs,detect_new,detect_latest,change_date
    
    cal_reward=get_all_A(ORG_TRADE_START,change_date,all_etf)
    target_reward=org_trade_reward*(1-TOLERANCE_RANGE)
    in_comb_etf=old_etfs.copy()
    in_comb_etf.remove(change_etf)
    r = []
    for etf in in_comb_etf:
        r.append(cal_reward['reward'][etf])
    thresh = (len(in_comb_etf)+1)*target_reward-sum(r)
    candidate_etfs = list(cal_reward[cal_reward['reward']>thresh].index)
    random.shuffle(candidate_etfs)
    print(len(candidate_etfs))
    
    change_success = False
    all_ok_comb=[]
    all_ok_comb_abc = []
    detect_result = []
    count_etf = 0
    for etf in candidate_etfs:
        count_etf+=1
        now_etfs = old_etfs.copy()
        change_date = change_date_org
        if etf not in now_etfs:
            now_etfs[change_etf_idx] = etf
        
        print(count_etf,now_etfs)
        avg_reward,avg_std,avg_mdd,avg_stdev_week = get_avg_ABC(now_etfs,ORG_TRADE_START,change_date)
        if avg_reward>(org_trade_reward*(1-TOLERANCE_RANGE)) and avg_stdev_week<(stdev_tolerance_range) and avg_mdd<(org_trade_mdd*(1.05)):
            print('abc ok')
        else:
            continue
        now_df = get_data(now_etfs,GET_DATA_TRADE_START,ORG_TRADE_END)
        
        if len(now_df.tic.unique())!=len(org_etfs):
            print(now_df.tic.unique())
            continue
        now_etfs = list(now_df.tic.unique())
        detect_new,df_concat,_ = detect_uncertainty(org_etfs,trained_model,now_df,reward_by,reward_by_value_list,uncertainty_thresh,reward_fuction_type,stdev_tolerance_range)
        
        std_col_name = []
        for i in range(len(org_etfs)):
            std_col_name.append(i+len(org_etfs))
    
        stdev_df = df_concat[std_col_name]
        sharpness = np.sqrt(np.mean(stdev_df.T[change_date].values ** 2))
                
        if sharpness<uncertainty_thresh:
            all_ok_comb.append(now_etfs)
            all_ok_comb_abc.append([avg_reward,avg_std,avg_mdd,sharpness])
            detect_result.append([detect_new,df_concat])
            
        if len(all_ok_comb)>=10:
            break
        
    if len(all_ok_comb)>0:
        print('candidate comb # :',len(all_ok_comb))
        all_ok_comb_not_sky = []
        for i in range(len(all_ok_comb)):
            for j in range(len(all_ok_comb)):
                if i>=j:
                    continue
                comb1 = all_ok_comb[i]
                comb2 = all_ok_comb[j]
                a1,b1,c1,s1 = all_ok_comb_abc[i]
                a2,b2,c2,s2 = all_ok_comb_abc[j]
                if a1>a2 and b1<b2 and c1<c2 and s1<s2:
                    if comb2 not in all_ok_comb_not_sky:
                        all_ok_comb_not_sky.append(comb2)
                elif a1<a2 and b1>b2 and c1>c2 and s1>s2:
                    if comb1 not in all_ok_comb_not_sky:
                        all_ok_comb_not_sky.append(comb1)
        all_ok_comb_skyline = all_ok_comb.copy()
        for del_comb in all_ok_comb_not_sky:
            all_ok_comb_skyline.remove(del_comb)
            
        if len(all_ok_comb_skyline)==1:
            now_etfs = all_ok_comb_skyline[0]
        else:
            idx = random.randint(0,len(all_ok_comb_skyline)-1)
            now_etfs = all_ok_comb_skyline[idx]
        idx = all_ok_comb.index(now_etfs)
        detect_new,df_concat = detect_result[idx]
        
    else:
        now_etfs = old_etfs.copy()
        etf = 'SHV'
        if 'SHV' not in now_etfs:
            now_etfs[change_etf_idx] = 'SHV'
        elif 'BIL' not in now_etfs:
            now_etfs[change_etf_idx] = 'BIL'
        elif 'PVI' not in now_etfs:
            now_etfs[change_etf_idx] = 'PVI'
        
        now_df = get_data(now_etfs,GET_DATA_TRADE_START,ORG_TRADE_END)
        now_etfs = list(now_df.tic.unique())
        detect_new,df_concat,_ = detect_uncertainty(org_etfs,trained_model,now_df,reward_by,reward_by_value_list,uncertainty_thresh,reward_fuction_type,stdev_tolerance_range)
           
    detect_latest = []
    dt_tmp = change_date.split('-')
    dt = datetime.date(int(dt_tmp[0]),int(dt_tmp[1]),int(dt_tmp[2]))
    for i in range(len(detect_new)):
        dt_tmp = detect_new[i][0].split('-')
        dt_new = datetime.date(int(dt_tmp[0]),int(dt_tmp[1]),int(dt_tmp[2]))
        if dt_new>dt: #dt_new較晚
            detect_latest = detect_new[i]
            print('detect_latest',detect_latest)
            break
            
    print(now_etfs,'successed','normal@',change_date)
    
    textfile = open(txt_path+"detect_record/"+reward_by+"/mean_stdev.txt", "a")
    textfile.write('normal'+'\t'+change_date+"\n")
    
    df_concat_new = df_concat.reset_index()
    df_concat_cut = df_concat_new[df_concat_new['date']==change_date]
    for col in range(len(org_etfs)):
        stdev = df_concat_cut[col+len(org_etfs)].values[0]
        mean = df_concat_cut[col].values[0]
        print(col,now_etfs[col],stdev,mean)
        textfile.write(now_etfs[col]+'\t'+str(stdev)+'\t'+str(mean)+"\n")
    
    try:
        textfile.write('abnormal'+'\t'+detect_latest[0]+'\t'+detect_latest[1]+"\n")
            
        df_concat_new = df_concat.reset_index()
        df_concat_cut = df_concat_new[df_concat_new['date']==detect_latest[0]]
                
        for col in range(len(org_etfs)):
            stdev = df_concat_cut[col+len(org_etfs)].values[0]
            mean = df_concat_cut[col].values[0]
            print(col,now_etfs[col],stdev,mean)
            textfile.write(now_etfs[col]+'\t'+str(stdev)+'\t'+str(mean)+"\n")
        textfile.close()
    except:
        textfile.close()
            
    return now_etfs,detect_new,detect_latest,change_date


def get_stdev_tolerance_range(train_stdev_week):
    if train_stdev_week<0.005:
        stdev_tolerance_range = 0.005
    elif train_stdev_week<0.02:
        stdev_tolerance_range = 0.02
    elif train_stdev_week<0.05:
        stdev_tolerance_range = 0.05
    elif train_stdev_week<0.1:
        stdev_tolerance_range = 0.1
    elif train_stdev_week<0.2:
        stdev_tolerance_range = 0.2
    elif train_stdev_week<0.25:
        stdev_tolerance_range = 0.25
    else:
        stdev_tolerance_range = train_stdev_week*(1+TOLERANCE_RANGE)
    return stdev_tolerance_range