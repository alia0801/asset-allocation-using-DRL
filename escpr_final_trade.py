import pandas as pd
from util import *
from config import *
from stable_baselines3 import PPO
# from myEnv import StockPortfolioEnv
import sys
import os
import argparse
sys.path.append("../FinRL-Library")

set_seed(100)

if __name__ == '__main__':
    print('final trade')

    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--comb', type=int, default=0, help='comb code')
    parser.add_argument('--rft', type=int, default=0, help='reward_fuction_type')
    parser.add_argument('--arg_from', type=bool, default=False, help='wheather arg from command')
    args = parser.parse_args()
    
    if args.arg_from == False: # get arg from parm_escpr.txt
        print('get arg from parm_escpr.txt')

        filename = './parm/parm_escpr.txt'
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

    # reward_fuction_type = int(parm[1])  # 1:bestABC, 2:bestA/B/C+closestABC, 3:bestA/B/C+closestABC+uncertainty ,4:bestA/B/C+uncertainty
    comb_key=list(COMBS.keys())[comb_num]
    org_etfs=COMBS[comb_key]['etfs']
    org_comb_weight=COMBS[comb_key]['weights']
    print('comb name:',comb_key)
    print('comb etfs:',org_etfs)
    print('comb weights:',org_comb_weight)

    save_path = './results/type_'+str(reward_fuction_type)+'/'+str(comb_num)+'/'
    if not os.path.exists(save_path+'csv/'):
        os.makedirs(save_path+'csv/')
        print('mkdir:',save_path)

    etf_record = [org_etfs]

    org_train_reward,org_train_stdev,org_train_mdd,train_stdev_week = get_comb_ABC(org_etfs,ORG_TRAIN_START,ORG_TRAIN_END, org_comb_weight)
    org_trade_reward,org_trade_stdev,org_trade_mdd,trade_stdev_week = get_comb_ABC(org_etfs,ORG_TRADE_START,ORG_TRADE_END, org_comb_weight)
    print(org_train_reward,org_train_stdev,org_train_mdd,train_stdev_week)
    print(org_trade_reward,org_trade_stdev,org_trade_mdd,trade_stdev_week)

    stdev_tolerance_range = get_stdev_tolerance_range(train_stdev_week)

    reward_by_value_list=[ org_train_reward,org_train_mdd,org_train_stdev ]      

    total_record = []
    for i in range(len(REWARD_BY_LIST)):

        filename = save_path+'detect_record/'+REWARD_BY_LIST[i]+'/mean_stdev.txt'#
        f = open(filename)
        flag = False
        dt_rcrd = []
        etf_rcrd = []
        tmp=[]
        for line in f:
            ttt = line[:-1]
            ttt_list = ttt.split('\t')
            print(ttt_list)
            if flag:
                tmp.append(ttt_list[0])
                print(tmp)
                if len(tmp)==len(org_etfs):
                    etf_rcrd.append(tmp)
                    tmp=[]
            if ttt_list[0]=='normal':
                flag=True
                tmp=[]
                dt_rcrd.append(ttt_list[1])
            if ttt_list[0]=='abnormal':
                flag = False
        print(dt_rcrd)
        print(etf_rcrd)
        print(len(dt_rcrd),len(etf_rcrd))
        total_record.append([dt_rcrd,etf_rcrd])

    all_new_trade_df_daily_return = []
    for j in range(len(REWARD_BY_LIST)):
        print(REWARD_BY_LIST[j])
        dt_rcrd = total_record[j][0]
        etf_rcrd = total_record[j][1]
        trained_model = PPO.load(save_path+'trained_models/'+REWARD_BY_LIST[j]+'.zip')#models[j]

        trade_action_list = []
        trade_weight_list = []
        trade_return_list = []
        etf_rcrd.insert(0,org_etfs)
        for i in range(len(etf_rcrd)):#etf_record
            now_etfs = etf_rcrd[i]
            now_df = get_data(now_etfs,GET_DATA_TRADE_START,ORG_TRADE_END)
            print(now_df.tic.unique())
            trade_df_daily_return, trade_df_actions, trade_df_weights = trade_rl(now_df,ORG_TRADE_START,ORG_TRADE_END,trained_model,stdev_tolerance_range,ORG_INIT_AMOUNT,REWARD_BY_LIST[j],reward_by_value_list)
            trade_action_list.append(trade_df_actions)
            trade_weight_list.append(trade_df_weights)
            trade_return_list.append(trade_df_daily_return)
            if i<10:
                iiiii = '0'+str(i)
            else:
                iiiii = str(i)
            trade_df_daily_return.to_csv(save_path+'csv/classic_'+str(REWARD_BY_LIST[j])+'_'+iiiii+'.csv')

        if len(etf_rcrd)==1:
            all_new_trade_df_daily_return.append(trade_return_list[0])
        else:
            concate = []
            detect_date_record_new = dt_rcrd.copy()#detect_date_record
            detect_date_record_new.insert(0, '2016-01-04')#"2016-01-04"trade_dates[0]
            for i in range(len(detect_date_record_new)):
                if i==0:
                    trade_df_daily_return_org = trade_return_list[i]
                if i!=len(detect_date_record_new)-1:
                    trade_start = detect_date_record_new[i] 
                    trade_end = detect_date_record_new[i+1]
                    trade_df_daily_return = trade_return_list[i]
                    start_idx = trade_df_daily_return[trade_df_daily_return['date'] == trade_start].index.tolist()[0]
                    end_idx = trade_df_daily_return[trade_df_daily_return['date'] == trade_end].index.tolist()[0]
                    df_toconcate = trade_df_daily_return[start_idx:end_idx]
                else:
                    trade_start = detect_date_record_new[i] 
                    trade_df_daily_return = trade_return_list[i]
                    start_idx = trade_df_daily_return[trade_df_daily_return['date'] == trade_start].index.tolist()[0]
                    df_toconcate = trade_df_daily_return[start_idx:]
                concate.append(df_toconcate)
            new_trade_df_daily_return = pd.concat(concate,axis=0,ignore_index = True)
            #new_trade_df_daily_return
            all_new_trade_df_daily_return.append(new_trade_df_daily_return)

    reward_tmp = []
    for tmp_df_org in all_new_trade_df_daily_return:
        tmp_df = tmp_df_org.copy()
        tmp_df['daily_return']+=1
        reward_tmp.append(tmp_df)
    df_concat = pd.concat(reward_tmp)
    by_row_index = df_concat.groupby(df_concat.index)
    final_reward = by_row_index.mean()-1
    final_reward = pd.concat([tmp_df['date'],final_reward],axis=1)

    final_reward.to_csv(save_path+'final_reward.csv')
