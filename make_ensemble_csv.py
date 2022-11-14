# %%
import pandas as pd
import os
from util import *
from config import *

# %%

filename = './parm/parm_escpr.txt'
parm = []  
f = open(filename)
for line in f:
    parm.append(line[:-1])
print(parm)
comb_num = int(parm[0])

file_path = './results/type_3/'
# comb_num = 10


comb_key=list(COMBS.keys())[comb_num]
org_etfs=COMBS[comb_key]['etfs']
os.mkdir(file_path + str(comb_num)+'/ensemble')

REWARD_BY_LIST=['reward','mdd','1mstdev']
total_record = []
for i in range(len(REWARD_BY_LIST)):

    filename = file_path + str(comb_num)+ '/detect_record/'+REWARD_BY_LIST[i]+'/mean_stdev.txt'#
    f = open(filename)
    flag = False
    dt_rcrd = []
    etf_rcrd = []
    tmp=[]
    for line in f:
#     print(line[:-1])
        ttt = line[:-1]
        ttt_list = ttt.split('\t')
        # print(ttt_list)
        if flag:
            tmp.append(ttt_list[0])
            # print(tmp)
            if len(tmp)==len(org_etfs):
                etf_rcrd.append(tmp)
                tmp=[]
        if ttt_list[0]=='normal':
#         print('normal')
            flag=True
            tmp=[]
            dt_rcrd.append(ttt_list[1])
        if ttt_list[0]=='abnormal':
            flag = False
    # print(dt_rcrd)
    # print(etf_rcrd)
    print(len(dt_rcrd),len(etf_rcrd))
    total_record.append([dt_rcrd,etf_rcrd])

all_new_trade_df_daily_return = []
for j in range(len(REWARD_BY_LIST)):
    # print(reward_by_list[j])
    dt_rcrd = total_record[j][0]
    etf_rcrd = total_record[j][1]
    etf_rcrd.insert(0,org_etfs)

    trade_return_list = []
    for i in range(len(etf_rcrd)):
        if i<10:
            iiiii = '0'+str(i)
        else:
            iiiii = str(i)
        trade_df_daily_return = pd.read_csv(file_path + str(comb_num)+'/csv/classic_'+str(REWARD_BY_LIST[j])+'_'+iiiii+'.csv')
        trade_return_list.append(trade_df_daily_return)
    print(len(trade_return_list))

    concate = []
    detect_date_record_new = dt_rcrd.copy()#detect_date_record
    detect_date_record_new.insert(0, '2016-01-04')#"2016-01-04"trade_dates[0]
    for i in range(len(detect_date_record_new)):
        # print('i',i)
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
    new_trade_df_daily_return.to_csv( file_path + str(comb_num)+'/ensemble/classic_'+str(REWARD_BY_LIST[j])+'_all.csv')
    # all_new_trade_df_daily_return.append(new_trade_df_daily_return)