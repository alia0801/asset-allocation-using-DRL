import random
from util import *
from config import *
from stable_baselines3 import PPO
# from myEnv import StockPortfolioEnv
import sys
import os
sys.path.append("../FinRL-Library")

set_seed(100)

if __name__ == '__main__':
    reward_by_i = 1 # 0:reward 1:mdd 2:std

    filename = './parm/parm_escpr.txt'
    parm = []  
    f = open(filename)
    for line in f:
        parm.append(line[:-1])
    print(parm)

    # reward_by_list=['reward','mdd','1mstdev']
    reward_fuction_type = int(parm[1])  # 1:bestABC, 2:bestA/B/C+closestABC, 3:bestA/B/C+closestABC+uncertainty ,4:bestA/B/C+uncertainty
    comb_key=list(COMBS.keys())[int(parm[0])]
    org_etfs=COMBS[comb_key]['etfs']
    org_comb_weight=COMBS[comb_key]['weights']
    print('comb name:',comb_key)
    print('comb etfs:',org_etfs)
    print('comb weights:',org_comb_weight)

    save_path = './results/type_'+parm[1]+'/'+parm[0]+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path+'detect_record/'):
        os.makedirs(save_path+'detect_record/')
    if not os.path.exists(save_path+'detect_record/'+REWARD_BY_LIST[reward_by_i]+'/'):
        os.makedirs(save_path+'detect_record/'+REWARD_BY_LIST[reward_by_i]+'/')
    if not os.path.exists(save_path+'trained_models/'):
        os.makedirs(save_path+'trained_models/')

    etf_record = [org_etfs]

    org_train_reward,org_train_stdev,org_train_mdd,train_stdev_week = get_comb_ABC(org_etfs,ORG_TRAIN_START,ORG_TRAIN_END, org_comb_weight)
    org_trade_reward,org_trade_stdev,org_trade_mdd,trade_stdev_week = get_comb_ABC(org_etfs,ORG_TRADE_START,ORG_TRADE_END, org_comb_weight)
    # print(org_train_reward,org_train_stdev,org_train_mdd,train_stdev_week)
    # print(org_trade_reward,org_trade_stdev,org_trade_mdd,trade_stdev_week)

    stdev_tolerance_range = get_stdev_tolerance_range(train_stdev_week)

    all_etf = []  
    f = open('./etf_list/sky_etf.txt')
    for line in f:
        all_etf.append(line[:-1])
    for etf in org_etfs:
        if etf not in all_etf:
            all_etf.append(etf)
    if 'SHV' not in all_etf:
        all_etf.append('SHV')
    if 'BIL' not in all_etf:
        all_etf.append('BIL')
    if 'PVI' not in all_etf:
        all_etf.append('PVI')

    print(len(all_etf))
    print(all_etf)

    reward_by_value_list=[ org_train_reward,org_train_mdd,org_train_stdev ]
    org_df = get_data(org_etfs,GET_DATA_TRAIN_START,ORG_TRADE_END)

    models,org_detect,trade_df_daily_return_org,uncertainty_thresh = org_detect_func(save_path,org_etfs,reward_by_i,reward_by_value_list,reward_fuction_type,stdev_tolerance_range)
    trade_dates = list(trade_df_daily_return_org['date'])
    print(org_detect)

    # first detect
    detect = org_detect[0].copy()
    if len(detect)==0:
        textfile = open(save_path+"detect_record/"+REWARD_BY_LIST[reward_by_i]+"/detect_record.txt", "w")
        textfile.write('all dates are normal ! \n')
        # continue
    textfile = open(save_path+"detect_record/"+REWARD_BY_LIST[reward_by_i]+"/detect_record.txt", "w")
    textfile.write("00"+'\n')
    for dtct_r in detect:
        for kkk in range(len(dtct_r)):
            element = dtct_r[kkk]
            textfile.write(str(element))
            if kkk<len(dtct_r)-1:
                textfile.write("\t")
            else:
                textfile.write("\n")
    textfile.close()
    try:
        textfile = open(save_path+"detect_record/"+REWARD_BY_LIST[reward_by_i]+"/detect_date_record.txt", "w")
        textfile.write(detect[0][0]+'\n')
        textfile.close()
    except:
        pass

    # detect & replace & save
    trained_model = models[0]
    trained_model.save(save_path+'trained_models/'+REWARD_BY_LIST[reward_by_i]+'.zip')
    trained_model = PPO.load(save_path+'trained_models/'+REWARD_BY_LIST[reward_by_i]+'.zip')
    old_etfs = org_etfs.copy()
    detect = org_detect[0].copy()
    if len(detect)==0:
        pass
    else:
        detect_latest = detect[0]
        last_detect_date = trade_dates[0] #'2016-01-04'# test first date
        # print(detect)
        count=1
        while True:
            random.shuffle(all_etf)
            print('old_etfs:',old_etfs)
            print('detect',detect)
            now_etfs,detect_new,detect_latest,last_detect_date = find_new_target(save_path,stdev_tolerance_range,trade_dates,org_etfs,old_etfs,detect,detect_latest,trained_model,last_detect_date,REWARD_BY_LIST[reward_by_i],reward_by_value_list,org_train_reward,org_train_stdev,org_train_mdd,uncertainty_thresh,all_etf,reward_fuction_type)
            etf_record.append(now_etfs) 

            old_etfs = now_etfs.copy()
            detect = detect_new 

            textfile = open(save_path+"detect_record/"+REWARD_BY_LIST[reward_by_i]+"/detect_record.txt", "a")
            if count<10:
                textfile.write('0'+str(count)+'\n')
            else:
                textfile.write(str(count)+'\n')
            for dtct_r in detect:
                for kkk in range(len(dtct_r)):
                    element = dtct_r[kkk]
                    textfile.write(str(element))
                    if kkk<len(dtct_r)-1:
                        textfile.write("\t")
                    else:
                        textfile.write("\n")
            textfile.close()

            count+=1
            if detect_latest==[]:
                break   

            textfile = open(save_path+"detect_record/"+REWARD_BY_LIST[reward_by_i]+"/detect_date_record.txt", "a")
            textfile.write(detect_latest[0]+'\n')
            textfile.close()
    
