# %%
import os
import pandas as pd
import numpy as np
import statistics
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from datetime import datetime
import datetime
from util import *
from config import *
# %%
def get_df_ABC(df_org,init_money=ORG_INIT_AMOUNT):
    df_org = df_org.reset_index()

    df_org['Close'] = init_money
    for i in range(len(df_org)-1):
        df_org.loc[i+1,'Close'] = (df_org['daily_return'][i]+1)*df_org['Close'][i]
    
    df_cp = df_org.copy(deep=True)
    sub_df = df_cp[df_cp['index']%5==0].reset_index(drop=True)
    sub_df['w_return'] = 0 
    for i in range(len(sub_df)-1):
        sub_df.loc[i+1,'w_return'] = (sub_df['Close'][i+1] - sub_df['Close'][i])/sub_df['Close'][i]
    try:
        stdev_week = statistics.stdev(sub_df['w_return'])* math.pow( 52, 0.5 )
    except:
        stdev_week=0.00001

    df_org = df_org.fillna(0)
    df_org['max']=0
    s1 = df_org['Close']
    for i in range(len(df_org)):
        df_org.loc[i,'max'] = s1[0:i+1].max() 
    
    df_org['dd'] = 0
    df_org['dd'] = 1-(df_org['Close']/df_org['max'])
    
    mdd = df_org['dd'].max()

    df_org['total_value'] = ORG_INIT_AMOUNT
    for i in range(1,len(df_org)):
        df_org.loc[i,'total_value'] = df_org['total_value'][i-1]*(df_org['daily_return'][i]+1)
    try:
        ann_stdev = statistics.stdev(df_org['daily_return'])* math.pow( 252, 0.5 )
    except:
        ann_stdev = 0.00001

    ann_reward = (df_org['total_value'][len(df_org)-1]/df_org['total_value'][0])**(252/len(df_org))-1
    
    df_org['ann_reward'] = 0
    for i in range(1,len(df_org)):
        df_org.loc[i,'ann_reward'] = (df_org['total_value'][i]/df_org['total_value'][0])**(252/i)-1

    return ann_reward,ann_stdev,mdd,stdev_week,df_org

def get_df(final_reward,cut_date,days):

    idx = list(final_reward.index[final_reward['date'] == cut_date])[0]
    if idx-days>=0:
        idx_start = idx-days
    else:
        idx_start = 0
    if idx+days<len(final_reward):
        idx_end = idx+days
    else:
        idx_end = len(final_reward)-1
    df1 = final_reward[idx_start:idx]
    df2 = final_reward[idx:idx_end]
    return df1,df2

# %%

def get_every_month_ann_reward(df_processed):
    ann_reward_list = []
    for i in range(len(df_processed)):
        if i%21==0:
            ann_reward_list.append(df_processed['ann_reward'][i])
        elif i==len(df_processed)-1:
            ann_reward_list.append(df_processed['ann_reward'][i])
    return ann_reward_list

def plot_all_figure(df_list,legends,save_path,org_train_stdev,org_train_mdd,org_train_reward):
    ann_reward = []
    money_sim = []
    rewards = []
    risks = []
    mdds = []
    tmp_df_list = []
    date_list = []
    plt.figure(figsize=(16,8))
    for i in range(len(df_list)):
        df = df_list[i]
        a,b,c,stdev_week,df_processed = get_df_ABC(df)
        ann_reward_list = get_every_month_ann_reward(df_processed)
        ann_reward.append(ann_reward_list)        
        money_sim.append(df_processed['Close'].values)
        rewards.append(a)
        risks.append(b)
        mdds.append(c)
        tmp_df_list.append([legends[i],a,b,c,stdev_week])
        date_list.append(pd.to_datetime(df_processed['date'], format='%Y/%m/%d')) 
        plt.plot(ann_reward_list,label=legends[i], color=COLORS[i])

    lg = plt.legend(prop={'size': 7},loc='lower right')
    plt.savefig(save_path+'ann_reward.png')
    # plt.show()
    plt.close()
    plt.figure(figsize=(16,8))
    for i in range(len(df_list)):
        plt.plot(date_list[i],money_sim[i],label=legends[i], color=COLORS[i])
        
    lg = plt.legend(prop={'size': 7},loc='upper left')
    plt.savefig(save_path+'money_sim.png')
    # plt.show()
    plt.close()

    p=[]
    ax = plt.subplot(projection='3d')
    legends_new=legends.copy()
    for i in range(len(rewards)):
        ttt_fig = ax.scatter([risks[i]],[mdds[i]],[rewards[i]],  marker='o', s=40 ,c=COLORS[i])#,c=colors[i]
        p.append(ttt_fig)
    ttt_fig = ax.scatter([org_train_stdev],[org_train_mdd],[org_train_reward],  marker='o', s=40,c=COLORS[i+1])#,c=colors[i]
    p.append(ttt_fig)
    ax.set_zlabel('Annual Reward') 
    ax.set_ylabel('MDD')
    ax.set_xlabel('Annual Stdev')
    legends_new.append('target')
    print(len(p),len(legends_new))
    lg = ax.legend(p,legends_new,prop={'size': 7},bbox_to_anchor=(1.05, 1.0),loc='upper left')
    plt.savefig(save_path+'3d.png',bbox_extra_artists=(lg,),bbox_inches='tight')
    # plt.show()
    plt.close()
    
    performance_df = pd.DataFrame(tmp_df_list)
    performance_df.columns = ['method','Annual Reward','Annual Stdev','MDD','Annual Stdev (week)']
    return performance_df
# %%
def get_dd_df(df,init_money=ORG_INIT_AMOUNT):
    df = df.reset_index()

    df['Close'] = init_money
    for i in range(len(df)-1):
        df.loc[i+1,'Close'] = (df['daily_return'][i]+1)*df['Close'][i]
    
    df = df.fillna(0)
    df['max']=0
    s1 = df['Close']
    for i in range(len(df)):
        df.loc[i,'max'] = s1[0:i+1].max() 
    
    df['dd'] = 0
    df['dd'] = 1-(df['Close']/df['max'])
    return df

def plot_dd_change(df_list,legends,org_train_mdd,comb_num,save_path,comb_key):
    plt.figure(figsize=(8,4))
    for i in range(len(df_list)):
        df = get_dd_df(df_list[i])
        plt.plot(pd.to_datetime(df['date'], format='%Y/%m/%d'),df['dd'],label=legends[i], color=COLORS[i])
    plt.axhline(org_train_mdd,label='threshold',color='black')
    lg = plt.legend(prop={'size': 12},loc='upper left')
    plt.title(comb_key,fontsize=12)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.savefig(save_path+'dd_change_'+str(comb_num)+'.png')
    # plt.show()
    plt.close()


# %%
def q_result(comb_num,#comb_num_list,
            save_path='./evaluation_result/q_result/',
            tr_path='./result_baseline_traditional/',
            drl_path='./result_RL/',
            best_path='./result/type1/',
            best_close_path='./result/type2/',
            best_un_path='./result/type4/',
            escpr_path='./result/type3/'
            ):

    print('comb_num',comb_num)
    save_path += (str(comb_num)+'/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ### traditional baseline
    buy_hold = pd.read_csv(tr_path+str(comb_num)+'/buyandhold_daily_return.csv')
    mean_rev = pd.read_csv(tr_path+str(comb_num)+'/meanreversion_daily_return.csv')
    buy_hold.columns = ['Unnamed: 0', 'date', 'daily_return']
    mean_rev.columns = ['Unnamed: 0', 'date', 'daily_return']

    ### drl
    drl_ppo_best = pd.read_csv(drl_path+str(comb_num)+'/df_daily_return_ppo.csv')#[:1258]
    drl_td3_best = pd.read_csv(drl_path+str(comb_num)+'/df_daily_return_td3.csv')#[:1258]
    drl_ppo_closest = pd.read_csv(drl_path+str(comb_num)+'/df_daily_return_ppo_close.csv')
    drl_ppo_best_closest = pd.read_csv(drl_path+str(comb_num)+'/df_daily_return_ppo_opt_close.csv')
    ucl_best = pd.read_csv(best_path+str(comb_num)+'/final_reward.csv')
    ucl_best_closest = pd.read_csv(best_close_path+str(comb_num)+'/final_reward.csv')
    ucl_best_un = pd.read_csv(best_un_path+str(comb_num)+'/final_reward.csv')
    ucl_best_closest_un = pd.read_csv(escpr_path+str(comb_num)+'/final_reward.csv')
    ucl_best_closest_un_rwd = pd.read_csv(escpr_path+str(comb_num)+'/ensemble/classic_reward_all.csv')
    ucl_best_closest_un_std = pd.read_csv(escpr_path+str(comb_num)+'/ensemble/classic_1mstdev_all.csv')
    ucl_best_closest_un_mdd = pd.read_csv(escpr_path+str(comb_num)+'/ensemble/classic_mdd_all.csv')
    ucl_org_r = pd.read_csv(escpr_path+str(comb_num)+'/csv/classic_reward_00.csv')
    
    ucl_org = ucl_org_r
    df_list = [buy_hold,       mean_rev,       drl_ppo_best, drl_ppo_closest ,drl_ppo_best_closest ,drl_td3_best ,ucl_best,    ucl_best_closest, ucl_best_un,  ucl_best_closest_un, ucl_best_closest_un_rwd, ucl_best_closest_un_std, ucl_best_closest_un_mdd, ucl_org]
    legends = ['Buy and Hold','Mean Reversion',   'PPO',     'PPO close',    'PPO opt&close',   'TD3',   'ES-CPR -CR-UR','ES-CPR -UR'   , 'ES-CPR -CR', 'ES-CPR' , 'ES-CPR- 1agent-rwd'  , 'ES-CPR- 1agent-std' , 'ES-CPR- 1agent-mdd' ,'ES-CPR- no replace']#
    
    comb_key=list(COMBS.keys())[comb_num]
    org_etfs = COMBS[comb_key]['etfs'] 
    org_comb_weight = COMBS[comb_key]['weights'] 
    org_train_reward,org_train_stdev,org_train_mdd,train_week_stdev = get_comb_ABC(org_etfs,ORG_TRAIN_START,ORG_TRAIN_END, org_comb_weight)
    print(org_train_reward,org_train_stdev,org_train_mdd,train_week_stdev)
    org_trade_reward,org_trade_stdev,org_trade_mdd,trade_week_stdev = get_comb_ABC(org_etfs,ORG_TRADE_START,ORG_TRADE_END, org_comb_weight)
    print(org_trade_reward,org_trade_stdev,org_trade_mdd,trade_week_stdev)
    performance_df = plot_all_figure(df_list,legends,save_path,org_train_stdev,org_train_mdd,org_train_reward)
        
    performance_df_copy = performance_df.copy(deep=True)
    df2 = {'method': 'target', 'Annual Reward': org_train_reward, 'Annual Stdev': org_train_stdev,'MDD':org_train_mdd,'Annual Stdev (week)':train_week_stdev}
    performance_df_copy = performance_df_copy.append(df2, ignore_index = True)
    performance_df_copy['Reward_error'] = abs(performance_df_copy['Annual Reward']-org_train_reward)
    performance_df_copy['Stdev_error'] = abs(performance_df_copy['Annual Stdev']-org_train_stdev)
    performance_df_copy['MDD_error'] = abs(performance_df_copy['MDD']-org_train_mdd)
    performance_df_copy['error'] = np.sqrt(performance_df_copy['Reward_error']**2+performance_df_copy['Stdev_error']**2+performance_df_copy['MDD_error']**2)
    performance_df_copy.to_csv(save_path+'performance_df_error_rate.csv')
    print(performance_df_copy)

def casestudy_dd(comb_num_list,
            save_path='./evaluation_result/casestudy_dd/',
            tr_path='./result_baseline_traditional/',
            escpr_path='./result/type3/'
            ):
    for comb_num in comb_num_list:

        print('comb_num',comb_num)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        ### traditional baseline
        buy_hold = pd.read_csv(tr_path+str(comb_num)+'/buyandhold_daily_return.csv')
        buy_hold.columns = ['Unnamed: 0', 'date', 'daily_return']

        ### drl
        ucl_best_closest_un = pd.read_csv(escpr_path+str(comb_num)+'/final_reward.csv')

        df_list = [buy_hold,        ucl_best_closest_un]
        legends = ['Buy and Hold',    'ES-CPR' ]
        
        comb_key=list(COMBS.keys())[comb_num]

        org_etfs = COMBS[comb_key]['etfs'] 
        org_comb_weight = COMBS[comb_key]['weights'] 
        org_train_reward,org_train_stdev,org_train_mdd,train_week_stdev = get_comb_ABC(org_etfs,ORG_TRAIN_START,ORG_TRAIN_END, org_comb_weight)
        print(org_train_reward,org_train_stdev,org_train_mdd,train_week_stdev)

        plot_dd_change(df_list,legends,org_train_mdd,comb_num,save_path,comb_key)

def casestudy_dd_cal_ratio(comb_num_list):
    for comb_num in comb_num_list:

        print('comb_num',comb_num)

        ### traditional baseline
        buy_hold = pd.read_csv(tr_path+str(comb_num)+'/buyandhold_daily_return.csv')
        buy_hold.columns = ['Unnamed: 0', 'date', 'daily_return']

        ### drl
        ucl_best_closest_un = pd.read_csv(escpr_path+str(comb_num)+'/final_reward.csv')

        df_list = [buy_hold,        ucl_best_closest_un]
        legends = ['Buy and Hold',    'ES-CPR' ]#

        comb_key=list(COMBS.keys())[comb_num]

        org_etfs = COMBS[comb_key]['etfs'] 
        org_comb_weight = COMBS[comb_key]['weights'] 
        org_train_reward,org_train_stdev,org_train_mdd,train_week_stdev = get_comb_ABC(org_etfs,ORG_TRAIN_START,ORG_TRAIN_END, org_comb_weight)
        print(org_train_reward,org_train_stdev,org_train_mdd,train_week_stdev)

        for i in range(len(df_list)):
            df = get_dd_df(df_list[i])
            df['diff'] = org_train_mdd-df['dd']
            print(legends[i],len(df[df['diff']>0]),len(df[df['diff']>0])/len(df))

def plot_replace_timing(comb_num_list,
            save_path='./evaluation_result/change_timing/',
            tr_path='./result_baseline_traditional/',
            drl_path='./result_RL/',
            escpr_path='./result/type3/'
            ):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    keys = list(COMBS.keys())

    for comb_num in comb_num_list:

        print('comb_num',comb_num)
        comb_name = keys[comb_num]
        org_etfs = COMBS[comb_name]['etfs']
        print(comb_name,org_etfs)
        exp_file_path = escpr_path+str(comb_num)+'/'

        ### traditional baseline
        buy_hold = pd.read_csv(tr_path+str(comb_num)+'/buyandhold_daily_return.csv')[:-1]
        buy_hold.columns = ['Unnamed: 0', 'date', 'daily_return']

        ### drl
        drl_ppo_best = pd.read_csv(drl_path+str(comb_num)+'/df_daily_return_ppo.csv')

        ucl_best_closest = pd.read_csv(exp_file_path+'final_reward.csv')[:-1]
        ucl_org = pd.read_csv(exp_file_path+'csv/classic_reward_00.csv')[:-1]

        buy_hold['money'] = ORG_INIT_AMOUNT
        drl_ppo_best['money'] = ORG_INIT_AMOUNT
        ucl_best_closest['money'] = ORG_INIT_AMOUNT
        ucl_org['money'] = ORG_INIT_AMOUNT
        for i in range(len(drl_ppo_best)-1):
            drl_ppo_best.loc[i+1,'money'] = (drl_ppo_best['daily_return'][i]+1)*drl_ppo_best['money'][i]
            ucl_best_closest.loc[i+1,'money'] = (ucl_best_closest['daily_return'][i]+1)*ucl_best_closest['money'][i]
            ucl_org.loc[i+1,'money'] = (ucl_org['daily_return'][i]+1)*ucl_org['money'][i]
        date_list = pd.to_datetime(drl_ppo_best['date'], format='%Y/%m/%d')
        plt.figure(figsize=(8,4))
        plt.plot(date_list,drl_ppo_best['money'],label='PPO',c=COLORS[0])
        plt.plot(date_list,ucl_best_closest['money'],label='ES-CPR',c=COLORS[1])
        plt.plot(date_list,ucl_org['money'],label='ES-CPR no replace',c=COLORS[2])
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        lg = plt.legend(prop={'size': 12},loc='upper left')
        max_money = max([buy_hold['money'].max(),drl_ppo_best['money'].max(),ucl_best_closest['money'].max(),ucl_org['money'].max()])
        min_money = min([buy_hold['money'].min(),drl_ppo_best['money'].min(),ucl_best_closest['money'].min(),ucl_org['money'].min()])

        for i in range(len(REWARD_BY_LIST)):

            filename = exp_file_path+'detect_record/'+REWARD_BY_LIST[i]+'/mean_stdev.txt'#
            f = open(filename)
            flag = False
            dt_rcrd = []
            etf_rcrd = []
            tmp=[]
            for line in f:
                ttt = line[:-1]
                ttt_list = ttt.split('\t')
                if flag:
                    tmp.append(ttt_list[0])
                    if len(tmp)==len(org_etfs):
                        etf_rcrd.append(tmp)
                        tmp=[]
                if ttt_list[0]=='normal':
                    flag=True
                    tmp=[]
                    replace_dt = datetime.datetime.strptime(ttt_list[1], "%Y-%m-%d")
                    plt.vlines(replace_dt,min_money,max_money,color="gray")
                    dt_rcrd.append(replace_dt)
                if ttt_list[0]=='abnormal':
                    flag = False
        plt.title(comb_name,fontsize=12)
        plt.savefig(save_path+str(comb_num)+'.png')

def plot_bef_aft(comb_num_list,
            save_path='./evaluation_result/change_bef_aft/',
            escpr_path='./result/type3/'
            ):
    for comb_num in comb_num_list:
        print('comb_num',comb_num)
        read_filepath = escpr_path+str(comb_num)+'/'
        save_filepath = save_path+str(comb_num)+'/'
        if not os.path.exists(save_filepath):
            os.makedirs(save_filepath)

        all_record=[]
        for j in range(len(REWARD_BY_LIST)):
            reward_by = REWARD_BY_LIST[j]
            
            
            record_file = read_filepath+'detect_record/'+reward_by+'/mean_stdev.txt'
            f = open(record_file)
            flag = False
            dt_rcrd = []
            etf_rcrd = []
            tmp=[]
            for line in f:
                ttt = line[:-1]
                ttt_list = ttt.split('\t')
                if flag:
                    tmp.append(ttt_list[0])
                    if len(tmp)==4:
                        etf_rcrd.append(tmp)
                        tmp=[]
                if ttt_list[0]=='normal':
                    flag=True
                    tmp=[]
                    dt = ttt_list[1]
                    dt_rcrd.append(dt)
                    l = len(dt_rcrd)
                    if l<10:
                        i1 = '0'+str(l-1)
                        i2 = '0'+str(l)
                    elif l==10:
                        i1 = '0'+str(l-1)
                        i2 = str(l)
                    else:
                        i1 = str(l-1)
                        i2 = str(l)
        
                    df_org = pd.read_csv(read_filepath+'csv/classic_'+str(REWARD_BY_LIST[j])+'_'+i1+'.csv')
                    df_ch = pd.read_csv(read_filepath+'csv/classic_'+str(REWARD_BY_LIST[j])+'_'+i2+'.csv')
        
                    df_org_bef,df_org_aft = get_df(df_org,dt,21)
                    _,df_ch_aft = get_df(df_ch,dt,21)
                    if len(df_ch_aft)<3 or len(df_org_aft)<3:
                        continue
        
                    org_bef_A,org_bef_B,org_bef_C,org_bef_D,df_org_bef_all = get_df_ABC(df_org_bef)
                    init_money = list(df_org_bef_all['Close'].values)[-1]
                    org_aft_A,org_aft_B,org_aft_C,org_aft_D,df_org_aft_all = get_df_ABC(df_org_aft,init_money)
                    ch_aft_A,ch_aft_B,ch_aft_C,org_aft_D,df_ch_aft_all = get_df_ABC(df_ch_aft,init_money)

                    if ch_aft_A>org_aft_A and ch_aft_B<org_aft_B and ch_aft_C<org_aft_C:
                        label=1
                        print(dt,'ok')
                        title = REWARD_BY_LIST[j]+' '+dt+' ok'
                        fig_name = 'ok_'+REWARD_BY_LIST[j]+dt+'.png'
                    elif ch_aft_A<org_aft_A and ch_aft_B>org_aft_B and ch_aft_C>org_aft_C:
                        label=-1
                        print(dt,'fail')
                        title = REWARD_BY_LIST[j]+' '+dt+' fail'
                        fig_name = 'fail_'+REWARD_BY_LIST[j]+dt+'.png'
                    else:
                        label=0
                        title = REWARD_BY_LIST[j]+' '+dt
                        fig_name = 'other_'+REWARD_BY_LIST[j]+dt+'.png'
                    all_record.append([dt,label,[org_bef_A,org_bef_B,org_bef_C],[org_aft_A,org_aft_B,org_aft_C],[ch_aft_A,ch_aft_B,ch_aft_C]])
                    org_money = list(df_org_bef_all['Close'].values) + list(df_org_aft_all['Close'].values)
                    ch_money = [None]*len(df_org_bef_all)+list(df_ch_aft_all['Close'].values)
                    plt.plot(org_money,label='org')
                    plt.plot(ch_money,label='change')
                    lg = plt.legend(prop={'size': 7},loc='best')
                    plt.title(title)
                    plt.xlabel("investment days")
                    plt.ylabel("portfolio value")
                    plt.xlim([-25,70])
                    plt.savefig(save_filepath+fig_name)
                    plt.close()
                    # plt.show()
        
                if ttt_list[0]=='abnormal':
                    flag = False

if __name__ == '__main__':
    comb_num_list = [0]#,1,2,3,4,5,6,7
    tr_path='./result_baseline_traditional/'
    drl_path='./result_baseline_rl/'
    best_path='./results/type_1/'
    best_close_path='./results/type_2/'
    best_un_path='./results/type_4/'
    escpr_path='./results/type_3/'

    for comb_num in comb_num_list:
        q_result(comb_num,
                save_path='./evaluation_result/q_result/',
                tr_path=tr_path,
                drl_path=drl_path,
                best_path=best_path,
                best_close_path=best_close_path,
                best_un_path=best_un_path,
                escpr_path=escpr_path
                )

    casestudy_dd(comb_num_list,
            save_path='./evaluation_result/casestudy_dd/',
            tr_path=tr_path,
            escpr_path=escpr_path
            )

    plot_replace_timing(comb_num_list,
            save_path='./evaluation_result/change_timing/',
            tr_path=tr_path,
            drl_path=drl_path,
            escpr_path=escpr_path
            )

    plot_bef_aft(comb_num_list,
            save_path='./evaluation_result/change_bef_aft/',
            escpr_path=escpr_path
            )
