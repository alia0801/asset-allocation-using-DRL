#!/usr/bin/env python
# coding: utf-8

# ## Relative Momentum Trading Strategy
# 參考[How to Create a Relative Momentum Trading Strategy in Python](https://www.the905guy.ca/article/how-to-create-a-relative-momentum-trading-strategy-in-python?fbclid=IwAR1fX2udgpI2SJ9omlEyK81nQJgmFMQVU4XfRzbbG7hFCrPVgYcLP4YARXc)

import pandas as pd
import yfinance as yf
pd.core.common.is_list_like = pd.api.types.is_list_like
import os
from util import *
from config import *
import argparse

# ## Mean Reversion
# 參考[A Simple Mean Reversion System in Simple Python Code](https://gist.github.com/AnthonyFJGarner/ccd23f0e9d46214612f59c7b92a82149?fbclid=IwAR009-5uFVZAMK2klvNQWrCA8Le7FX9sULY8YA1nLz0KMkb0_kzZRi7oaJs)

# ### Mean Reversion - Portfolio

def trade(stock, length):

    temp_dict = {}
    # If window length is 0, algorithm doesn't make sense, so exit
    if length == 0:
        return 0

    # Compute rolling means and rolling standard deviation
    #sma and lma are filters to prevent taking long or short positions against the longer term trend
    rolling_window = stock.Adj_Close.rolling(window=length)
    mu = rolling_window.mean()
    sma = stock.Adj_Close.rolling(window=length*1).mean()
    lma = stock.Adj_Close.rolling(window=length * 10).mean()
    std = rolling_window.std()

    #If you don't use a maximum position size the positions will keep on pyramidding.
    #Set max_position to a high number (1000?) to disable this parameter
    #Need to beware of unintended leverage
    max_position = 1
    percent_per_trade = 1.0

    #Slippage and commission adjustment  - simply reduces equity by a percentage guess
    # a setting of 1 means no slippage, a setting of 0.999 gives 0.1% slippage
    slippage_adj = 1

    # Compute the z-scores for each day using the historical data up to that day
    zscores = (stock.Adj_Close - mu) / std

    # Simulate trading
    # Start with your chosen starting capital and no positions
    money = 1000.00
    position_count = 0

    for i, row in enumerate(stock.itertuples(), 0):

        #set up position size so that each position is a fixed position of your account equity
        equity = money + (stock.Adj_Close[i] * position_count)
        if equity > 0:
            fixed_frac = (equity * percent_per_trade) / stock.Adj_Close[i]
        else:
            fixed_frac = 0
        fixed_frac = int(round(fixed_frac))

        #exit all positions if zscore flips from positive to negative or vice versa without going through
        #the neutral zone
        if i > 0:
            if (zscores[i - 1] > 0.5
                    and zscores[i] < -0.5) or (zscores[i - 1] < -0.5
                                               and zscores[i] > 0.5):

                if position_count > 0:
                    money += position_count * stock.Adj_Close[i] * slippage_adj
                elif position_count < 0:
                    money += position_count * stock.Adj_Close[i] * (
                        1 / slippage_adj)
                position_count = 0

        # Sell short if the z-score is > 1 and if the longer term trend is negative
        if (zscores[i] > 1) & (position_count > max_position * -1) & (sma[i] <
                                                                      lma[i]):

            position_count -= fixed_frac
            money += fixed_frac * stock.Adj_Close[i] * slippage_adj

        # Buy long if the z-score is < 1 and the longer term trend is positive
        elif zscores[i] < -1 and position_count < max_position and sma[i] > lma[i]:

            position_count += fixed_frac
            money -= fixed_frac * stock.Adj_Close[i] * (1 / slippage_adj)

        # Clear positions if the z-score between -.5 and .5
        elif abs(zscores[i]) < 0.5:
            #money += position_count * stock.Adj_Close[i]
            if position_count > 0:
                money += position_count * stock.Adj_Close[i] * slippage_adj
            elif position_count < 0:
                money += position_count * stock.Adj_Close[i] * (
                    1 / slippage_adj)
            position_count = 0

        #fill dictionary with the trading results.
        temp_dict[stock.Date[i]] = [
            stock.Adj_Open[i], stock.Adj_Close[i], mu[i], std[i], zscores[i],
            money, position_count, fixed_frac, sma[i], lma[i]
        ]
    #create a dataframe to return for use in calculating and charting the trading results
    pr = pd.DataFrame(data=temp_dict).T
    pr.index.name = 'Date'
    pr.index = pd.to_datetime(pr.index)
    pr.columns = [
        'Open', 'Close', 'mu', 'std', 'zscores', 'money', 'position_count',
        'fixed_frac', 'sma', 'lma'
    ]
    pr['equity'] = pr.money + (pr.Close * pr.position_count)
    #
    return pr

def get_1_stock_simulate(stock_name,start_date,end_date,moving_average):
    data = yf.download(stock_name, start=start_date, end=end_date)
    data.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)
    data['Adj_Open']=data.Open*(data.Adj_Close/data.Close)
    pricing = data.reset_index()[['Date','Adj_Open', 'Adj_Close']]
    stock=pricing.copy()
    stock.Adj_Close=stock.Adj_Close.shift(1)
    profit = trade(stock, moving_average)
    return profit

def mean_rev(comb,start_date,end_date,moving_average,weights=None):

    all_simulate = []
    for stock_name in comb:
        profit = get_1_stock_simulate(stock_name,start_date,end_date,moving_average)
        all_simulate.append(profit)

    if weights is None:
        weights = [1/len(comb)]*len(comb)
    
    all_series = []
    for i in range(len(all_simulate)):
        series = profit[['equity']].copy()
        series *= weights[i]
        all_series.append(series)

    final_money = pd.concat(all_series,axis=1)
    final_money.columns = comb


    final_money_sum = final_money.T.sum()
    final_money_sum = pd.DataFrame(final_money_sum)
    final_money_sum.columns = ['equity']


    final_money_sum = final_money_sum[final_money_sum['equity']!=0]
    
    return final_money_sum

def buy_and_hold(start_date,end_date,comb,weights):
    if weights is None:
        weights = [1/len(comb)]*len(comb)
    df_close = pd.read_csv('./price_indicator_data/all_etf_close_2001_2022_all.csv')
    df_close.drop('Unnamed: 0', inplace=True, axis=1)
    col = comb.copy()
    col.insert(0,'Date')
    df = df_close[col]

    start_list = start_date.split('-')
    start_list2 = [str(int(i)) for i in start_list]
    start = '/'.join(start_list2)
    
    end_list = end_date.split('-')
    end_list2 = [str(int(i)) for i in end_list]
    end = '/'.join(end_list2)


    df = df[(df['Date']>=start)& (df['Date']<=end) ]
    df = df.reset_index()

    for i in range(len(comb)):
        name = comb[i]
        df[name+'_d_return'] = (df[name] - df[name].shift(1))/df[name].shift(1)
        df[name+'_money'] = 1000000*weights[i]
        for i in range(1,len(df)):
            df.loc[i,name+'_money'] = df[name+'_money'][i-1]*(1+df[name+'_d_return'][i])
    df['Close'] = 0
    for name in comb:
        df['Close']+=df[name+'_money']
    df = df.reset_index(drop=True)

    df['daily_return'] = 0 
    for i in range(len(df)-1):
        df.loc[i+1,'daily_return'] = (df['Close'][i+1] - df['Close'][i])/df['Close'][i]

    # df[['Date','daily_return']].to_csv(path+'buyandhold_daily_return.csv')
    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--comb', type=int, default=0, help='comb code')
    parser.add_argument('--arg_from', type=bool, default=False, help='wheather arg from command')
    args = parser.parse_args()
    
    if args.arg_from == False: # get arg from parm_escpr.txt
        print('get arg from parm_traditional.txt')

        filename = './parm/parm_traditional.txt'
        parm = []  
        f = open(filename)
        for line in f:
            parm.append(line[:-1])
        print(parm)
        comb_num = int(parm[0])
    else:
        comb_num=args.comb

    print('comb_num=',comb_num)
    
    comb_key=list(COMBS.keys())[comb_num]
    comb=COMBS[comb_key]['etfs']
    weights=COMBS[comb_key]['weights']
    print('comb name:',comb_key)
    print('comb etfs:',comb)
    print('comb weights:',weights)
    
    start_date = ORG_TRADE_START#'2016-01-01' 
    end_date = ORG_TRADE_END#'2022-01-01'
    
    path = './result_baseline_traditional/'+str(comb_num)+'/'
    if not os.path.exists(path):
        os.makedirs(path)


    # Mean Reversion
    moving_average=10
    final_money_sum_org = mean_rev(comb,start_date,end_date,moving_average,weights)

    final_money_sum = final_money_sum_org.reset_index()
    final_money_sum['daily_return'] = 0 
    for i in range(len(final_money_sum)-1):
        final_money_sum.loc[i+1,'daily_return'] = (final_money_sum['equity'][i+1] - final_money_sum['equity'][i])/final_money_sum['equity'][i]


    final_money_sum[['Date','daily_return']].to_csv(path+'meanreversion_daily_return.csv')


    # Buy and Hold

    df = buy_and_hold(start_date,end_date,comb,weights)
    df[['Date','daily_return']].to_csv(path+'buyandhold_daily_return.csv')

