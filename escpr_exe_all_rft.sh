#!/bin/bash

BASEDIR=$(dirname "$0")
echo "$BASEDIR"
for reward_fuction_type in {1..4}
do
    for comb_num in {0..7}
    do 
        echo "$comb_num"
        echo "start abc"
        /home/alia880801/anaconda3/bin/python3.7 $BASEDIR/escpr_agentA.py --comb=$comb_num --rft=$reward_fuction_type --arg_from=True && echo "Job A is done" &
        /home/alia880801/anaconda3/bin/python3.7 $BASEDIR/escpr_agentB.py --comb=$comb_num --rft=$reward_fuction_type --arg_from=True && echo "Job B is done" &
        /home/alia880801/anaconda3/bin/python3.7 $BASEDIR/escpr_agentC.py --comb=$comb_num --rft=$reward_fuction_type --arg_from=True && echo "Job C is done" &
        wait

        echo "start final trade"

        /home/alia880801/anaconda3/bin/python3.7 $BASEDIR/escpr_final_trade.py  --comb=$comb_num --rft=$reward_fuction_type --arg_from=True 
    done
done