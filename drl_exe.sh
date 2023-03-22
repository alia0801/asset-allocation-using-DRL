#!/bin/bash
reward_fuction_type=1
BASEDIR=$(dirname "$0")
echo "$BASEDIR"
for comb_num in {0..7}
do 
    /home/alia880801/anaconda3/bin/python3.7 $BASEDIR/baseline_RL.py  --comb=$comb_num --rft=$reward_fuction_type --arg_from=True 
done