#!/bin/bash
reward_fuction_type=0
BASEDIR=$(dirname "$0")
echo "$BASEDIR"
for comb_num in {0..7}
do 
    /home/alia880801/anaconda3/bin/python3.7 $BASEDIR/baseline_traditional.py  --comb=$comb_num --arg_from=True 
done