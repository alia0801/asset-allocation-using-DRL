#!/bin/bash
BASEDIR=$(dirname "$0")
echo "$BASEDIR"

echo "start abc"
/home/alia880801/anaconda3/bin/python3.7 $BASEDIR/escpr_agentA.py && echo "Job A is done" &
/home/alia880801/anaconda3/bin/python3.7 $BASEDIR/escpr_agentB.py && echo "Job B is done" &
/home/alia880801/anaconda3/bin/python3.7 $BASEDIR/escpr_agentC.py && echo "Job C is done" &
wait
echo "start final trade"
/home/alia880801/anaconda3/bin/python3.7 $BASEDIR/escpr_final_trade.py
