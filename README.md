# ESCPR: Deep Reinforcement Learning Ensemble Strategy for Customized Portfolio Rebalancing

![](https://i.imgur.com/POJPaSm.png)

## Requirements
This is a Pytorch implementation of ESCPR architecture as described in the paper ESCPR: Deep Reinforcement Learning Ensemble Strategy for Customized Portfolio Rebalancing
* python 3
* see `requirements.txt`

## Run the code
### Generate daily return record of each method
#### ESCPR
* Modify `parm/parm_escpr.txt` to set which rewards function to use for which portfolio rebalancing.
    * The first parameter is the portfolio code, and the secode code is reward function type.
* Run `escpr_agentA.py`, `escpr_agentB.py`, `escpr_agentC.py`, respectively.
    * Each of the 3 agents is trained during the training period and rebalanced during the trading period, recording the daily asset weights and details of asset changes.
    * They can be executed at the same time to speed up the efficiency.
* Run `escpr_final_trade.py`
    * It combines the results produced by the 3 agents and generates a daily return record, save at `'./results/type_<reward function type>/<portfolio code>/`.
#### Traditional methods
* Modify `parm/parm_traditional.txt` to set which portfolio rebalancing
    * The parameter is the portfolio code.
* Run `baseline_traditional.py`
    * It will generate a daily return record, save at `'./result_baseline_traditional/<portfolio code>/`.
#### DRL methods
* Modify `parm/parm_drl.txt` to set which rewards function to use for which portfolio rebalancing.
    * The first parameter is the portfolio code, and the secode code is reward function type.
* Run `baseline_RL.py`
    * It will generate a daily return record and the daily asset weights, save at `'./result_baseline_rl/<portfolio code>/`.
### Evaluation
* Modify `comb_num_list` at line 451 in `evaluation.py` to set which portfolios are to be evaluated.
* Run `evaluation.py`, and get quantitative results in the folder `./evaluation_result/q_result/`.
* Other results are also saved in the folder `./evaluation_result/` .