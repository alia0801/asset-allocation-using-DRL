# ESCPR: Deep Reinforcement Learning Ensemble Strategy for Customized Portfolio Rebalancing

![](https://i.imgur.com/POJPaSm.png)

## Requirements
This is a Pytorch implementation of ESCPR architecture as described in the paper ESCPR: Deep Reinforcement Learning Ensemble Strategy for Customized Portfolio Rebalancing
* python 3
* see `requirements.txt`

## Run the code
### Generate daily return record of each method
#### ESCPR
* Modify python path in `escpr_exe_1comb.sh` and `escpr_exe_all.sh`
* If only run one comb
    * Modify `parm/parm_escpr.txt` to set which rewards function to use for which portfolio rebalancing.
        * The first parameter is the portfolio code, and the secode code is reward function type.
    * Run `escpr_exe_1comb.sh`
* If run all comb
    * Modify reward_fuction_type in `escpr_exe_all.sh`
    * Run `escpr_exe_all.sh`
* Each of the 3 agents is trained during the training period and rebalanced during the trading period, recording thdaily asset weights and details of asset changes.
* They will be executed at the same time to speed up the efficiency.
* It will combines the results produced by the 3 agents and generates a daily return record, save at `resulttype_<reward function type>/<portfolio code>/`.

#### Traditional methods
* If only run one comb
    * Modify `parm/parm_traditional.txt` to set which portfolio rebalancing
        * The parameter is the portfolio code.
    * Run `baseline_traditional.py`
        * It will generate a daily return record, save at `result_baseline_traditional/<portfolio code>/`.
* If run all comb
    * Run `traditional_exe.sh`
#### DRL methods
* If only run one comb
    * Modify `parm/parm_drl.txt` to set which rewards function to use for which portfolio rebalancing.
        * The first parameter is the portfolio code, and the secode code is reward function type.
    * Run `baseline_RL.py`
        * It will generate a daily return record and the daily asset weights, save at `result_baseline_rl/<portfolio code>/`.
* If run all comb
    * Modify reward_fuction_type in `drl_exe.sh`
    * Run `drl_exe.sh`
### Evaluation
* Modify `comb_num_list` at line 451 in `evaluation.py` to set which portfolios are to be evaluated.
* Run `evaluation.py`, and get quantitative results in the folder `evaluation_result/q_result/`.
* Other results are also saved in the folder `evaluation_result/` .