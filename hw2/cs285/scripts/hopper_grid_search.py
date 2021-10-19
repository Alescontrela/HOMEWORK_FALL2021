import os

lambds = [0, 0.95, 0.99, 1]

for lambd in lambds: 
    comm = "python cs285/scripts/run_hw2.py \--env_name Hopper-v2 --ep_len 1000 --discount 0.99 -n 300 -l 2 -s 32 -b 2000 -lr 0.001 \--reward_to_go --nn_baseline --action_noise_std 0.5 --gae_lambda {lambd} \--exp_name q5_b2000_r0.001_lambda{lambd}".format(lambd=lambd)
    os.system(comm)