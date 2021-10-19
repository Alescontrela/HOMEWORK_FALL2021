import os

batch_sizes = [10000,30000,50000]
lrs = [0.005,0.01,0.02]

for b in batch_sizes:
    for r in lrs:   
        comm = "python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \--discount 0.95 -n 100 -l 2 -s 32 -b {b} -lr {r} -rtg --nn_baseline \--exp_name q4_search_b{b}_lr{r}_rtg_nnbaselin".format(b=b, r=r)
        os.system(comm)