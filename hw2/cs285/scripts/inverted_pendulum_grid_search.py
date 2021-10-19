import os

batch_size = 200
learning_rate = 0.02
comm = "python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 \--ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b {bs} -lr {r} -rtg \--exp_name q2_b{bs}_r{r}".format(bs=batch_size, r=learning_rate)

os.system(comm)