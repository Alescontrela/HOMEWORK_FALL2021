# Q1 - Behavior Cloning Commands

Ant
```
python cs285/scripts/run_hw1.py \--expert_policy_file cs285/policies/experts/Ant.pkl \--env_name Ant-v2 --exp_name bc_ant --n_iter 1 \--expert_data cs285/expert_data/expert_data_Ant-v2.pkl \--video_log_freq -1 --eval_batch_size=10000
```

Humanoid
```
python cs285/scripts/run_hw1.py \--expert_policy_file cs285/policies/experts/Humanoid.pkl \--env_name Humanoid-v2 --exp_name bc_humanoid --n_iter 1 \--expert_data cs285/expert_data/expert_data_Humanoid-v2.pkl \--video_log_freq -1 --eval_batch_size=10000
```

# Q2 - DAgger commands

Ant
```
python cs285/scripts/run_hw1.py \--expert_policy_file cs285/policies/experts/Ant.pkl \--env_name Ant-v2 --exp_name dagger_ant --n_iter 30 \--do_dagger --expert_data cs285/expert_data/expert_data_Ant-v2.pkl \--video_log_freq -1 --eval_batch_size=5000 --ep_len 1000 --train_batch_size 500 --batch_size=2500
```

Humanoid
```
 python cs285/scripts/run_hw1.py \--expert_policy_file cs285/policies/experts/Humanoid.pkl \--env_name Humanoid-v2 --exp_name dagger_humanoid --n_iter 30 \--do_dagger --expert_data cs285/expert_data/expert_data_Humanoid-v2.pkl \--video_log_freq -1 --eval_batch_size=5000 --ep_len 1000 --train_batch_size 500 --batch_size=2500
```