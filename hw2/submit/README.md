

# Experiment 1: CartPole

```bash
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \-dsa --exp_name q1_sb_no_rtg_dsa
```

```bash
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \-rtg -dsa --exp_name q1_sb_rtg_dsa
```

```bash
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \-rtg --exp_name q1_sb_rtg_na
```

```bash
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \-dsa --exp_name q1_lb_no_rtg_dsa
```

```bash
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \-rtg -dsa --exp_name q1_lb_rtg_dsa
```

```bash
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \-rtg --exp_name q1_lb_rtg_na
```

# Experiment 2: Inverted Pendulum
Run grid search using script

```bash
python cs285/scripts/inverted_pendulum_grid_search.py
```

# Experiment 3: Lunar Lander

```bash
python cs285/scripts/run_hw2.py \--env_name LunarLanderContinuous-v2 --ep_len 1000--discount 0.99 -n 100 -l 2 -s 64 -b 40000 -lr 0.005 \--reward_to_go --nn_baseline --exp_name q3_b40000_r0.005
```

# Experiment 4: HalfCheetah

Run grid search using script

```bash
python cs285/scripts/cheetah_grid_search.py
```

# Experiment 