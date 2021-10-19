# import torch.multiprocessing as mp
from os import path
from time import time
import numpy as np
import multiprocessing as mp
from os import getpid


from cs285.infrastructure import utils

def get_paths(args):
    return args[0]

def sample_trajectory(env, policy, max_path_length, render, render_mode, procnum, return_dict):
    return_dict[procnum] = utils.sample_trajectory(env, policy, max_path_length, render, render_mode)
    return True

def sample_trajectories(
    envs, policy, min_timesteps_per_batch, max_path_length,
    render=False, render_mode=('rgb_array')):
    # TODO: get this from hw1
    """
        Collect rollouts until we have collected min_timesteps_per_batch steps.

        TODO implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
        Hint2: use get_pathlength to count the timesteps collected in each path
    """
    timesteps_this_batch = 0
    paths = []
    jobs = []
    while timesteps_this_batch < min_timesteps_per_batch:
        manager = mp.Manager()
        return_dict = manager.dict()
        for rank in range(len(envs)):
            p = mp.Process(target=sample_trajectory, args=(envs[rank], policy, max_path_length, render, render_mode, rank, return_dict,))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        curr_paths = return_dict.values()
        pathlengths = list(map(lambda x: utils.get_pathlength(x), curr_paths))
        timesteps_this_batch += sum(pathlengths)
        # print(pathlengths, timesteps_this_batch)

        # Wut?
        for env in envs:
            env.seed(np.random.randint(1999))

        paths = paths + list(curr_paths)

    return paths, timesteps_this_batch