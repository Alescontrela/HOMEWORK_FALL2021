import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    Y = []
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:

            if v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
    return Y

if __name__ == '__main__':
    import glob

    dqn_logdirs = [
        'data/q2_dqn_1_LunarLander-v3_19-10-2021_20-18-27/events*',
        'data/q2_dqn_2_LunarLander-v3_19-10-2021_20-25-14/events*',
        'data/q2_dqn_3_LunarLander-v3_19-10-2021_20-26-26/events*'
    ]

    doubledqn_logdirs = [
        'data/q2_doubledqn_1_LunarLander-v3_19-10-2021_20-24-17/events*',
        'data/q2_doubledqn_2_LunarLander-v3_19-10-2021_20-25-41/events*',
        'data/q2_doubledqn_3_LunarLander-v3_19-10-2021_20-26-46/events*'
    ]

    dqn_eventfiles = [
        glob.glob(logdir)[0] for logdir in dqn_logdirs]
    doubledqn_eventfiles = [
        glob.glob(logdir)[0] for logdir in doubledqn_logdirs]

    dqn_avg_returns = np.stack([
        get_section_results(eventfile) for eventfile in dqn_eventfiles])
    doubledqn_avg_returns = np.stack([
        get_section_results(eventfile) for eventfile in doubledqn_eventfiles])

    dqn_avg_returns = np.mean(dqn_avg_returns, axis=0)
    doubledqn_avg_returns = np.mean(doubledqn_avg_returns, axis=0)

    plt.plot(dqn_avg_returns, label='dqn return')
    plt.plot(doubledqn_avg_returns, label='doubledqn avg return')
    plt.legend()
    plt.ylim([-100, 150])
    plt.show()

