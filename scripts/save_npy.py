import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorflow.python.framework import tensor_util
import numpy as np
import os
import glob
# TAG = "epoch/grad_true_dpi_sample_A_cos_sim"
TAG = 'eval/mean_reward'
# TAG = "Train/AverageReturns"

x = []
# f = '/home/nycucpu1/grid/runs/grid_size=10, batch_size=10, policy_lr=0.001, value_lr=1, init-/events.out.tfevents.1643165851.cn0125.36612.0'
# f = '/home/nycucpu1/grid/runs/grid_size=10, batch_size=10, policy_lr=0.001, value_lr=10, init/events.out.tfevents.1643071743.cn0117.45284.0'
ALGO = "CAPO2"
# env = 'Freeway'

# DIR= "/home/nycucpu1/final_run/revisiting_rainbowlogs"
# envs = ['Freeway', 'Breakout', 'Seaquest', 'SpaceInvaders', 'Asterix']

envs = ['Freeway1']
SAVE_DIR = f'/home/nycucpu1/final_run/rl-baselines3-zoo/npy/{ALGO}/'
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)
for env in envs:
    DIR = f'/home/nycucpu1/final_run/rl-baselines3-zoo/tb_{ALGO}/MinAtar/{env}-v0/'
    for i, f in enumerate(sorted(glob.glob(os.path.join(DIR, '*', 'events.*')))):
        print(f)
        y = []
        for event in summary_iterator(f):
            for value in event.summary.value:
                if value.tag == TAG:
                    y.append(value.simple_value)
                    # print(value)

        # filename = f.split('/')[-2]
        if env == 'SpaceInvaders':
            np.save(os.path.join(SAVE_DIR, f"space_invaders_{i}.npy"), y)
        else:
            np.save(os.path.join(SAVE_DIR, f"{env.lower()}_{i}.npy"), y)