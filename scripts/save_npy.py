import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorflow.python.framework import tensor_util
import numpy as np
import os
import glob
# TAG = "epoch/grad_true_dpi_sample_A_cos_sim"
TAG = 'eval/mean_reward'

x = []
# f = '/home/nycucpu1/grid/runs/grid_size=10, batch_size=10, policy_lr=0.001, value_lr=1, init-/events.out.tfevents.1643165851.cn0125.36612.0'
# f = '/home/nycucpu1/grid/runs/grid_size=10, batch_size=10, policy_lr=0.001, value_lr=10, init/events.out.tfevents.1643071743.cn0117.45284.0'
DIR = '/home/darklanx/update/rl-baselines3-zoo/tensorboard2/MinAtar/Freeway-v0'
for i, f in enumerate(glob.glob(os.path.join(DIR, "CAPO*", '*.cpu2.*'))):
    y = []
    for event in summary_iterator(f):
        for value in event.summary.value:
            if value.tag == TAG:
                y.append(value.simple_value)
                print(value)
            # print(value.tag)
            # if value.HasField('simple_value'):
            # print(type(value.simple_value))
            # print(len(value.simple_value))
    filename = f.split('/')[-2]
    algo = 'freeway'
    np.save(f'./npy/{algo}_{i}.npy', y)