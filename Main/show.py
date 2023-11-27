import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import collections
from statistics import mean

def smooth(csv_path, weight=0.85):
    data = pd.read_csv(filepath_or_buffer=csv_path, header=0, names=['Step','Value'], dtype={'Step':np.int64, 'Value':np.int64})
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    save = pd.DataFrame({'Step':data['Step'].values, 'Value':smoothed})
    save.to_csv('smooth_' + csv_path)


def smooth_and_plot(csv_path, weights = 0.99):
    weight = weights
    
    data = pd.read_csv(filepath_or_buffer=csv_path, header=0, names=['Step','Value'], dtype={'Step':np.int64, 'Value':np.float64})
    scalar = data['Value'].values
    
    smoothed = []
    max_value = collections.deque(maxlen=int(50))
    for idx, point in enumerate(scalar):
        max_value.append(point)
        smoothed.append(mean(max_value))

    steps = data['Step'].values
    steps = steps.tolist()
    origin = scalar.tolist()

    fig = plt.figure(1)
    # plt.plot(steps, origin, label='origin')
    plt.plot(steps, smoothed)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.xlabel('Range')
    plt.ylabel('Loss')
    plt.show()
    print(weight)

if __name__=='__main__':
    # smooth('C:/Users/kmes5/Downloads/1017reward/final_reward.csv')
    smooth_and_plot('C:/Users/kmes5/Desktop/BBTan_1108/1114_500.csv')