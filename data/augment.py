import numpy as np
import random

def spec_augment(feat, T = 20, F = 27, time_mask_num = 2, freq_mask_num = 2):
    feat_size = feat.shape[0]
    seq_len = feat.shape[1]
    # print(feat_size)
    # freq mask
    for _ in range(freq_mask_num):
        f = np.random.uniform(low=0.0, high=F)
        f = int(f)
        f0 = random.randint(0, seq_len - f)
        feat[f0: f0 + f, :] = 0

    # time mask
    for _ in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=T)
        t = int(t)
        t0 = random.randint(0, feat_size - t)
        feat[:, t0: t0 + t] = 0

    return feat