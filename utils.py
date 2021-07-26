import numpy as np

import torch.nn as nn

class ScheduleAdam():

    def __init__(self, optimizer, hidden_dim, warm_steps):
        self.init_lr = np.power(hidden_dim, -0.5) # 1 / sqrt(hidden_dim)
        self.optimizer = optimizer
        self.current_steps = 0
        self.warm_steps = warm_steps

    def step(self):
        # current_step 정보를 이용해서 lr Update
        self.current_steps += 1
        lr = self.init_lr * self.get_scale()

        for p in self.optimizer.param_groups:
            p['lr'] = lr # lr Update

        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_scale(self):
        return np.min([np.power(self.current_steps, -0.5),
                       self.current_steps * np.power(self.warm_steps, -1.5)
                       ])

def init_weight(m):
    nn.init.xavier_normal_(m.weight)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)