import os
import numpy as np
import torch
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

class Logger():
    def __init__(
                self, 
                experiment_name = None,
                time = None,
                ):
        self.time = time
        self.experiment_name = experiment_name
        self.name_to_value = defaultdict(float)
        self._configure_logger()

    def _configure_logger(self):
        time_record = os.path.join(self.time)
        self.folder = os.path.join(f'{self.experiment_name}_Record','logs' , time_record)

        os.makedirs(self.folder, exist_ok=True)
        self.writer = SummaryWriter(f'{self.folder}')

    def record(self, key, value):
        self.name_to_value[key] = value

    def to_tensorboard(self, data, time_count):
        self.writer.add_scalar('Episode_reward_mean', data, time_count)

    def write_out(self):
        for key in self.name_to_value:
            print(key,':' ,round(self.name_to_value[key], 2))
        print()

def obs_as_tensor(obs, device):
    obs = np.array(obs)
    obs = torch.as_tensor(obs, dtype=torch.float).to(device)
    return obs