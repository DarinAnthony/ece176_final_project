from agents.base import DQNAgent
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import os
import time
from tqdm import tqdm
import gc
import psutil

class CartPoleAgent(DQNAgent):
    def __init__(self, 
                 env, 
                 replayBufferClass, 
                 QNetwork, 
                 PreprocessorClass, 
                 device="cpu", 
                 memory_size=1000000, 
                 batch_size=32, 
                 gamma=0.99, 
                 eps_start=1.0, 
                 eps_end=0.1, 
                 eps_decay=1000000, 
                 target_update=10000,
                 learning_rate=0.00025, 
                 update_freq=4, 
                 replay_start_size=50000, 
                 no_op_max=30,
                 weights_path=None,
                 eval_interval=250000,
                 save_interval=200000):
        super().__init__(env, 
                 replayBufferClass=replayBufferClass, 
                 QNetwork=QNetwork, 
                 PreprocessorClass=PreprocessorClass, 
                 device=device, 
                 memory_size=memory_size, 
                 batch_size=batch_size, 
                 gamma=gamma, 
                 eps_start=eps_start, 
                 eps_end=eps_end, 
                 eps_decay=eps_decay, 
                 target_update=target_update,
                 learning_rate=learning_rate, 
                 update_freq=update_freq, 
                 replay_start_size=replay_start_size, 
                 no_op_max=no_op_max,
                 weights_path=weights_path,
                 eval_interval=eval_interval,
                 save_interval=save_interval)
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
    def get_state(self, obs):
        """
        Stack 4 frames together to create the state
        """
        # get a state
        state, _, _, _, _ = self.env.step(self.env.action_space.sample())
        
        return state