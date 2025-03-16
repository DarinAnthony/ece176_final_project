import torch
import torch.optim as optim
from agents.base import DQNAgent

class FlappyAgent(DQNAgent):
    def __init__(self, 
                 env, 
                 replayBufferClass,
                 frameShape, 
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
                 replayBufferClass,
                 frameShape, 
                 QNetwork, 
                 PreprocessorClass, 
                 device, 
                 memory_size, 
                 batch_size, 
                 gamma, 
                 eps_start, 
                 eps_end, 
                 eps_decay, 
                 target_update,
                 learning_rate,
                 update_freq, 
                 replay_start_size, 
                 no_op_max,
                 weights_path,
                 eval_interval,
                 save_interval)
        
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate,  # A good default for Adam with DQN
            eps=1e-8    # Helps with numerical stability
        )