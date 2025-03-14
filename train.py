import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from utils.preprocessor import DQNPreprocessor
from models.base import DQN
from utils.replayBuffer import ReplayBuffer
from agents.base import DQNAgent

# Function to train and evaluate on a specific Atari game
def train_atari(game_name="ALE/Breakout-v5", 
                num_frames=1000000, 
                memory_size=100000, 
                batch_size=32, 
                gamma=0.99,
                eps_start=1.0, 
                eps_end=0.1, 
                eps_decay=1000000, 
                target_update=10000,
                learning_rate=0.00025):
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create environment
    env = gym.make(game_name)
    
    # Create agent
    agent = DQNAgent(env, 
                 ReplayBuffer,
                 DQN,
                 DQNPreprocessor,
                 device,
                 memory_size=memory_size, 
                 batch_size=batch_size, 
                 gamma=gamma,
                 eps_start=eps_start, 
                 eps_end=eps_end, 
                 eps_decay=eps_decay, 
                 target_update=target_update,
                 learning_rate=learning_rate,
                 update_freq=4,
                 replay_start_size=50000,
                 no_op_max=30)
    
    # Train agent
    print(f"Training on {game_name} for {num_frames} frames...")
    start_time = time.time()
    episode_rewards, eval_rewards = agent.train(num_frames)
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    
    # Plot training results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title(f'Episode Rewards: {game_name}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    frames, rewards = zip(*eval_rewards)
    plt.plot(frames, rewards)
    plt.title(f'Evaluation Rewards: {game_name}')
    plt.xlabel('Frames')
    plt.ylabel('Avg Reward over 10 Episodes')
    
    plt.tight_layout()
    plt.savefig(f"{game_name.split('/')[-1]}_rewards.png")
    plt.show()
    
    # Save results
    np.save(f"{game_name.split('/')[-1]}_episode_rewards.npy", np.array(episode_rewards))
    np.save(f"{game_name.split('/')[-1]}_eval_rewards.npy", np.array(eval_rewards))
    
    # Close environment
    env.close()
    
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN on Atari games')
    parser.add_argument('--game', type=str, default='ALE/Breakout-v5', help='Atari game to train on')
    parser.add_argument('--frames', type=int, default=1000000, help='Number of frames to train for')
    parser.add_argument('--memory-size', type=int, default=100000, help='Replay buffer size')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--eps-start', type=float, default=1.0, help='Starting epsilon for exploration')
    parser.add_argument('--eps-end', type=float, default=0.1, help='Final epsilon for exploration')
    parser.add_argument('--eps-decay', type=int, default=1000000, help='Frames over which to decay epsilon')
    parser.add_argument('--target-update', type=int, default=10000, help='Frequency of target network updates')
    parser.add_argument('--lr', type=float, default=0.00025, help='Learning rate')
    
    args = parser.parse_args()
    
    train_atari(game_name=args.game,
                num_frames=args.frames,
                memory_size=args.memory_size,
                batch_size=args.batch_size,
                gamma=args.gamma,
                eps_start=args.eps_start,
                eps_end=args.eps_end,
                eps_decay=args.eps_decay,
                target_update=args.target_update,
                learning_rate=args.lr)