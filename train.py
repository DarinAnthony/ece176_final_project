import torch
import numpy as np
import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
import os

# Import your modules
from utils.preprocessor import DQNPreprocessor
from models.base import DQN
from models.dqn2015 import DQN2
from utils.replayBuffer import ReplayBuffer
from agents.base import DQNAgent
from utils.visualizer import DQNVisualizer  # The visualizer class you shared

def train_breakout_dqn():
    """
    Train a DQN agent on Breakout until it reaches strong performance,
    loosely following the 2015 DeepMind DQN hyperparameters.
    """
    # 1. Environment setup
    env_name = "ALE/Breakout-v5"
    env = gym.make(env_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Create the agent with "paper-like" hyperparameters
    # NOTE: The values below reflect what was used in Mnih et al. (2015), but feel free to tweak.
    num_frames = 100_000

    agent = DQNAgent(
        env=env,
        replayBufferClass=ReplayBuffer,
        QNetwork=DQN2,
        PreprocessorClass=DQNPreprocessor,
        device=device,
        memory_size=100_000, 
        batch_size=32,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.1,
        eps_decay=200_000,   # smaller than 1 million
        target_update=2_000, # update target more frequently
        learning_rate=0.00025,
        update_freq=4,
        replay_start_size=2_000,  # small replay start
        no_op_max=30
    )


    # 3. Train for many frames. The original paper used up to 10^7 or more.
    #    This can take many hours/days on CPU and ~1-2 days on a good GPU.
    num_frames = 10_000_000
    print(f"Starting long training for {num_frames} frames...")

    episode_rewards, eval_rewards = agent.train(num_frames=num_frames)
    print("Training completed!")
    
    # 4. Plot training episode rewards
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(12,6))
    plt.plot(episode_rewards, label="Episode Reward")
    plt.title("Training Rewards over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/breakout_dqn_training_rewards.png")
    plt.close()

    # 5. Evaluate the final policy (optional)
    eval_score = agent.evaluate(num_episodes=10, epsilon=0.05)
    print(f"Evaluation over 10 episodes: {eval_score:.2f} average reward")

    # 6. Clean up
    env.close()

if __name__ == "__main__":
    train_breakout_dqn()
