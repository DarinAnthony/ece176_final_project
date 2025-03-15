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
from utils.visualizer import DQNVisualizer

def train_breakout_dqn():
    """
    Short test training of a DQN agent on Breakout.
    """
    # 1. Environment setup
    env_name = "ALE/Breakout-v5"
    env = gym.make(env_name)
    
    # Force CPU usage
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # 2. Create the agent with minimal hyperparameters for quick testing
    num_frames = 2000  # Very small number of frames for quick testing
    
    agent = DQNAgent(
        env=env,
        replayBufferClass=ReplayBuffer,
        QNetwork=DQN2,
        PreprocessorClass=DQNPreprocessor,
        device=device,
        memory_size=10000,          # Smaller replay buffer
        batch_size=32,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.1,
        eps_decay=2000,             # Very fast epsilon decay
        target_update=500,          # Update target network frequently
        learning_rate=0.0025,       # Slightly higher learning rate
        update_freq=4,
        replay_start_size=1000,     # Start training after just 1000 frames
        no_op_max=5,                # Fewer no-ops at start of episode
    )

    # 3. Train for a small number of frames
    print(f"Starting quick test training for {num_frames} frames...")
    
    episode_rewards, eval_rewards = agent.train(num_frames=num_frames)
    print("Training completed!")
    
    # 4. Plot training episode rewards
    os.makedirs("train_runs", exist_ok=True)
    plt.figure(figsize=(12,6))
    plt.plot(episode_rewards, label="Episode Reward")
    plt.title("Training Rewards over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("train_runs/breakout_dqn_test_training_rewards.png")
    plt.close()

    # 5. Quick evaluation - just 3 episodes
    eval_score = agent.evaluate(num_episodes=3)
    print(f"Evaluation over 3 episodes: {eval_score:.2f} average reward")

    # 6. Clean up
    env.close()

if __name__ == "__main__":
    train_breakout_dqn()