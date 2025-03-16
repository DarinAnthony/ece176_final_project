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
from utils.replayBuffer import SequentialGPUReplayBuffer
from agents.base import DQNAgent
from utils.visualizer import DQNVisualizer

def train_breakout_dqn(num_frames=10000000,
                    memory_size=500000,          # Smaller replay buffer
                    batch_size=32,
                    gamma=0.99,
                    eps_start=1.0,
                    eps_end=0.1,
                    eps_decay=250000,             # Very fast epsilon decay
                    target_update=5000,          # Update target network frequently
                    learning_rate=0.0025,       # Slightly higher learning rate
                    update_freq=4,
                    replay_start_size=25000,     # Start training after just 1000 frames
                    no_op_max=5,
                    eval_interval=250000,
                    save_interval=500000):
    """
    Short test training of a DQN agent on Breakout.
    """
    # 1. Environment setup
    env_name = "BreakoutDeterministic-v4"
    env = gym.make(env_name)
    
    # Set device
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # 2. Create the agent with minimized hyperparameters
    agent = DQNAgent(
        env=env,
        replayBufferClass=SequentialGPUReplayBuffer,
        frameShape=(4, 84, 84),
        QNetwork=DQN2,
        PreprocessorClass=DQNPreprocessor,
        device=device,
        memory_size=memory_size,          # Smaller replay buffer
        batch_size=batch_size,
        gamma=gamma,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,             # Very fast epsilon decay
        target_update=target_update,          # Update target network frequently
        learning_rate=learning_rate,       # Slightly higher learning rate
        update_freq=update_freq,
        replay_start_size=replay_start_size,     # Start training after just 1000 frames
        no_op_max=no_op_max,                # Fewer no-ops at start of episode
        eval_interval=eval_interval,
        save_interval=save_interval
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
    # train_breakout_dqn(num_frames=10000000,
    #                 memory_size=1000000,         
    #                 batch_size=32,
    #                 gamma=0.99,
    #                 eps_start=1.0,
    #                 eps_end=0.1,
    #                 eps_decay=1000000,             
    #                 target_update=5000,          
    #                 learning_rate=0.00025,       
    #                 update_freq=4,
    #                 replay_start_size=50000,
    #                 no_op_max=30,
    #                 eval_interval=250000,
    #                 save_interval=500000)
    
    
    train_breakout_dqn(num_frames=2000,
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
        no_op_max=5)