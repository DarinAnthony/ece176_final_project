import gymnasium as gym
from flappy_bird_env import flappy_bird_env
from gymnasium.envs.registration import register
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Import your modules
from utils.flappyPreprocessor import FlappyBirdPreprocessor
from models.flappyModel import FlappyBirdDQN
from utils.replayBuffer import SequentialGPUReplayBuffer
from agents.flappy import FlappyAgent

# Register FlappyBird environment if not already registered
try:
    register(
        id="FlappyBird-v0",
        entry_point="flappy_bird_env.flappy_bird_env:FlappyBirdEnv",
        max_episode_steps=1000,
    )
except:
    pass  # Already registered

def train_flappy_bird_dqn(num_frames=1000000,
                    memory_size=100000,
                    batch_size=32,
                    gamma=0.99,
                    eps_start=1.0,
                    eps_end=0.1,
                    eps_decay=250000,
                    target_update=5000,
                    learning_rate=0.0025,
                    update_freq=4,
                    replay_start_size=10000,
                    no_op_max=5,
                    eval_interval=50000,
                    save_interval=100000):
    """
    Train a DQN agent to play Flappy Bird.
    """
    # 1. Environment setup
    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    
    # Set device
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # 3. Create the agent
    agent = FlappyAgent(
        env=env,
        replayBufferClass=SequentialGPUReplayBuffer,
        frameShape=(4, 84, 84),
        QNetwork=FlappyBirdDQN,
        PreprocessorClass=FlappyBirdPreprocessor,
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
        eval_interval=eval_interval,
        save_interval=save_interval
    )

    # 4. Train
    print(f"Starting FlappyBird DQN training for {num_frames} frames...")
    
    episode_rewards, eval_rewards = agent.train(num_frames=num_frames)
    print("Training completed!")
    
    # 5. Plot training episode rewards
    os.makedirs("train_runs", exist_ok=True)
    plt.figure(figsize=(12,6))
    plt.plot(episode_rewards, label="Episode Reward")
    plt.title("FlappyBird DQN Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("train_runs/flappy_bird_dqn_training_rewards.png")
    plt.close()

    # 6. Evaluate
    eval_score = agent.evaluate(num_episodes=5)
    print(f"Evaluation over 5 episodes: {eval_score:.2f} average reward")

    # 7. Clean up
    env.close()

if __name__ == "__main__":
    # For quick testing with few frames:
    train_flappy_bird_dqn(num_frames=5000,
                    memory_size=50000,
                    batch_size=32,
                    gamma=0.99,
                    eps_start=1.0,
                    eps_end=0.1,
                    eps_decay=5000,
                    target_update=1000,
                    learning_rate=0.001,
                    update_freq=4,
                    replay_start_size=500,
                    no_op_max=5,
                    eval_interval=2500,
                    save_interval=25000)
    
    # For full training (uncomment this):
    # train_flappy_bird_dqn(num_frames=1000000,
    #                memory_size=200000,         # Increased from 100,000
    #                batch_size=32,
    #                gamma=0.99,
    #                eps_start=1.0,
    #                eps_end=0.1,
    #                eps_decay=250000,
    #                target_update=5000,
    #                learning_rate=0.00025,      # Recommended standard value
    #                update_freq=4,
    #                replay_start_size=10000,
    #                no_op_max=5)