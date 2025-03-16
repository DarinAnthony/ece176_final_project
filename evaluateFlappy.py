#!/usr/bin/env python
# evaluate_flappy.py
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os
import argparse
import time
import json
from pathlib import Path
from collections import defaultdict
import random

# Import your modules
from utils.flappyPreprocessor import FlappyBirdPreprocessor
from models.flappyModel import FlappyBirdDQN
from utils.visualizer import DQNVisualizer
from agents.base import DQNAgent
from utils.replayBuffer import SequentialGPUReplayBuffer
from flappy_bird_env import flappy_bird_env
from gymnasium.envs.registration import register

# Register FlappyBird environment if not already registered
try:
    register(
        id="FlappyBird-v0",
        entry_point="flappy_bird_env.flappy_bird_env:FlappyBirdEnv",
        max_episode_steps=1000,
    )
except:
    pass  # Already registered

def evaluate_model(model_path, num_episodes=30, record_episodes=5, 
                  output_dir='./eval_runs', device=None,
                  record_length=1000):
    """
    Evaluate a trained DQN model on Flappy Bird.
    
    Args:
        model_path (str): Path to the trained model
        num_episodes (int): Total number of evaluation episodes
        record_episodes (int): Number of episodes to record videos for
        output_dir (str): Directory to save evaluation results
        device (str): Device to run evaluation on ('cuda' or 'cpu')
        record_length (int): Maximum length of recorded episodes
        
    Returns:
        dict: Evaluation results
    """
    # Setup device
    if device is None:
        device = None
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")
    
    # Setup output directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"FlappyBird_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)
    
    # Create environment
    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    
    # Create agent
    agent = DQNAgent(
        env=env,
        replayBufferClass=SequentialGPUReplayBuffer,
        frameShape=(4, 80, 80),  # Using 80x80 for Flappy Bird
        QNetwork=FlappyBirdDQN,  # Use the Flappy Bird network
        PreprocessorClass=FlappyBirdPreprocessor,
        device=device,
        memory_size=10000,  # Doesn't matter for evaluation
        batch_size=32,      # Doesn't matter for evaluation
        target_update=500,  # Doesn't matter for evaluation
        weights_path=model_path  # Load weights from path
    )
    
    # Create visualizer
    visualizer = DQNVisualizer(
        env_name="FlappyBird-v0",
        agent=agent,
        output_dir=run_dir,
        show_metrics=True,
        show_q_values=True,
        show_preprocessed=True
    )
    
    print(f"\n=== Starting Evaluation ({num_episodes} episodes) ===")
    
    # Metrics storage
    rewards = []
    lengths = []
    q_values = []
    actions_taken = defaultdict(int)
    
    # 1. Run standard evaluation using agent's built-in method
    print("\nRunning standard evaluation...")
    standard_mean_reward = agent.evaluate(num_episodes=num_episodes)
    print(f"Standard evaluation complete. Mean reward: {standard_mean_reward:.2f}")
    
    # 2. Record specific episodes for visualization
    record_indices = random.sample(range(num_episodes), min(record_episodes, num_episodes))
    
    for episode in range(num_episodes):
        is_recording = episode in record_indices
        
        if is_recording:
            print(f"\nRecording episode {episode+1}/{num_episodes}...")
            # Use visualizer to record this episode
            episode_stats = visualizer.record_episode(
                filename=f"episode_{episode+1}",
                max_steps=record_length,
                render=True
            )
            # Get metrics from recording
            rewards.append(episode_stats.get('reward', 0))
            lengths.append(episode_stats.get('length', 0))
            
            # Count actions taken
            for action in episode_stats['actions']:
                actions_taken[int(action)] += 1
            
            # Collect Q-values if available
            if episode_stats['q_values']:
                q_values.extend(episode_stats['q_values'])
        else:
            print(f"\nRunning episode {episode+1}/{num_episodes}...")
            # Run episode without recording using agent's run_episode function
            reward, length, episode_actions, episode_q_values = agent.run_episode(
                max_steps=record_length, 
                record=False,
                evaluate=True
            )
            
            # Store metrics
            rewards.append(reward)
            lengths.append(length)
            if episode_q_values:
                q_values.extend(episode_q_values)
            
            # Count actions
            for action in episode_actions:
                actions_taken[int(action)] += 1
        
        # Print progress
        print(f"Episode {episode+1} - Reward: {rewards[-1]}, Length: {lengths[-1]}")
    
    # Calculate aggregate metrics
    mean_reward = np.mean(rewards)
    median_reward = np.median(rewards)
    std_reward = np.std(rewards)
    max_reward = np.max(rewards)
    
    mean_length = np.mean(lengths)
    median_length = np.median(lengths)
    
    # Calculate average max Q-value if we have q_values
    mean_max_q = float(np.mean([np.max(q) for q in q_values])) if q_values else None
    
    # Compile all results
    results = {
        "env_name": "FlappyBird-v0",
        "model_path": model_path,
        "num_episodes": num_episodes,
        "rewards": {
            "mean": float(mean_reward),
            "median": float(median_reward),
            "std": float(std_reward),
            "max": float(max_reward),
            "all_rewards": [float(r) for r in rewards]
        },
        "lengths": {
            "mean": float(mean_length),
            "median": float(median_length),
            "all_lengths": [int(l) for l in lengths]
        },
        "actions": {
            "distribution": {str(k): int(v) for k, v in actions_taken.items()}
        },
        "timestamp": timestamp
    }
    
    if mean_max_q is not None:
        results["q_values"] = {
            "mean_max_q": mean_max_q
        }
    
    # Before saving to JSON
    results = make_json_serializable(results)
    
    # Save results to JSON
    results_path = os.path.join(run_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Generate plots
    generate_plots(results, run_dir)
    
    print("\n=== Evaluation Summary ===")
    print(f"Environment: FlappyBird-v0")
    print(f"Model: {Path(model_path).name}")
    print(f"Episodes: {num_episodes}")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Median reward: {median_reward:.2f}")
    print(f"Max reward: {max_reward:.2f}")
    
    print(f"\nResults saved to: {run_dir}")
    return results


def generate_plots(results, output_dir):
    """
    Generate and save evaluation plots.
    
    Args:
        results (dict): Evaluation results
        output_dir (str): Output directory
    """
    # 1. Rewards plot
    plt.figure(figsize=(10, 6))
    plt.plot(results['rewards']['all_rewards'])
    plt.axhline(y=results['rewards']['mean'], color='r', linestyle='--', 
                label=f'Mean: {results["rewards"]["mean"]:.2f}')
    plt.title(f"Rewards for {results['env_name']}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "metrics/rewards.png"))
    
    # 2. Episode lengths plot
    plt.figure(figsize=(10, 6))
    plt.plot(results['lengths']['all_lengths'])
    plt.axhline(y=results['lengths']['mean'], color='r', linestyle='--', 
                label=f'Mean: {results["lengths"]["mean"]:.2f}')
    plt.title(f"Episode Lengths for {results['env_name']}")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "metrics/episode_lengths.png"))
    
    # 3. Action distribution
    if 'distribution' in results['actions']:
        plt.figure(figsize=(10, 6))
        actions = [int(k) for k in results['actions']['distribution'].keys()]
        counts = [int(v) for v in results['actions']['distribution'].values()]
        plt.bar(actions, counts)
        plt.title(f"Action Distribution for {results['env_name']}")
        plt.xlabel("Action")
        plt.ylabel("Count")
        plt.xticks(actions)
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(output_dir, "metrics/action_distribution.png"))
    
    plt.close('all')
    
    
def make_json_serializable(obj):
    """Convert any object to JSON-compatible types."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu().detach().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, bool):  # Boolean handling
        return bool(obj)
    elif isinstance(obj, (int, float, str, type(None))):
        return obj  # These types are natively JSON serializable
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):  # Handle custom objects
        return {k: make_json_serializable(v) for k, v in obj.__dict__.items()
                if not k.startswith('_')}
    else:
        # Last resort: convert to string
        return str(obj)


def main():
    """Parse command-line arguments and run evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate a trained Flappy Bird DQN model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--num_episodes', type=int, default=30,
                        help='Number of episodes to evaluate')
    parser.add_argument('--record_episodes', type=int, default=5,
                        help='Number of episodes to record videos for')
    parser.add_argument('--output_dir', type=str, default='./eval_runs',
                        help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run evaluation on (cuda or cpu)')
    parser.add_argument('--record_length', type=int, default=1000,
                        help='Maximum steps for recorded episodes')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_model(
        model_path=args.model_path,
        num_episodes=args.num_episodes,
        record_episodes=args.record_episodes,
        output_dir=args.output_dir,
        device=args.device,
        record_length=args.record_length
    )


if __name__ == "__main__":
    main()