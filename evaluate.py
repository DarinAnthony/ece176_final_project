#!/usr/bin/env python
# evaluate.py
import torch
import numpy as np
import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
import os
import argparse
import time
from collections import defaultdict
import json
from pathlib import Path

from utils.preprocessor import DQNPreprocessor
from models.base import DQN
from models.dqn2015 import DQN2
from utils.visualizer import DQNVisualizer
from agents.base import DQNAgent
from utils.replayBuffer import ReplayBuffer


class ModelEvaluator:
    """Class to evaluate a trained DQN model and collect performance metrics."""
    
    def __init__(self, model_path, env_name, num_episodes=30, max_steps=108000, 
                 output_dir='./evaluation', epsilon=0.05, record_every=5, 
                 record_length=3000, device=None):
        """
        Initialize the evaluator.
        
        Args:
            model_path (str): Path to the trained model checkpoint
            env_name (str): Name of the Atari environment
            num_episodes (int): Number of episodes to evaluate
            max_steps (int): Maximum steps per episode (108000 = 30 mins at 60 fps)
            output_dir (str): Directory to save evaluation results
            epsilon (float): Exploration rate during evaluation
            record_every (int): Record video every N episodes
            record_length (int): Maximum steps for recorded episodes
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.env_name = env_name
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.record_every = record_every
        self.record_length = record_length
        
        # Set up device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        # Set up output directory
        self.output_dir = os.path.join(output_dir, f"{env_name.replace('/', '_')}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create environment
        self.env = gym.make(env_name)
        self.num_actions = self.env.action_space.n
        
        # Load model and set up agent
        self._setup_agent()
        
        # Create visualizer
        self.visualizer = DQNVisualizer(
            env_name=env_name,
            agent=self.agent,
            output_dir=self.output_dir,
            show_metrics=True,
            show_q_values=True,
            show_preprocessed=True
        )
        
        # Metrics storage
        self.episode_rewards = []
        self.episode_lengths = []
        self.q_values = []
        self.action_distributions = defaultdict(int)
    
    def _setup_agent(self):
        """Set up the agent with the loaded model."""
        # Create a temporary agent for loading
        self.agent = DQNAgent(
            env=self.env,
            replayBufferClass=ReplayBuffer,
            QNetwork=DQN2,  # Use the same network architecture as in training
            PreprocessorClass=DQNPreprocessor,
            device=self.device,
            memory_size=10000,  # Doesn't matter for evaluation
            batch_size=32,      # Doesn't matter for evaluation
            target_update=500,  # Doesn't matter for evaluation
            epsilon=self.epsilon
        )
        
        # Load the saved model
        self.agent.load_weights(self.model_path)
        
        # Set agent to eval mode
        self.agent.policy_net.eval()
        self.agent.target_net.eval()
        
        # Add methods required by the visualizer if they don't exist
        if not hasattr(self.agent, 'get_q_values'):
            def get_q_values(state):
                if isinstance(state, np.ndarray):
                    state = np.transpose(state, (2, 0, 1)) if state.shape[-1] == 4 else state
                    state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    return self.agent.policy_net(state).cpu().numpy()[0]
            self.agent.get_q_values = get_q_values
        
        if not hasattr(self.agent, 'process_observation'):
            def process_observation(obs):
                self.agent.preprocessor.reset()
                state = None
                while state is None:
                    state = self.agent.preprocessor.process(obs)
                return state
            self.agent.process_observation = process_observation
    
    def evaluate(self):
        """Run the evaluation process across multiple episodes."""
        print(f"\nEvaluating model on {self.env_name} for {self.num_episodes} episodes...")
        
        for episode in range(self.num_episodes):
            # Decide if this episode should be recorded
            should_record = episode % self.record_every == 0
            max_ep_steps = self.record_length if should_record else self.max_steps
            
            if should_record:
                print(f"\nRecording episode {episode+1}/{self.num_episodes}...")
                episode_stats = self.visualizer.record_episode(
                    filename=f"episode_{episode+1}",
                    max_steps=max_ep_steps,
                    render=True
                )
                # Store metrics from recorded episode
                self.episode_rewards.append(episode_stats['total_reward'])
                self.episode_lengths.append(episode_stats['steps'])
                
                # Store Q-values
                if 'q_values' in episode_stats:
                    self.q_values.extend(episode_stats['q_values'])
                
                # Track action distribution
                if 'actions' in episode_stats:
                    for action in episode_stats['actions']:
                        self.action_distributions[int(action)] += 1
            else:
                print(f"\nRunning episode {episode+1}/{self.num_episodes}...")
                reward, length, ep_q_values, actions = self._run_episode(max_ep_steps)
                
                # Store metrics
                self.episode_rewards.append(reward)
                self.episode_lengths.append(length)
                self.q_values.extend(ep_q_values)
                
                # Track action distribution
                for action in actions:
                    self.action_distributions[int(action)] += 1
            
            # Print episode results
            print(f"Episode {episode+1} - Reward: {self.episode_rewards[-1]}, "
                  f"Length: {self.episode_lengths[-1]} steps")
        
        # Save and plot all metrics
        self._save_metrics()
        self._plot_metrics()
        
        print("\nEvaluation complete!")
        return {
            'mean_reward': np.mean(self.episode_rewards),
            'median_reward': np.median(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'mean_length': np.mean(self.episode_lengths),
            'action_distribution': dict(self.action_distributions)
        }
    
    def _run_episode(self, max_steps):
        """Run a single evaluation episode without recording video."""
        obs, _ = self.env.reset()
        
        # Reset preprocessor
        self.agent.preprocessor.reset()
        
        state = self.agent.get_state(obs)
        
        total_reward = 0
        step = 0
        done = False
        
        # Track metrics for this episode
        episode_q_values = []
        episode_actions = []
        
        while not done and step < max_steps:
            # Select action
            action = self.agent.select_action(state, evaluate=True)
            episode_actions.append(action)
            
            # Get Q-values for current state
            with torch.no_grad():
                q_values = self.agent.policy_net(state).cpu().numpy()[0]
                episode_q_values.append(q_values)
            
            # Execute action
            obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Process next state
            next_state = self.agent.get_state(obs)
            
            # Update
            state = next_state
            total_reward += reward
            step += 1
            
            # Print progress occasionally
            if step % 1000 == 0:
                print(f"  Step {step}, Current reward: {total_reward}")
        
        return total_reward, step, episode_q_values, episode_actions
    
    def _save_metrics(self):
        """Save evaluation metrics to JSON file."""
        metrics = {
            'env_name': self.env_name,
            'model_path': self.model_path,
            'num_episodes': self.num_episodes,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'mean_reward': float(np.mean(self.episode_rewards)),
            'median_reward': float(np.median(self.episode_rewards)),
            'max_reward': float(np.max(self.episode_rewards)),
            'std_reward': float(np.std(self.episode_rewards)),
            'mean_q_value': float(np.mean([np.max(q) for q in self.q_values])),
            'action_distribution': {str(k): v for k, v in self.action_distributions.items()}
        }
        
        metrics_file = os.path.join(self.output_dir, 'evaluation_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Metrics saved to {metrics_file}")
    
    def _plot_metrics(self):
        """Generate and save plots of evaluation metrics."""
        # Plot episode rewards
        plt.figure(figsize=(12, 6))
        plt.plot(self.episode_rewards)
        plt.title(f"{self.env_name} - Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'episode_rewards.png'))
        
        # Plot episode lengths
        plt.figure(figsize=(12, 6))
        plt.plot(self.episode_lengths)
        plt.title(f"{self.env_name} - Episode Lengths")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'episode_lengths.png'))
        
        # Plot action distribution
        plt.figure(figsize=(12, 6))
        actions = list(self.action_distributions.keys())
        counts = list(self.action_distributions.values())
        plt.bar(actions, counts)
        plt.title(f"{self.env_name} - Action Distribution")
        plt.xlabel("Action")
        plt.ylabel("Count")
        plt.xticks(actions)
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(self.output_dir, 'action_distribution.png'))
        
        # Plot Q-value distribution if we have Q-values
        if self.q_values:
            max_q_values = [np.max(q) for q in self.q_values]
            plt.figure(figsize=(12, 6))
            plt.hist(max_q_values, bins=50)
            plt.title(f"{self.env_name} - Q-Value Distribution")
            plt.xlabel("Max Q-Value")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, 'q_value_distribution.png'))
        
        # Generate state-value visualization if visualizer has t-SNE capabilities
        if hasattr(self.visualizer, 'generate_tsne_visualization'):
            try:
                print("Generating t-SNE visualization of state values...")
                self.visualizer.generate_tsne_visualization(
                    num_states=1000,
                    filename='state_value_tsne'
                )
            except Exception as e:
                print(f"Could not generate t-SNE visualization: {e}")


def main():
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate a trained DQN model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--env_name', type=str, required=True,
                        help='Name of the Atari environment')
    parser.add_argument('--num_episodes', type=int, default=30,
                        help='Number of episodes to evaluate')
    parser.add_argument('--max_steps', type=int, default=108000,
                        help='Maximum steps per episode (default: 108000, approx. 30 mins of gameplay)')
    parser.add_argument('--output_dir', type=str, default='./evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='Epsilon value for epsilon-greedy policy')
    parser.add_argument('--record_every', type=int, default=5,
                        help='Record video every N episodes')
    parser.add_argument('--record_length', type=int, default=3000,
                        help='Maximum steps for recorded episodes')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run evaluation on (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        env_name=args.env_name,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        epsilon=args.epsilon,
        record_every=args.record_every,
        record_length=args.record_length,
        device=args.device
    )
    
    # Run evaluation
    results = evaluator.evaluate()
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Environment: {args.env_name}")
    print(f"Model: {Path(args.model_path).name}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Mean reward: {results['mean_reward']:.2f}")
    print(f"Median reward: {results['median_reward']:.2f}")
    print(f"Max reward: {results['max_reward']:.2f}")
    print(f"Mean episode length: {results['mean_length']:.2f}")
    print(f"Action distribution: {results['action_distribution']}")
    print(f"Results saved to: {evaluator.output_dir}")


if __name__ == "__main__":
    main()