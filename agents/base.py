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


# Main DQN Agent
class DQNAgent:
    def __init__(self, env, replayBufferClass, QNetwork, PreprocessorClass, device="cpu", memory_size=100000, batch_size=32, gamma=0.99, 
                 eps_start=1.0, eps_end=0.1, eps_decay=1000000, target_update=10000,
                 learning_rate=0.00025):
        
        self.env = env
        self.device = device
        self.memory = replayBufferClass(memory_size)
        self.preprocessor = PreprocessorClass()
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.frame_skip = 4
        
        # Action space
        self.num_actions = env.action_space.n
        
        # State shape (4 frames, 84x84)
        self.state_shape = (4, 84, 84)
        
        # Create Q networks
        self.policy_net = QNetwork(self.state_shape, self.num_actions).to(self.device)
        self.target_net = QNetwork(self.state_shape, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=learning_rate)
        
        # Counters
        self.steps_done = 0
        self.episode_rewards = []
    
    def select_action(self, state, evaluate=False):
        """
        Epsilon-greedy action selection
        """
        # Set epsilon based on steps (linear annealing)
        if evaluate:
            epsilon = 0.05  # Fixed epsilon for evaluation
        else:
            epsilon = max(self.eps_end, self.eps_start - (self.steps_done / self.eps_decay))
        
        if random.random() > epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1).item()
        else:
            return random.randrange(self.num_actions)
    
    def get_state(self, obs):
        """
        Stack 4 frames together to create the state
        """
        self.preprocessor.reset()
        
        # get a state
        state = None
        while state is None:
            state = self.preprocessor.process(obs)
            if state is None:
                # Take random action to get more frames
                obs, _, _, _, _ = self.env.step(self.env.action_space.sample())
                
        # Convert numpy array to PyTorch tensor and correct the dimension order
        state = np.transpose(state, (2, 0, 1))  # Convert from (84,84,4) to (4,84,84)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        return state
    
    def train(self, num_frames):
        """
        Main training loop
        """
        # Initialize progress tracking
        progress_bar = tqdm(total=num_frames, desc="Training")
        episode_rewards = []
        rewards_window = deque(maxlen=100)
        eval_rewards = []
        frame_count = 0
        episode_count = 0
        best_mean_reward = -float('inf')
        
        while frame_count < num_frames:
            # Reset environment
            obs, _ = self.env.reset()
            state = self.get_state(obs)
            
            episode_reward = 0
            done = False
            
            while not done and frame_count < num_frames:
                # Select and perform action
                action = self.select_action(state)
                
                # Skip frames (act every k frames)
                total_reward = 0
                for _ in range(self.frame_skip):
                    obs, reward, terminated, truncated, _ = self.env.step(action)
                    total_reward += reward
                    done = terminated or truncated
                    if done:
                        break
                
                # Process new frame
                next_state = self.get_state(obs)
                
                # Clip rewards to {-1, 0, 1} as per the paper
                clipped_reward = np.sign(total_reward)
                
                # Store transition in memory
                self.memory.add(state, action, clipped_reward, next_state, done)
                
                # Move to the next state
                state = next_state
                episode_reward += total_reward
                frame_count += 1
                self.steps_done += 1
                
                # Update progress bar
                progress_bar.update(1)
                
                # Train the model if enough samples are available
                if len(self.memory) > self.batch_size:
                    self._optimize_model()
                
                # Update the target network
                if self.steps_done % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                
                # Evaluate periodically
                if frame_count % 50000 == 0:
                    eval_reward = self.evaluate(10)
                    eval_rewards.append((frame_count, eval_reward))
                    print(f"\nFrame {frame_count}/{num_frames}, Mean Eval Reward: {eval_reward:.2f}")
                    
                    # Save best model
                    if eval_reward > best_mean_reward:
                        best_mean_reward = eval_reward
                        torch.save(self.policy_net.state_dict(), f"dqn_{self.env.unwrapped.spec.id}_best.pth")
            
            # Episode finished
            episode_count += 1
            rewards_window.append(episode_reward)
            episode_rewards.append(episode_reward)
            
            # Print episode stats
            mean_reward = np.mean(rewards_window) if rewards_window else episode_reward
            print(f"\rEpisode {episode_count}, Frames: {frame_count}, Reward: {episode_reward:.2f}, "
                  f"Avg Reward: {mean_reward:.2f}, Epsilon: {max(self.eps_end, self.eps_start - (self.steps_done / self.eps_decay)):.4f}",
                  end="")
        
        progress_bar.close()
        print("\nTraining completed!")
        
        # Save final model
        torch.save(self.policy_net.state_dict(), f"dqn_{self.env.unwrapped.spec.id}_final.pth")
        
        return episode_rewards, eval_rewards
    
    def _optimize_model(self):
        """
        Perform one step of optimization
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Sample from replay memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Compute Q values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute next state values with target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = F.smooth_l1_loss(q_values, expected_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def evaluate(self, num_episodes=10, epsilon=0.05):
        """
        Evaluate the agent over several episodes
        """
        total_rewards = []
        
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            
            # get a state
            state = self.get_state(obs)
            
            episode_reward = 0
            done = False
            
            while not done:
                # Select action with fixed epsilon
                if random.random() > epsilon:
                    with torch.no_grad():
                        action = self.policy_net(state).max(1)[1].view(1, 1).item()
                else:
                    action = random.randrange(self.num_actions)
                
                # Skip frames (act every k frames)
                total_reward = 0
                for _ in range(self.frame_skip):
                    obs, reward, terminated, truncated, _ = self.env.step(action)
                    total_reward += reward
                    done = terminated or truncated
                    if done:
                        break
                
                # Process new frame
                next_state = self.get_state(obs)
                
                # Move to the next state
                state = next_state
                episode_reward += total_reward
            
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards)