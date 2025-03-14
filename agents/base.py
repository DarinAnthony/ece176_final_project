import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import os
from tqdm import tqdm


# Main DQN Agent
class DQNAgent:
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
                 weights_path=None):
        
        self.env = env
        self.device = device
        self.memory = replayBufferClass(memory_size, self.device)
        self.preprocessor = PreprocessorClass()
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.frame_skip = 4
        self.update_freq = update_freq  # Default set to 4
        self.update_counter = 0  # For tracking updates
        self.no_op_max = no_op_max  # Default set to 30
        self.replay_start_size = replay_start_size  # Default set to 50,000
        
        # Action space
        self.num_actions = env.action_space.n
        
        # State shape (4 frames, 84x84)
        self.state_shape = (4, 84, 84)
        
        # Create Q networks
        self.policy_net = QNetwork(self.state_shape, self.num_actions).to(self.device)
        self.target_net = QNetwork(self.state_shape, self.num_actions).to(self.device)
        
        # Load pre-trained weights if provided
        if weights_path is not None:
            self.load_weights(weights_path)
        else:
            # Initialize target network with policy network weights
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(), 
            lr=learning_rate,
            alpha=0.95,  # squared gradient momentum
            eps=0.01,    # min squared gradient
            momentum=0.95,  # gradient momentum
            centered=False
            )
        
        # Counters
        self.steps_done = 0
        self.episode_rewards = []
        
        
######################################################################
############################# Helper Methods
######################################################################
    
    def reset_environment(self):
        """Reset with random number of no-ops"""
        obs, _ = self.env.reset()
        
        # Apply random number of no-op actions
        no_op_count = random.randint(0, self.no_op_max)
        
        # Assume action 0 is NOOP
        for _ in range(no_op_count):
            obs, _, terminated, truncated, _ = self.env.step(0)
            if terminated or truncated:
                obs, _ = self.env.reset()
                break
        
        return obs
    
    
    def fill_replay_memory(self):
        """
        NEW: Fill replay memory with random actions before training starts
        """
        print(f"\nFilling replay memory with {self.replay_start_size} frames of random experience...")
        
        progress_bar = tqdm(total=self.replay_start_size, desc="Filling Replay Memory")
        frame_count = 0
        
        while frame_count < self.replay_start_size:
            # Reset with NOOPs
            obs = self.reset_environment()
            state = self.get_state(obs)
            
            done = False
            
            while not done and frame_count < self.replay_start_size:
                # Select random action
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
                
                # Clip rewards
                clipped_reward = np.sign(total_reward)
                
                # Store transition in memory
                self.memory.add(state, action, clipped_reward, next_state, done)
                
                # Move to the next state
                state = next_state
                frame_count += 1
                
                # Update progress bar
                progress_bar.update(1)
            
        progress_bar.close()
        print("Replay memory filled!")
        
    
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
    
######################################################################
############################# Visualization Methods
######################################################################
    
    
    def get_q_values(self, state):
        """
        For visualization: Get Q-values for a state
        """
        # If state is a numpy array with shape (H, W, C)
        if isinstance(state, np.ndarray) and state.shape[2] == 4:
            # Convert to (C, H, W) and add batch dimension
            state = np.transpose(state, (2, 0, 1))
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            return self.policy_net(state).cpu().numpy()[0]
    
    def process_observation(self, observation):
        """
        For visualization: Process a single observation through the preprocessor
        """
        return self.preprocessor.process(observation)
    
    
######################################################################
############################# Main Methods
######################################################################
    
    def train(self, num_frames):
        """
        Main training loop
        """
        # Fill replay memory before training starts
        if len(self.memory) < self.replay_start_size:
            self.fill_replay_memory()
            
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
            obs = self.reset_environment()
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
                
                # Train the model only every update_freq steps and if there are enough training samples
                self.update_counter += 1
                if self.update_counter % self.update_freq == 0 and len(self.memory) > self.batch_size:
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
                        safe_id = self.env.unwrapped.spec.id.replace('/', '_')
                        torch.save(self.policy_net.state_dict(), f"weights/dqn_{safe_id}_best.pth")
            
            # Episode finished
            episode_count += 1
            rewards_window.append(episode_reward)
            episode_rewards.append(episode_reward)
            
            # Print episode stats
            mean_reward = np.mean(rewards_window) if rewards_window else episode_reward
            tqdm.write(f"Episode {episode_count}, Frames: {frame_count}, Reward: {episode_reward:.2f}, "
                  f"Avg Reward: {mean_reward:.2f}, Epsilon: {max(self.eps_end, self.eps_start - (self.steps_done / self.eps_decay)):.4f}")
        
        progress_bar.close()
        print("\nTraining completed!")
        
        # Save final model
        safe_id = self.env.unwrapped.spec.id.replace('/', '_')
        torch.save(self.policy_net.state_dict(), f"weights/dqn_{safe_id}_final.pth")
        
        return episode_rewards, eval_rewards
    
    
    def evaluate(self, num_episodes=10, epsilon=0.05):
        """
        Evaluate the agent over several episodes
        """
        
        total_rewards = []
        
        for _ in range(num_episodes):
            obs = self.reset_environment()
            
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
    
    
######################################################################
############################# Utility Methods
######################################################################
    
    def save_weights(self, path):
        """Save the current policy network weights to a file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.policy_net.state_dict(), path)
            print(f"Successfully saved weights to {path}")
        except Exception as e:
            print(f"Error saving weights to {path}: {e}")
            

    def load_weights(self, path):
        """Load weights from a file into both policy and target networks"""
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)
            print(f"Successfully loaded weights from {path}")
            return True
        except Exception as e:
            print(f"Error loading weights from {path}: {e}")
            return False