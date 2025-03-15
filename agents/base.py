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
                 weights_path=None,
                 eval_interval=250000,
                 save_interval=200000):
        
        self.env = env
        self.device = device
        self.memory_size = memory_size
        self.memory = replayBufferClass(self.memory_size, self.device)
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
        
        # Optimizer
        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(), 
            lr=learning_rate,
            alpha=0.95,  # squared gradient momentum
            eps=0.01,    # min squared gradient
            momentum=0.95,  # gradient momentum
            centered=False
            )
        
        # Load pre-trained weights if provided
        if weights_path is not None:
            _ = self.load_model(weights_path)
        else:
            # Initialize target network with policy network weights
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        self.target_net.eval()
        
        # Evaluation parameter
        self.eval_interval = eval_interval
        
        # Saving parameter
        self.save_interval = save_interval
        
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
############################# Visualization and Evaluation Methods
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
        
        
    def run_episode(self, max_steps=10000, record=False, visualizer=None, evaluate=True):
        """
        Run a single episode and optionally record it.
        
        Args:
            max_steps: Maximum number of steps per episode
            record: Whether to record the episode
            visualizer: Optional visualizer for recording
            evaluate: Whether to run in evaluation mode
        
        Returns:
            tuple: (total_reward, episode_length, actions_taken, q_values)
        """
        obs = self.reset_environment()
        state = self.get_state(obs)
        
        episode_reward = 0
        steps = 0
        done = False
        
        # Metrics collection
        actions_taken = []
        q_values = []
        
        while not done and steps < max_steps:
            # Select action
            action = self.select_action(state, evaluate=evaluate)
            actions_taken.append(action)
            

            q_values.append(self.get_q_values(state))
            
            # Execute action with frame skipping
            step_reward = 0
            for _ in range(self.frame_skip):
                obs, reward, terminated, truncated, info = self.env.step(action)
                step_reward += reward
                done = terminated or truncated
                if done:
                    break
            
            # Process next state
            next_state = self.get_state(obs)
            
            # Record frame if visualizer is provided
            if record and visualizer is not None:
                visualizer.add_frame(obs, state, action, step_reward, episode_reward, steps, q_values[-1] if q_values else None, info)
            
            
            # Update
            state = next_state
            episode_reward += step_reward
            steps += 1
        
        return episode_reward, steps, actions_taken, q_values
    
    
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
                    
                # Save model periodically
                if frame_count % self.save_interval == 0:
                    # modify the path name if there are instances of '/'
                    env_name = self.env.unwrapped.spec.id.replace('/', '_')
                    filename = f"weights/{env_name}_dqn_{frame_count}frames.pth"
                    self.save_model(filename, frames_trained=frame_count)
                    
                
                # Run evaluation periodically
                if frame_count % self.eval_interval == 0:
                    eval_reward = self.evaluate(num_episodes=10)
                    eval_rewards.append((frame_count, eval_reward))
                    tqdm.write(f"\nEvaluation at frame {frame_count}: {eval_reward:.2f}")
                               
                    # Save best model
                    if eval_reward > best_mean_reward:
                        best_mean_reward = eval_reward
                        
                        # modify the path name if there are instances of '/'
                        env_name = self.env.unwrapped.spec.id.replace('/', '_')
                        final_filepath = f"weights/{env_name}_dqn_best.pth"
                        self.save_model(final_filepath, frames_trained=frame_count)
            
            # Episode finished
            episode_count += 1
            rewards_window.append(episode_reward)
            episode_rewards.append(episode_reward)
            
            # Print episode stats
            mean_reward = np.mean(rewards_window) if rewards_window else episode_reward
            current_epsilon = max(self.eps_end, self.eps_start - (self.steps_done / self.eps_decay))
            tqdm.write(f"Episode {episode_count} completed | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg Reward: {mean_reward:.2f} | "
                  f"Frames: {frame_count} | "
                  f"Epsilon: {current_epsilon:.4f}")
        
        progress_bar.close()
        print("\nTraining completed!")
        
        # Save final model
        env_name = self.env.unwrapped.spec.id.replace('/', '_')
        final_filepath = f"weights/{env_name}_dqn_final.pth"
        self.save_model(final_filepath, frames_trained=frame_count)
        
        return episode_rewards, eval_rewards
    
    
    def evaluate(self, num_episodes=10):
        """
        Evaluate the agent over several episodes
        """
        
        total_rewards = []
    
        for _ in range(num_episodes):
            # Run a single episode using run_episode (no recording)
            reward, _, _, _ = self.run_episode(evaluate=True)
            total_rewards.append(reward)
        
        return np.mean(total_rewards)
    
    
######################################################################
############################# Utility Methods
######################################################################
            
    def save_model(self, filepath, frames_trained=None, additional_data=None):
        """
        Save the model weights and important training information.
        
        Args:
            filepath (str): Path where the model should be saved
            frames_trained (int, optional): Number of frames the model has been trained for
            additional_data (dict, optional): Any additional data to save with the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Prepare data to save
        save_data = {
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'frames': frames_trained if frames_trained is not None else 0,
            'hyperparams': {
                'batch_size': self.batch_size,
                'gamma': self.gamma,
                'epsilon_start': self.eps_start,
                'epsilon_end': self.eps_end,
                'epsilon_decay': self.eps_decay,
                'target_update': self.target_update,
                'update_freq': self.update_freq,
                'frame_skip': self.frame_skip,
                'no_op_max': self.no_op_max
            }
        }
      
        # Add any additional data
        if additional_data is not None and isinstance(additional_data, dict):
            save_data.update(additional_data)
        
        # Save to file
        torch.save(save_data, filepath)
        print(f"Model saved to {filepath}")
            
        
    def load_model(self, filepath):
        """
        Load a saved model and its training information.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            dict: The full checkpoint data including training information
        """
        try:
            # Load the checkpoint
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Load model weights
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
            self.target_net.load_state_dict(checkpoint['model_state_dict'])
            
            # Optionally load optimizer state if it exists and we have an optimizer
            if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Print loaded model info
            print(f"Loaded model from {filepath}")
            if 'frames' in checkpoint:
                print(f"Model was trained for {checkpoint['frames']} frames")
            if 'epsilon' in checkpoint:
                print(f"Epsilon value: {checkpoint['epsilon']}")
            if 'hyperparams' in checkpoint:
                print(f"Model hyperparameters: {checkpoint['hyperparams']}")
                
            return checkpoint
        except Exception as e:
            print(f"Error loading model from {filepath}: {e}")
            return {}