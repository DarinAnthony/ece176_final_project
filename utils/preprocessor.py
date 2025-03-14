import numpy as np
import cv2
from collections import deque

class DQNPreprocessor:
    """
    Complete implementation of the preprocessing steps as described in the DQN papers:
    1. Take max of two consecutive frames to remove flickering
    2. Convert to grayscale
    3. Resize to 84x84
    4. Stack 4 consecutive frames
    """
    def __init__(self, frame_count=4):
        """
        Initialize the preprocessor.
        
        Parameters:
        -----------
        frame_count : int
            Number of frames to stack (default: 4 as in the DQN papers)
        """
        self.frame_count = frame_count
        self.frame_buffer = deque(maxlen=2)  # For max-pooling between frames
        self.state_buffer = deque(maxlen=frame_count)  # For frame stacking
        self.is_initialized = False
    
    def preprocess_frame(self, frame):
        """
        Preprocess a single frame:
        - Convert to grayscale
        - Resize to 84x84
        - Normalize to [0, 1]
        
        Parameters:
        -----------
        frame : numpy.ndarray
            Raw RGB frame from Atari environment (210x160x3)
        
        Returns:
        --------
        numpy.ndarray
            Preprocessed frame (84x84)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        normalized = resized / 255.0
        
        return normalized
    
    def process_frame_pair(self):
        """
        Take the maximum value for each pixel between the two most recent frames
        to remove flickering (as described in the Nature paper).
        
        Returns:
        --------
        numpy.ndarray
            Max-pooled frame
        """
        if len(self.frame_buffer) == 2:
            # Take the maximum between the two most recent frames
            return np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        else:
            # If only one frame is available, return it
            return self.frame_buffer[0]
    
    def process(self, frame):
        """
        Process a new frame:
        1. Preprocess the frame
        2. Add it to the frame buffer
        3. Apply max-pooling if possible
        4. Add processed frame to state buffer
        5. Return stacked state if enough frames are collected
        
        Parameters:
        -----------
        frame : numpy.ndarray
            Raw RGB frame from Atari environment
        
        Returns:
        --------
        numpy.ndarray or None
            Stacked state (84x84x4) if enough frames are collected, None otherwise
        """
        # Preprocess the frame
        processed_frame = self.preprocess_frame(frame)
        
        # Add to frame buffer for max-pooling
        self.frame_buffer.append(processed_frame)
        
        # Apply max-pooling if possible
        if len(self.frame_buffer) > 1:
            max_pooled = self.process_frame_pair()
        else:
            max_pooled = processed_frame
        
        # Add to state buffer for stacking
        self.state_buffer.append(max_pooled)
        
        # If we don't have enough frames yet, fill the buffer with the same frame
        if not self.is_initialized:
            while len(self.state_buffer) < self.frame_count:
                self.state_buffer.append(max_pooled)
            self.is_initialized = True
        
        # Stack frames and return state
        if len(self.state_buffer) == self.frame_count:
            return np.stack(self.state_buffer, axis=2)  # Stack along third dimension (84x84x4)
        else:
            return None
    
    def reset(self):
        """
        Reset the preprocessor state.
        """
        self.frame_buffer.clear()
        self.state_buffer.clear()
        self.is_initialized = False


class DQNAgent:
    """
    Simple DQN Agent skeleton that uses the DQNPreprocessor.
    """
    def __init__(self, state_size=(84, 84, 4), action_size=4):
        """
        Initialize the DQN Agent.
        
        Parameters:
        -----------
        state_size : tuple
            Shape of the state (default: (84, 84, 4) as in the DQN papers)
        action_size : int
            Number of possible actions
        """
        self.state_size = state_size
        self.action_size = action_size
        self.preprocessor = DQNPreprocessor()
        
        # Here you would initialize your Q-network, replay buffer, etc.
        print(f"DQN Agent initialized with state size {state_size} and action size {action_size}")
    
    def process_observation(self, observation):
        """
        Process a new observation from the environment.
        
        Parameters:
        -----------
        observation : numpy.ndarray
            Raw RGB frame from Atari environment
        
        Returns:
        --------
        numpy.ndarray or None
            Stacked state if enough frames are collected, None otherwise
        """
        return self.preprocessor.process(observation)
    
    def reset(self):
        """
        Reset the agent between episodes.
        """
        self.preprocessor.reset()
        
    def select_action(self, state):
        """
        Select an action based on the current state.
        This is just a placeholder - in a real implementation this would use the Q-network.
        
        Parameters:
        -----------
        state : numpy.ndarray
            Stacked state (84x84x4)
        
        Returns:
        --------
        int
            Selected action
        """
        # In a real implementation, this would use the Q-network
        return np.random.randint(self.action_size)


# Example usage
def example_usage():
    """
    Demonstrate how to use the DQN preprocessor with an Atari environment.
    """
    import gymnasium as gym
    import matplotlib.pyplot as plt
    
    # Create environment
    env = gym.make('ALE/Breakout-v5')
    
    # Create agent
    agent = DQNAgent(action_size=env.action_space.n)
    
    # Reset environment
    observation, _ = env.reset()
    agent.reset()
    
    # Process first frame
    state = agent.process_observation(observation)
    
    # Take some random steps to collect more frames
    for i in range(5):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        next_state = agent.process_observation(observation)
        
        if terminated or truncated:
            break
        
        state = next_state
    
    # Visualize the stacked frames
    if state is not None:
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        
        # Original frame
        axes[0].imshow(observation)
        axes[0].set_title('Original Frame')
        axes[0].axis('off')
        
        # Stacked frames
        for i in range(4):
            axes[i+1].imshow(state[:, :, i], cmap='gray')
            axes[i+1].set_title(f'Stacked Frame {i+1}')
            axes[i+1].axis('off')
        
        plt.tight_layout()
        plt.savefig('dqn_preprocessing.png')
        plt.show()
    
    env.close()


if __name__ == "__main__":
    example_usage()