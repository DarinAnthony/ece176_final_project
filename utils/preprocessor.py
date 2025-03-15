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