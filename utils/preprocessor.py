import numpy as np
from collections import deque
import cv2

class DQNPreprocessor:
    """
    Complete implementation of preprocessing for DQN:
    1. Convert to grayscale
    2. Resize to 84x84
    3. Stack 4 consecutive frames
    
    Note: BreakoutDeterministic-v4 handles frame skipping but NOT frame stacking!
    """
    def __init__(self, stack_size=4):
        self.stack_size = stack_size
        self.frame_buffer = deque(maxlen=stack_size)
        self.is_initialized = False

    def preprocess_frame(self, frame):
        """Preprocess a single frame: grayscale + downsample + resize"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        return resized
    
    def reset(self):
        """Reset the frame buffer"""
        self.frame_buffer.clear()
        self.is_initialized = False
    
    def process(self, frame):
        """
        Process a frame: preprocess it and add to the stack.
        Returns a stacked state (84x84x4) if enough frames are collected.
        """
        # Preprocess the frame
        processed_frame = self.preprocess_frame(frame)
        
        # Add to frame buffer
        self.frame_buffer.append(processed_frame)
        
        # If buffer not fully initialized, duplicate the frame
        if not self.is_initialized:
            while len(self.frame_buffer) < self.stack_size:
                self.frame_buffer.append(processed_frame)
            self.is_initialized = True
        
        # Stack frames (last dimension is time/channels)
        stacked_state = np.stack(self.frame_buffer, axis=2)  # Shape: (84, 84, 4)
        
        return stacked_state