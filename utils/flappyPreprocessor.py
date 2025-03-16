import numpy as np
from collections import deque
import cv2

class FlappyBirdPreprocessor:
    def __init__(self, stack_size=4):
        self.stack_size = stack_size
        self.frame_buffer = deque(maxlen=stack_size)
        self.is_initialized = False

    def preprocess_frame(self, frame):
        """Preprocess specifically for Flappy Bird"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize to 80x80 (as in the example)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Apply binary thresholding
        _, binary = cv2.threshold(resized, 1, 255, cv2.THRESH_BINARY)
        
        return binary
    
    def reset(self):
        """Reset the frame buffer"""
        self.frame_buffer.clear()
        self.is_initialized = False
    
    def process(self, frame):
        """Process a frame and add to the stack"""
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
        stacked_state = np.stack(self.frame_buffer, axis=2)  # Shape: (80, 80, 4)
        
        return stacked_state
    
    def _add_frame_to_buffer(self, frame):
        """Helper method to add a frame to the buffer"""
        self.frame_buffer.append(frame)
        if not self.is_initialized:
            while len(self.frame_buffer) < self.stack_size:
                self.frame_buffer.append(frame)
            self.is_initialized = True
        
        return np.stack(self.frame_buffer, axis=2)