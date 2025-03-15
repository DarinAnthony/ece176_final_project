import torch
import random
import numpy as np

class GPUReplayBuffer:
    """
    ReplayBuffer that stores everything directly on the GPU.
    Only recommended if you have sufficient GPU memory (16GB+)
    
    This implementation is optimized for the BreakoutDeterministic-v4 environment
    which already handles frame stacking.
    """
    def __init__(self, capacity, frame_shape=(4, 84, 84), device="cuda"):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0
        
        # Pre-allocate tensors on GPU
        # Store frames in half precision (float16) to save memory
        self.states = torch.zeros((capacity, *frame_shape), dtype=torch.float16, device=device)
        self.next_states = torch.zeros((capacity, *frame_shape), dtype=torch.float16, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.int32, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        with torch.no_grad():
            # Convert numpy arrays to tensors if needed
            if isinstance(state, np.ndarray):
                # If in NHWC format, transpose to NCHW
                if state.shape[-1] == 4:
                    state = np.transpose(state, (2, 0, 1))  # HWC -> CHW
                # Convert to tensor
                state = torch.tensor(state, dtype=torch.float16, device=self.device)
            elif state.dtype != torch.float16:
                state = state.to(dtype=torch.float16)
                
            if isinstance(next_state, np.ndarray):
                # If in NHWC format, transpose to NCHW
                if next_state.shape[-1] == 4:
                    next_state = np.transpose(next_state, (2, 0, 1))  # HWC -> CHW
                # Convert to tensor
                next_state = torch.tensor(next_state, dtype=torch.float16, device=self.device)
            elif next_state.dtype != torch.float16:
                next_state = next_state.to(dtype=torch.float16)
            
            # Remove batch dimension if present
            if state.dim() == 4 and state.shape[0] == 1:
                state = state.squeeze(0)
            if next_state.dim() == 4 and next_state.shape[0] == 1:
                next_state = next_state.squeeze(0)
                
            # Store in buffer
            self.states[self.position] = state
            self.actions[self.position] = action
            self.rewards[self.position] = reward
            self.next_states[self.position] = next_state
            self.dones[self.position] = done
            
            # Update position and size
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        # Get batch from buffer (already on GPU)
        states = self.states[indices].float()  # Convert from float16 to float32 for training
        actions = self.actions[indices].long()
        rewards = self.rewards[indices]
        next_states = self.next_states[indices].float()  # Convert from float16 to float32
        dones = self.dones[indices].float()  # Convert bool to float
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return the current size of internal memory."""
        return self.size