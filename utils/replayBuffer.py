import torch
import random
import numpy as np

class SequentialGPUReplayBuffer:
    """
    Memory-optimized GPU ReplayBuffer that eliminates state duplication.
    Uses a single tensor for states and reconstructs state-next_state pairs when sampling.
    
    Reduces memory usage by ~50% compared to standard implementations.
    """
    def __init__(self, capacity, frame_shape=(4, 84, 84), device="cuda"):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0
        self.frame_shape = frame_shape
        
        # Allocate tensors with progress updates
        print(f"Allocating sequential replay buffer with capacity {capacity} on {device}...")
        
        # Store only unique states (we need capacity+1 to handle the last next_state)
        print("Allocating sequential states tensor...")
        self.states = torch.zeros((capacity+1, *frame_shape), dtype=torch.uint8, device=device)
        
        print("Allocating actions tensor...")
        self.actions = torch.zeros(capacity, dtype=torch.uint8, device=device)
        
        print("Allocating rewards tensor...")
        self.rewards = torch.zeros(capacity, dtype=torch.int8, device=device)
        
        print("Allocating dones tensor...")
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        
        # Track episode boundaries (where next_state doesn't correspond to state[i+1])
        self.episode_ends = set()
        
        print("Sequential buffer allocation complete!")
        
        # Calculate memory usage
        states_mem = self.states.element_size() * self.states.nelement() / (1024**3)
        actions_mem = self.actions.element_size() * self.actions.nelement() / (1024**3)
        rewards_mem = self.rewards.element_size() * self.rewards.nelement() / (1024**3)
        dones_mem = self.dones.element_size() * self.dones.nelement() / (1024**3)
        total_mem = states_mem + actions_mem + rewards_mem + dones_mem
        
        print(f"Estimated GPU memory usage: {total_mem:.2f} GiB")
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        with torch.no_grad():
            # Format state
            if isinstance(state, np.ndarray):
                # If in NHWC format, transpose to NCHW
                if state.shape[-1] == 4:
                    state = np.transpose(state, (2, 0, 1))
                
                # Convert to uint8 if needed
                if state.dtype == np.float32 or state.dtype == np.float64:
                    if state.max() <= 1.0:
                        state = (state * 255).astype(np.uint8)
                
                # Convert to tensor
                state = torch.tensor(state, dtype=torch.uint8, device=self.device)
            elif state.dtype != torch.uint8:
                if state.dtype == torch.float32 or state.dtype == torch.float16:
                    if state.max() <= 1.0:
                        state = (state * 255).to(torch.uint8)
            
            # Format next_state
            if isinstance(next_state, np.ndarray):
                if next_state.shape[-1] == 4:
                    next_state = np.transpose(next_state, (2, 0, 1))
                
                if next_state.dtype == np.float32 or next_state.dtype == np.float64:
                    if next_state.max() <= 1.0:
                        next_state = (next_state * 255).astype(np.uint8)
                
                next_state = torch.tensor(next_state, dtype=torch.uint8, device=self.device)
            elif next_state.dtype != torch.uint8:
                if next_state.dtype == torch.float32 or next_state.dtype == torch.float16:
                    if next_state.max() <= 1.0:
                        next_state = (next_state * 255).to(torch.uint8)
            
            # Remove batch dimension if present
            if state.dim() == 4 and state.shape[0] == 1:
                state = state.squeeze(0)
            if next_state.dim() == 4 and next_state.shape[0] == 1:
                next_state = next_state.squeeze(0)
                
            # Store current state
            self.states[self.position] = state
            
            # Store action, reward, done
            self.actions[self.position] = action
            self.rewards[self.position] = np.sign(reward) if isinstance(reward, (int, float)) else reward
            self.dones[self.position] = done
            
            # If end of episode, store the next_state separately
            if done:
                self.episode_ends.add(self.position)
                next_position = (self.position + 1) % (self.capacity + 1)
                self.states[next_position] = next_state
            
            # Update position and size
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        # Sample indices - don't sample from episode boundaries
        valid_indices = [i for i in range(min(self.size, self.capacity)) 
                         if i not in self.episode_ends]
        if len(valid_indices) < batch_size:
            # If not enough valid indices, allow duplicates
            indices = random.choices(valid_indices, k=batch_size)
        else:
            indices = random.sample(valid_indices, batch_size)
        
        # Convert to tensor for indexing
        indices_tensor = torch.tensor(indices, device=self.device)
        
        # Get states
        states = self.states[indices_tensor].float() / 255.0
        
        # Get next_states (just index i+1)
        next_indices = (indices_tensor + 1) % (self.capacity + 1)
        next_states = self.states[next_indices].float() / 255.0
        
        # Get other elements
        actions = self.actions[indices_tensor].long()
        rewards = self.rewards[indices_tensor].float()
        dones = self.dones[indices_tensor].float()
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return the current size of internal memory."""
        return self.size