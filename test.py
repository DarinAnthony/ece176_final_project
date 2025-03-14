# test_dqn.py
import torch
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
import os

# Import your modules
from utils.preprocessor import DQNPreprocessor
from models.base import DQN
from utils.replayBuffer import ReplayBuffer
from agents.base import DQNAgent
from utils.visualizer import DQNVisualizer  # The visualizer class you shared

def quick_validation(agent, env_name, max_frames=1000):
    """
    Run a quick validation of the agent to check for obvious errors.
    """
    env = gym.make(env_name)
    obs, _ = env.reset()
    
    # Reset preprocessor
    if hasattr(agent, 'preprocessor'):
        agent.preprocessor.reset()
    
    state = agent.get_state(obs)
    
    frame_count = 0
    total_reward = 0
    done = False
    
    # Track tensors for sanity checks
    states_sample = []
    
    print("Starting quick validation...")
    
    while not done and frame_count < max_frames:
        # Select action
        action = agent.select_action(state)
        
        # Execute action
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Process next state
        next_state = agent.get_state(obs)
        
        # Store in replay buffer
        agent.memory.add(state, action, np.sign(reward), next_state, done)
        
        # Store sample for checking
        if len(states_sample) < 5:
            states_sample.append(state.cpu().numpy())
        
        # Update
        state = next_state
        total_reward += reward
        frame_count += 1
        
        # Run optimization steps
        if frame_count % 10 == 0 and len(agent.memory) > agent.batch_size:
            agent._optimize_model()
            print(f"Frame {frame_count}: Optimization step completed")
    
    print(f"Validation run completed: {frame_count} frames, reward: {total_reward}")
    
    # Sanity checks
    print("\nRunning sanity checks:")
    
    # Check state shapes
    print(f"State shape: {state.shape} (expected: [1, 4, 84, 84])")
    
    # Check replay buffer
    print(f"Replay buffer size: {len(agent.memory)}")
    
    # Check if states differ (not all identical)
    if len(states_sample) > 1:
        differences = np.mean(np.abs(states_sample[0] - states_sample[1]))
        print(f"Mean difference between states: {differences}")
        if differences < 0.01:
            print("WARNING: States appear very similar, check preprocessing")
        else:
            print("States appear to differ correctly")
    
    # Check optimization
    if len(agent.memory) > agent.batch_size:
        states, actions, rewards, next_states, dones = agent.memory.sample(agent.batch_size)
        print(f"Sample batch shapes: states {states.shape}, actions {actions.shape}")
        
        # Try forward pass
        with torch.no_grad():
            q_values = agent.policy_net(states)
            print(f"Q-values shape: {q_values.shape} (expected: [{agent.batch_size}, {agent.num_actions}])")
            print(f"Q-values range: {q_values.min().item():.4f} to {q_values.max().item():.4f}")
    
    env.close()
    return True

def mini_training(agent, env_name, num_frames=10000, eval_interval=2000):
    """
    Run a mini training session to ensure all components work together.
    """
    env = gym.make(env_name)
    progress_bar = tqdm(total=num_frames, desc="Mini training")
    
    frame_count = 0
    episode_count = 0
    rewards_history = []
    
    while frame_count < num_frames:
        # Reset environment
        obs, _ = env.reset()
        state = agent.get_state(obs)
        
        episode_reward = 0
        done = False
        
        while not done and frame_count < num_frames:
            # Select action
            action = agent.select_action(state)
            
            # Skip frames (act every k frames)
            total_reward = 0
            for _ in range(agent.frame_skip):
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
                if done:
                    break
            
            # Process new frame
            next_state = agent.get_state(obs)
            
            # Clip rewards
            clipped_reward = np.sign(total_reward)
            
            # Store transition
            agent.memory.add(state, action, clipped_reward, next_state, done)
            
            # Update
            state = next_state
            episode_reward += total_reward
            frame_count += 1
            
            # Optimize model
            if len(agent.memory) > agent.batch_size:
                agent._optimize_model()
            
            # Update target network
            if frame_count % agent.target_update == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
            progress_bar.update(1)
        
        # Episode finished
        episode_count += 1
        rewards_history.append(episode_reward)
        
        # Print stats
        print(f"\nEpisode {episode_count}, Frames: {frame_count}, Reward: {episode_reward:.2f}")
    
    progress_bar.close()
    print("Mini training completed!")
    
    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    torch.save(agent.policy_net.state_dict(), f"models/mini_train_{env_name.replace('/', '_')}.pth")
    
    return rewards_history

def visualize_agent(agent, env_name, num_episodes=2, max_steps=1000):
    """
    Create and run a visualizer to see the agent in action.
    """
    # Create visualizer
    visualizer = DQNVisualizer(
        env_name=env_name,
        agent=agent,
        output_dir='./videos',
        show_metrics=True,
        show_q_values=True,
        show_preprocessed=True
    )
    
    # Add a Q-values method to the agent for visualization if it doesn't exist
    if not hasattr(agent, 'get_q_values'):
        def get_q_values(state):
            # Convert numpy array to PyTorch tensor if it's not already a tensor
            if isinstance(state, np.ndarray):
                # Convert from HWC to CHW format and add batch dimension
                state = np.transpose(state, (2, 0, 1))  # Convert from (84,84,4) to (4,84,84)
                state = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            
            with torch.no_grad():
                return agent.policy_net(state).cpu().numpy()[0]
        agent.get_q_values = get_q_values
    
    # Add process_observation method if needed
    if not hasattr(agent, 'process_observation'):
        def process_observation(obs):
            # Call get_state but extract the numpy array before tensor conversion
            agent.preprocessor.reset()
            state = None
            while state is None:
                state = agent.preprocessor.process(obs)
                if state is None:
                    # This should handle itself in the visualizer
                    return None
            return state
        agent.process_observation = process_observation
    
    # Record episodes
    for episode in range(num_episodes):
        print(f"\nRecording episode {episode+1}...")
        episode_stats = visualizer.record_episode(
            filename=f"{env_name.replace('/', '_')}_episode_{episode+1}",
            max_steps=max_steps,
            render=True
        )
    
    # Plot performance metrics
    visualizer.plot_performance_metrics(
        filename=f"{env_name.replace('/', '_')}_performance",
        show=True
    )

def test_dqn():
    # Set up environment
    env_name = 'ALE/Breakout-v5'
    env = gym.make(env_name)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create agent
    agent = DQNAgent(
        env=env,
        replayBufferClass=ReplayBuffer,
        QNetwork=DQN,
        PreprocessorClass=DQNPreprocessor,
        device=device,
        memory_size=10000,  # Smaller for testing
        batch_size=32,
        target_update=500   # Smaller for testing
    )
    
    # Quick validation (catches basic errors)
    print("\n=== Running Quick Validation ===")
    quick_validation(agent, env_name, max_frames=500)
    
    # Mini training session (catches training-related issues)
    print("\n=== Running Mini Training ===")
    mini_training(agent, env_name, num_frames=5000)
    
    # Visualize the agent
    print("\n=== Visualizing Agent ===")
    visualize_agent(agent, env_name, num_episodes=1, max_steps=500)
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    test_dqn()