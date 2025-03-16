import gymnasium as gym
from flappy_bird_env import flappy_bird_env
from gymnasium.envs.registration import register
import cv2
import numpy as np
import time

# Register the environment if not already done
try:
    register(
        id="FlappyBird-v0",
        entry_point="flappy_bird_env.flappy_bird_env:FlappyBirdEnv",
        max_episode_steps=1000,
    )
except:
    pass  # Already registered

# Create the environment with rgb_array render mode
env = gym.make("FlappyBird-v0", render_mode="rgb_array")

# Reset the environment to get the initial observation
observation, info = env.reset()

print(f"Observation shape: {observation.shape}")
print(f"Observation dtype: {observation.dtype}")
print(f"Observation min/max: {observation.min()}/{observation.max()}")

# Run a simple loop using only observations
done = False
step_count = 0
total_reward = 0

# Create a window
cv2.namedWindow('Flappy Bird Observation', cv2.WINDOW_NORMAL)

while not done:
    # Display the observation directly
    display_frame = cv2.cvtColor(observation.astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    # Add some info text
    cv2.putText(display_frame, f"Step: {step_count}, Score: {total_reward:.1f}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow('Flappy Bird Observation', display_frame)
    
    # Wait for key press - essential for OpenCV window update
    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break
    
    # Take a random action with 20% chance to jump
    action = 1 if np.random.random() < 0.2 else 0
    
    # Step the environment
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Update counters
    step_count += 1
    total_reward += reward
    
    # Print some information
    print(f"Step: {step_count}, Action: {action}, Reward: {reward:.3f}, Total: {total_reward:.1f}", end="\r")
    
    # Check if episode is done
    done = terminated or truncated
    
    # Small delay for better visualization
    time.sleep(0.01)

print("\nGame finished!")
print(f"Total steps: {step_count}, Final score: {total_reward:.1f}")
cv2.destroyAllWindows()
env.close()