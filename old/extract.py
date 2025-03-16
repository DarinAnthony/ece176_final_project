import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Path to your log file
log_file = "extract.txt"

# Define frames per epoch (100k frames per epoch)
FRAMES_PER_EPOCH = 100000  

# Regex pattern to extract Frames and Avg Reward
pattern = r"Training:.*?\| (\d+)/\d+.*?Avg Reward: ([\d\.]+)"

# Store extracted values
frames = []
avg_rewards = []

with open(log_file, "r") as file:
    for line in file:
        match = re.search(pattern, line)
        if match:
            frame_count = int(match.group(1))  # Extract frame count
            avg_reward = float(match.group(2))  # Extract avg reward
            
            frames.append(frame_count)
            avg_rewards.append(avg_reward)

# Convert frames to epochs (each epoch = 100k frames)
epochs = np.array(frames) // FRAMES_PER_EPOCH
avg_rewards = np.array(avg_rewards)

# Ensure we have the correct number of epochs (~48)
max_epochs = 48
valid_indices = epochs < max_epochs
epochs = epochs[valid_indices]
avg_rewards = avg_rewards[valid_indices]

# Reduce the number of points displayed (take every 3rd point)
epochs = epochs[::3]
avg_rewards = avg_rewards[::3]

# Apply a moving average for smoothing (window size = 3)
window_size = 3
smoothed_rewards = np.convolve(avg_rewards, np.ones(window_size)/window_size, mode='valid')

# Convert to DataFrame
df = pd.DataFrame({"Epoch": epochs[:len(smoothed_rewards)], "Avg Reward": smoothed_rewards})

# Save as CSV for later use
df.to_csv("avg_reward_data.csv", index=False)

# Plot settings
plt.figure(figsize=(8, 5))
plt.plot(df["Epoch"], df["Avg Reward"], marker='o', linestyle='-', markersize=0.8, linewidth=0.4, color='navy')

# Labels and title
plt.xlabel("Training Epochs", fontsize=14)
plt.ylabel("Average Reward per Episode", fontsize=14)
plt.title("Average Reward on Breakout", fontsize=16)

# Formatting
plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show plot
plt.show()

# Save plot
plt.savefig("train_runs/breakout_dqn_training_avg_reward_plot.png", dpi=300)
