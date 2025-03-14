# ece176_final_project
This repository holds experiments run on the Atari games using Deep Reinforcement Learning algorithms such as DQN, using various different CNN architectures

# FOLDER STRUCTURE

project/
├── networks/
│   ├── baseline_cnn.py      # Original DQN architecture
│   ├── resnet_cnn.py        # ResNet-style architecture
│   └── attention_cnn.py     # Attention-based architecture
├── train.py                 # Main training script
├── evaluate.py              # Evaluation script
└── utils/
    ├── preprocessing.py     # Custom preprocessing if needed
    └── visualization.py     # For visualizing results

# Preprocessing

The original Atari 2600 frames are 210 x 160 pixel images with 3 channels (RGB)

The preprocessing step goes as follows:

- In the Nature paper, they first take the maximum value for each pixel color across the current and previous frame (frame-stacking) to remove flickering artifacts from the Atari emulator
- Convert RGB to grayscale (extract the Y/luminance channel)
- Downsample to a smaller size (84×84 pixels)

**Frame stacking**

: The final input representation combines multiple preprocessed frames:

- They stack the last 4 preprocessed frames to create an 84×84×4 tensor
- This provides temporal information to the network (motion detection)
- This is why the input has 4 channels despite starting with RGB images

Each channel represents a different time step (not color channels anymore, but consecutive grayscale frames)