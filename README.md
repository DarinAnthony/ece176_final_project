# ECE176 Final Project
This repository holds experiments run on the Atari games using Deep Reinforcement Learning algorithms such as DQN, using various different CNN architectures

# Folder Structure

    project/

    ├── models/

        ├── base.py      # DQN 2013 architecture

        ├── base.py      # DQN 2015 architecture

        ├── resnet_cnn.py        # ResNet-style architecture

        └── attention_cnn.py     # Attention-based architecture

    ├── agents/

        ├── base.py      # Original Q-Learning Agent from 2015

    ├── utils/

        ├── preprocessing.py     # Methods for preprocessing images before passing into the QNetwork

        └── visualization.py     # Methods for visualizing results

        ├── replayBuffer.py     # Implementation of the replay buffer

    ├── weights/

        ├── dqn_ALE_Breakout-v5_final.pth      # Stored weights from a training run

    ├── runs/

        ├── videos/

            ├── video1.mp4

            ├── ...

        ├── metrics/

            ├── metric1.png

            ├── ...

    ├── train.py                 # Main training script

    └── evaluate.py              # Evaluation script


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

# Model Architecture



# Notes

For each file such as the preprocessor/visualizer, there may be an example usage section in the bottom you may use to understand how it is used at an abstracted level