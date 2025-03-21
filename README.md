# ECE176 Final Project
Youtube video can be found here: https://www.youtube.com/watch?v=oFNBHbHPo6M

This repository holds experiments run on the Atari games using Deep Reinforcement Learning algorithms such as DQN, using various different CNN architectures

# Folder Structure

    project/

    ├── models/

        ├── base.py      # DQN 2013 architecture

        ├── dqn2015.py      # DQN 2015 architecture

        └── ...            # other attempts at modifying architecture

    ├── agents/

        ├── base.py      # Original Q-Learning Agent from 2015

        └── flappy.py      # Agent inherited from base for different optimizer

    ├── utils/

        ├── preprocessing.py     # Methods for preprocessing images before passing into the QNetwork

        └── visualization.py     # Methods for visualizing results

        ├── replayBuffer.py     # Implementation of the replay buffer

    ├── weights/

        ├── dqn_ALE_Breakout-v5_final.pth      # Stored weights from a training run

    ├── eval_runs/

        ├── game1/

            ├── videos/
    
                ├── video1.mp4
    
                ├── ...
    
            ├── metrics/
    
                ├── metric1.png
    
                ├── ...

        ├── game2/

            ├── ...

    ├── train.py                 # Main training script

    ├── train.ipynb              # Main training script in notebook format (for Google Colab GPU access)

    ├── ...                      # Other training scripts for other environments

    ├── evaluate.py              # Main evaluation script

    └── evaluate.py              # Main evaluation script


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


# References

Flappy bird gym environment:
https://github.com/robertoschiavone/flappy-bird-env

The FASTEST introduction to Reinforcement Learning on the internet: https://www.youtube.com/watch?v=VnpRp7ZglfA
RL Course by David Silver - Lecture 6: Value Function Approximation: https://www.youtube.com/watch?v=UoPei5o4fps

Papers:



