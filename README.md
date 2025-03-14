# ece176_final_project
This repository holds experiments run on the Atari games using Deep Reinforcement Learning algorithms such as DQN, using various different CNN architectures

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