# PPO Racing Agent

> A custom implementation of Proximal Policy Optimization (PPO) in PyTorch for a modified Gymnasium CarRacing environment using engineered numerical observations instead of raw image inputs.

---

## Overview

Most reinforcement learning solutions for the Gymnasium CarRacing environment learn directly from raw 96×96 RGB images. While effective, image-based observations require convolutional neural networks and significantly increase computational complexity.

This project explores an alternative approach by replacing raw pixel observations with a compact numerical observation vector containing carefully engineered geometric and kinematic features.

The environment is based on the official **Gymnasium CarRacing** environment but has been extensively modified to support numerical observations, custom reward shaping, and improved reinforcement learning experimentation.

A complete Proximal Policy Optimization (PPO) algorithm was implemented in PyTorch to train and evaluate the agent.

---

## Project Highlights

- Custom PPO implementation in PyTorch
- Modified Gymnasium CarRacing environment
- Numerical observation space
- Engineered path-angle cues
- Heading error computation
- Custom reward shaping
- Beta-distribution policy for continuous actions
- Generalized Advantage Estimation (GAE)
- PPO clipped objective
- Model checkpointing
- Training reward visualization
- Evaluation script

---

## Project Origin

This project is built upon the **Gymnasium CarRacing** environment developed by the Farama Foundation.

The original environment was extensively modified to support a different reinforcement learning pipeline.

Major modifications include:

- Replacing RGB image observations with engineered numerical observations
- Designing a compact observation vector
- Adding path-angle cues
- Computing heading error
- Modifying the reward function
- Implementing a complete PPO training pipeline in PyTorch
- Adding checkpointing and evaluation utilities

The original Gymnasium project provides the simulation environment, while this repository focuses on the reinforcement learning algorithm and environment modifications.

---

## Repository Structure

```text
ppo-racing-agent/
│
├── models/
├── train_agent.py
├── test_agent.py
├── test.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Observation Space

Instead of processing raw RGB images, the agent receives a numerical observation vector describing the current driving state.

The observation vector contains features including:

- Vehicle speed
- Heading error
- Path-angle cues
- Track-relative information
- Additional geometric driving features

This representation greatly reduces the observation dimensionality while preserving the information required for effective driving.

---

## Reward Function

The reward function is designed to encourage:

- Staying on the racing track
- Following the track direction
- Smooth steering behaviour
- Forward progress
- Efficient lap completion
- Avoiding off-track driving

The reward function was refined through multiple iterations during development.

---

## PPO Implementation

The training algorithm is a custom implementation of Proximal Policy Optimization (PPO) written entirely in PyTorch.

Implemented components include:

- Actor-Critic neural network
- Beta action distribution
- Generalized Advantage Estimation (GAE)
- PPO clipped surrogate objective
- Entropy regularization
- Gradient clipping
- Mini-batch optimization
- Model checkpointing
- Resume training support

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<YOUR_USERNAME>/ppo-racing-agent.git
cd ppo-racing-agent
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python train_agent.py
```

---

## Evaluation

```bash
python test_agent.py
```

---

## Results

The repository includes:

- Trained PPO checkpoints
- Training reward curves
- Evaluation script

Future updates will include:

- Demonstration GIF
- Environment screenshots
- Training plots
- Architecture diagrams

---

## Future Work

- Parallel environment training
- Curriculum learning
- Hyperparameter optimization
- Domain randomization
- Multi-track generalization
- Performance benchmarking

---

## Technologies Used

- Python
- PyTorch
- Gymnasium
- Box2D
- NumPy
- Matplotlib

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- Gymnasium (Farama Foundation)
- PyTorch
- Box2D
- Stable-Baselines3 (used as a reinforcement learning reference during development)