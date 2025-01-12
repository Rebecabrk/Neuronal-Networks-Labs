
# Flappy Bird Agent Training Report

## First Model Architecture
### CNN Architecture
- **Conv1**: 1 input channel, 32 output channels, kernel size 8, stride 4
- **Conv2**: 32 input channels, 64 output channels, kernel size 4, stride 2
- **Conv3**: 64 input channels, 64 output channels, kernel size 3, stride 1
- **FC1**: 512 hidden neurons
- **Output Layer**: Number of actions = 2

### Q-Learning Parameters
- **Gamma (Discount Factor)**: 0.99
- **Epsilon (Exploration)**: Start = 1.0, End = 0.1, Decay = 0.995
- **Learning Rate**: 0.0001
- **Replay Buffer Size**: 10,000
- **Batch Size**: 32

### Training Configuration
- **Environment**: FlappyBird-v0, render_mode='rgb_array'
- **Episodes**: 1000
- **Replay Buffer**
- **Epsilon Greedy**
- **Image Preprocessing** : Crop, Resize, Grayscale, Pixel Normalization, Thresholding

## Second Model Architecture
### CNN Architecture (identical to the previous one)
- **Conv1**: 1 input channel, 32 output channels, kernel size 8, stride 4
- **Conv2**: 32 input channels, 64 output channels, kernel size 4, stride 2
- **Conv3**: 64 input channels, 64 output channels, kernel size 3, stride 1
- **FC1**: 512 hidden neurons
- **Output Layer**: Number of actions = 2

### Q-Learning Parameters (identical to the previous one)
- **Gamma (Discount Factor)**: 0.99
- **Epsilon (Exploration)**: Start = 1.0, End = 0.1, Decay = 0.995
- **Learning Rate**: 0.0001
- **Replay Buffer Size**: 50,000
- **Batch Size**: 32

### Training Configuration
- **Environment**: FlappyBird-v0
- **Episodes**: 1000
- **Target Network **: Update Frequency = after each episode
- **Replay Buffer**
- **Epsilon Greedy**
- **Image Preprocessing** : Resize, Pixel Normalization

## Third Model Architecture
### CNN Architecture (identical to the previous one)
- **Conv1**: 1 input channel, 32 output channels, kernel size 8, stride 4
- **Conv2**: 32 input channels, 64 output channels, kernel size 4, stride 2
- **Conv3**: 64 input channels, 64 output channels, kernel size 3, stride 1
- **FC1**: 512 hidden neurons
- **Output Layer**: Number of actions = 2

### Q-Learning Parameters (identical to the previous one)
- **Gamma (Discount Factor)**: 0.99
- **Epsilon (Exploration)**: Start = 1.0, End = 0.1, Decay = 0.995
- **Learning Rate**: 0.0001
- **Replay Buffer Size**: 50,000
- **Batch Size**: 32

### Training Configuration
- **Environment**: FlappyBird-v0, render_mode='rgb_array'
- **Episodes**: 1000
- **Target Network **: Update Frequency = after each episode
- **Replay Buffer**
- **Epsilon Greedy**
- **Image Preprocessing** : Grayscale, Resize, Pixel Normalization
