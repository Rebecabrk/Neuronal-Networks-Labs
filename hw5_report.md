# Agent Training Report


## CNN Architecture
- **Conv1**: 1 input channel, 32 output channels, kernel size 8, stride 4
- **Conv2**: 32 input channels, 64 output channels, kernel size 4, stride 2
- **Conv3**: 64 input channels, 64 output channels, kernel size 3, stride 1
- **FC1**: 512 hidden units
- **Output Layer**: Number of actions

## Q-Learning Parameters
- **Gamma (Discount Factor)**: 0.99
- **Epsilon (Exploration)**: Start = 1.0, End = 0.1, Decay = 0.995
- **Learning Rate**: 0.0001
- **Replay Buffer Size**: 50,000
- **Batch Size**: 32

## Training Configuration
- **Environment**: FlappyBird-v0
- **Episodes**: 1000
- **Target Network Update Frequency**: After each episode

![alt text](image.png)