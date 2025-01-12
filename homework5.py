import torch
import torch.nn as nn
import torch.nn.functional as F

import random
from collections import deque
import numpy as np
import torch.optim as optim
import cv2

import gymnasium as gym
import flappy_bird_gymnasium
import matplotlib.pyplot as plt
import cv2

GAMMA = 0.99
LEARNING_RATE = 1e-4
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
REPLAY_BUFFER_SIZE = 50000
BATCH_SIZE = 32

class CNN(nn.Module):
    def __init__(self, action_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ReplayBuffer:
  def __init__(self, capacity):
    self.buffer = deque(maxlen=capacity)

  def add(self, experience):
    self.buffer.append(experience)

  def sample(self, batch_size):
    return random.sample(self.buffer, batch_size)

  def size(self):
    return len(self.buffer)

class QLearningAgent:
  def __init__(self, input_shape, num_actions):
    self.model = CNN(input_shape, num_actions)
    self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
    self.criterion = nn.MSELoss()
    self.gamma = GAMMA
    self.epsilon = EPSILON_START
    self.epsilon_decay = EPSILON_DECAY
    self.epsilon_min = EPSILON_END
    self.replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE)

  def select_action(self, state):
    if np.random.rand() <= self.epsilon:
      return np.random.randint(0, env.action_space.n)
    state = torch.FloatTensor(state).unsqueeze(0)
    q_values = self.model(state)
    return torch.argmax(q_values).item()

  # def train(self, state, action, reward, next_state, done): online training
  def train(self, batch_size):
    if self.replay_buffer.size() < batch_size:
      return

    batch = self.replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    q_values = self.model(states)
    next_q_values = self.model(next_states)
    target_q_values = rewards + self.gamma * torch.max(next_q_values, dim=1)[0] * (1-dones)

    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = self.criterion(q_values, target_q_values)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

  def save(self, filename):
        torch.save(self.model.state_dict(), filename)

  def load(self, filename):
        self.model.load_state_dict(torch.load(filename))


def crop(image, x_start=0, y_start=0, x_end=288, y_end=410):
  return image[y_start:y_end, x_start:x_end]

def resize(image): 
  return cv2.resize(image, (84, 84))

def convert_to_grayscale(image):
  return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def normalize_pixel_values(image):
  return image / 255.0

def apply_threshold(image, threshold=0.5):
  _, binary_image = cv2.threshold(image, threshold, 1, cv2.THRESH_BINARY)
  return binary_image

def preprocess_image(image):
  image = crop(image)
  image = resize(image)
  image = convert_to_grayscale(image)
  image = normalize_pixel_values(image)
  image = apply_threshold(image)
  return image


env = gym.make('FlappyBird-v0', render_mode='rgb_array')
input_shape = (1, 178, 125)
num_actions = env.action_space.n
agent = QLearningAgent(input_shape=input_shape, num_actions=num_actions)
state = env.reset()

for episode in range(1000):
  state = env.render()
  state = preprocess_image(state)
  state = np.expand_dims(state, axis=0) # adds a dimension on the first position for batch size
  total_reward = 0
  best_total_reward = float('-inf')

  done = False
  while not done:
    action = agent.select_action(state)
    next_state, reward, done, _, _ = env.step(action)
    next_state = env.render()
    next_state = preprocess_image(next_state)
    next_state = np.expand_dims(next_state, axis=0)
    total_reward += reward

    agent.replay_buffer.add((state, action, reward, next_state, done))
    agent.train(batch_size=BATCH_SIZE)

    state = next_state

    if action == 1: # jumping
      for _ in range(3): # skip 3 frames
        next_state, reward, done, _, _ = env.step(1 - action)
        if done:
          break
  if total_reward > best_total_reward:
        best_total_reward = total_reward
        agent.save("flappy_bird_best_model.pth")        

  print(f'Episode {episode} completed, total reward: {total_reward}')

def create_evaluation_video(agent, env, output_path="evaluation_video.mp4", max_steps=500):
    # Load the model from the file
    model_path = "flappy_bird_best_model.pth"
    agent.load(model_path)
    agent.model.eval()  # Set the model to evaluation mode

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (env.render().shape[1], env.render().shape[0]))
    
    state, _ = env.reset()
    state = env.render()
    state = preprocess_image(state)
    state = np.expand_dims(state, axis=0)
    total_reward = 0

    for _ in range(max_steps):
        frame = env.render()
        video_writer.write(frame)
        
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = env.render()
        next_state = preprocess_image(next_state)
        next_state = np.expand_dims(next_state, axis=0)
        total_reward += reward
        state = next_state
        
        if done:
            break

    video_writer.release()
    env.close()

create_evaluation_video(agent, env, output_path="agent_performance.mp4")

def create_md_report(output_path="report.md"):
    architecture = """
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
"""
    with open(output_path, "w") as file:
        file.write(f"# Agent Training Report\n\n{architecture}")
    print(f"Report saved to {output_path}.")

create_md_report(output_path="training_report.md")