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

class QLearningAgent:
    def __init__(self, action_size, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, batch_size=64, memory_size=10000, lr=0.001):
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = CNN(action_size).to(self.device)
        self.target_network = CNN(action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

    def preprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (84, 84))
        frame = frame / 255.0
        return frame

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)  # Add batch and channel dimensions
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).unsqueeze(1).to(self.device)  # Add channel dimension
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).unsqueeze(1).to(self.device)  # Add channel dimension
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = F.mse_loss(q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, filename):
        torch.save(self.q_network.state_dict(), filename)

    def load(self, filename):
        self.q_network.load_state_dict(torch.load(filename))
        self.target_network.load_state_dict(self.q_network.state_dict())

env = gym.make('FlappyBird-v0')
action_size = env.action_space.n

agent = QLearningAgent(action_size=action_size)

n_episodes = 1000
save_interval = 100  # Save the model every 100 episodes
best_total_reward = -float('inf')
scores = []

for episode in range(n_episodes):
    state, _ = env.reset()
    state = agent.preprocess_frame(state)
    total_reward = 0

    while True:
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = agent.preprocess_frame(next_state)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            break

        agent.replay()

    agent.update_target_network()
    scores.append(total_reward)
    print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

    if episode % save_interval == 0:
        agent.save(f"flappy_bird_model_{episode}.pth")

    if total_reward > best_total_reward:
        best_total_reward = total_reward
        agent.save("flappy_bird_best_model.pth")

plt.plot(scores)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.show()

import torch

def create_evaluation_video(agent, env, output_path="evaluation_video.mp4", max_steps=500):
    # Load the model from the file
    model_path = "flappy_bird_best_model.pth"
    agent.model.load_state_dict(torch.load(model_path))
    agent.model.eval()  # Set the model to evaluation mode

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (env.render(mode='rgb_array').shape[1], env.render(mode='rgb_array').shape[0]))
    
    state, _ = env.reset()
    state = agent.preprocess_frame(state)
    total_reward = 0

    for _ in range(max_steps):
        frame = env.render(mode='rgb_array')
        video_writer.write(frame)
        
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = agent.preprocess_frame(next_state)
        total_reward += reward
        state = next_state
        
        if done:
            break

    video_writer.release()
    env.close()
create_evaluation_video(agent, env, output_path="agent_performance.mp4")

def create_md_report(output_path="report.md"):
    architecture = """
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
"""
    with open(output_path, "w") as file:
        file.write(f"# Agent Training Report\n\n{architecture}")
    print(f"Report saved to {output_path}.")

create_md_report(output_path="training_report.md")