
import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

# Add the parent directory to sys.path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.grid_world import GridWorld

# Hyperparameters
GAMMA = 0.9
LEARNING_RATE = 0.001
BATCH_SIZE = 128
BUFFER_SIZE = 10000
TARGET_UPDATE_INTERVAL = 200 
NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 200

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class QNetwork(nn.Module):
    # Modified Network Architecture:
    # 3 Inputs (Normalized Row, Normalized Col, Normalized Action) -> 1 Output (Q-Value)
    def __init__(self, input_dim=3, output_dim=1):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # We store raw state/action indices/coords and normalize during sampling to keep buffer general
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def normalize_state(state_xy, env_size):
    """
    Input 1: Normalized Row Index (y) [0, 1]
    Input 2: Normalized Column Index (x) [0, 1]
    Note: GridWorld size tuple is (Width, Height) = (Cols, Rows)
    state_xy is (x, y) = (col, row)
    """
    width, height = env_size
    x, y = state_xy
    # x is col, y is row
    # normalize x to [0, 1], y to [0, 1]
    norm_x = x / (width - 1) if width > 1 else 0.0
    norm_y = y / (height - 1) if height > 1 else 0.0
    
    # Requirement: 
    # "前两个输入是状态对应的归一化后的行和列的索引" -> Row first, then Col
    # So we return [norm_y, norm_x]
    return np.array([norm_y, norm_x], dtype=np.float32)

def normalize_action(action_idx, num_actions):
    """
    Input 3: Normalized Action Index [0, 1]
    """
    return action_idx / (num_actions - 1) if num_actions > 1 else 0.0

def train_dqn():
    env = GridWorld()
    env_size = env.env_size # (width, height)
    num_actions = len(env.action_space)

    # Initialize networks
    # Input dim is 3: Norm Row, Norm Col, Norm Act
    main_net = QNetwork(input_dim=3, output_dim=1).to(device)
    target_net = QNetwork(input_dim=3, output_dim=1).to(device)
    target_net.load_state_dict(main_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(main_net.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    iteration_count = 0
    
    print("Starting Training (Optimized Network: 3 inputs -> 1 output)...")

    for episode in range(NUM_EPISODES):
        state_xy, _ = env.reset()
        
        total_loss = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            # Behavior Policy: Random Uniform
            action_idx = random.randint(0, num_actions - 1)
            action = env.action_space[action_idx]

            # Execute action
            next_state_xy, reward, done, _ = env.step(action)

            # Store in replay buffer
            replay_buffer.push(state_xy, action_idx, reward, next_state_xy, done)
            
            state_xy = next_state_xy

            # Train only if we have enough samples
            if len(replay_buffer) > BATCH_SIZE:
                transitions = replay_buffer.sample(BATCH_SIZE)
                # Unzip transitions
                # state: (x,y), action: int(idx), reward: float, next_state: (x,y), done: bool
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

                # Prepare Inputs for Main Network (Current State, Action)
                # Normalize States and Actions
                norm_states = np.array([normalize_state(s, env_size) for s in batch_state]) # shape (B, 2)
                norm_actions = np.array([normalize_action(a, num_actions) for a in batch_action], dtype=np.float32).reshape(-1, 1) # shape (B, 1)
                
                # Concatenate to form (B, 3) input
                network_input = np.hstack([norm_states, norm_actions])
                
                b_input = torch.FloatTensor(network_input).to(device) # (B, 3)
                b_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device) # (B, 1)
                b_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device) # (B, 1)

                # Compute Q(s, a) using Main Network
                current_q_values = main_net(b_input) # Output is (B, 1)

                # Compute Target: max_a' Q_target(s', a')
                # We need to construct inputs for ALL actions for each next state
                
                # Normalize next states
                norm_next_states = np.array([normalize_state(s, env_size) for s in batch_next_state]) # shape (B, 2)
                
                # We need to broadcast next states and iterate over all possible actions
                # Reshape for broadcasting: (B, 1, 2)
                norm_next_states_expanded = torch.FloatTensor(norm_next_states).unsqueeze(1).to(device) # (B, 1, 2)
                norm_next_states_tiled = norm_next_states_expanded.repeat(1, num_actions, 1) # (B, N, 2)
                
                # All actions normalized: (N, 1)
                all_actions_indices = np.arange(num_actions)
                norm_all_actions = np.array([normalize_action(a, num_actions) for a in all_actions_indices], dtype=np.float32).reshape(1, num_actions, 1)
                norm_all_actions_tensor = torch.FloatTensor(norm_all_actions).to(device) # (1, N, 1)
                norm_all_actions_tiled = norm_all_actions_tensor.repeat(BATCH_SIZE, 1, 1) # (B, N, 1)
                
                # Concatenate along last dim: (B, N, 3)
                next_network_input = torch.cat([norm_next_states_tiled, norm_all_actions_tiled], dim=2)
                
                # Flatten to feed into network: (B * N, 3)
                next_network_input_flat = next_network_input.view(-1, 3)
                
                with torch.no_grad():
                    # Get Q values for all (s', a') pairs
                    all_next_q_values_flat = target_net(next_network_input_flat) # (B*N, 1)
                    
                    # Reshape to (B, N) to take max over actions
                    all_next_q_values = all_next_q_values_flat.view(BATCH_SIZE, num_actions)
                    
                    # Max over actions
                    max_next_q = all_next_q_values.max(1)[0].unsqueeze(1) # (B, 1)
                    
                    # Target: r + gamma * max Q * (1 - done)
                    target_q_values = b_reward + GAMMA * max_next_q * (1 - b_done)

                loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

                iteration_count += 1
                if iteration_count % TARGET_UPDATE_INTERVAL == 0:
                    target_net.load_state_dict(main_net.state_dict())
            
            if done:
                break

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}/{NUM_EPISODES}, Loss: {total_loss:.4f}")

    print("Training finished.")
    
    # Visualization: Extract Policy and Values
    visualize_result(env, main_net)

def visualize_result(env, model):
    model.eval()
    env_size = env.env_size
    num_actions = len(env.action_space)
    
    policy_matrix = np.zeros((env.num_states, num_actions))
    value_vector = np.zeros(env.num_states)
    
    with torch.no_grad():
        for s_idx in range(env.num_states):
            state_xy = env.state_index_to_xy(s_idx)
            
            # Prepare inputs for all actions for this single state
            norm_state = normalize_state(state_xy, env_size) # (2,)
            
            # Repeat state N times
            repeated_state = np.tile(norm_state, (num_actions, 1)) # (N, 2)
            
            # All actions normalized
            all_actions_indices = np.arange(num_actions)
            norm_all_actions = np.array([normalize_action(a, num_actions) for a in all_actions_indices], dtype=np.float32).reshape(-1, 1)
            
            # Concatenate (N, 3)
            input_tensor_np = np.hstack([repeated_state, norm_all_actions])
            state_action_tensor = torch.FloatTensor(input_tensor_np).to(device)
            
            # Forward pass: Get Q(s, a) for all a
            q_values = model(state_action_tensor).cpu().numpy().flatten() # (N,)
            
            # Policy: Greedy w.r.t Q
            best_action = np.argmax(q_values)
            policy_matrix[s_idx][best_action] = 1.0 # Deterministic policy
            
            # Value: Max Q
            value_vector[s_idx] = np.max(q_values)

    # Use environment's visualization
    print("Visualizing Policy and Values...")
    # Initialize the plot to ensure self.ax exists
    env.render()
    
    env.add_policy(policy_matrix)
    env.add_state_values(value_vector)
    env.render()
    try:
        input("Press Enter to close...")
    except EOFError:
        pass

if __name__ == "__main__":
    train_dqn()
