from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tqdm.auto import tqdm

from collections import deque
import random

from utils import flat_state
from config import device

BUFFER_LIMIT = 50000
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
K = 10
EPSILON = 0.02
INTERVAL = 1000
MIN_MEMORY_STACK = 5000

gamma = 1
n_episodes = 10000

class ReplayBuffer(): 
    def __init__(self) -> None:
        self._buffer = deque(maxlen=BUFFER_LIMIT)
    
    def put(self, transition: Tuple[np.ndarray, int, int, np.ndarray, float]) -> None: 
        self._buffer.append(transition)
    
    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
        # states, actions, rewards, next_states, done_masks
        samples: List[Tuple[np.ndarray, int, int, np.ndarray, float]] = random.sample(self._buffer, n)
        s_list, a_list, r_list, s_prime_list, done_mask_list = tuple(zip(*samples))
        
        return (
            torch.tensor(np.array(s_list), dtype=torch.float32, device=device), 
            torch.tensor(np.array(a_list), device=device).unsqueeze(dim=1), 
            torch.tensor(np.array(r_list), device=device).unsqueeze(dim=1), 
            torch.tensor(np.array(s_prime_list), dtype=torch.float32, device=device), 
            torch.tensor(np.array(done_mask_list), dtype=torch.float32, device=device).unsqueeze(dim=1)
        )
    
    def __len__(self):
        return len(self._buffer)

# Add softmax to the output layer
class QNet(nn.Module): 
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(18, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 9)

        self.relu = nn.ReLU()
    
    def forward(self, x): 
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def sample_action(q: QNet, state: np.ndarray) -> int: 
    if random.random() < EPSILON:
        return random.randint(0, 8)
    tensor = torch.FloatTensor(flat_state(state)).to(device)
    q_table = q(tensor)
    return torch.argmax(q_table).item()

def train(q: QNet, q_target: QNet, memory: ReplayBuffer, optimizer: optim.Optimizer) -> float: 
    loss_sum = 0.0
    criteria = F.smooth_l1_loss
    for _ in range(K): 
        # TODO: What about using cross entropy loss?
        s, a, r, s_prime, done_mask = memory.sample(BATCH_SIZE)
        q_out: torch.Tensor = q(s)
        q_a = q_out.gather(dim=1, index=a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target: torch.Tensor = r + gamma * max_q_prime * done_mask
        loss = criteria(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
    return loss_sum / K

def train_loop(): 
    # Import here to avoid circular import
    from game import TicTacToe
    from agent import RandomAgent

    env = TicTacToe()
    q = QNet().to(device)
    q_target = QNet().to(device)
    q_target.load_state_dict(q.state_dict())
    
    memory = ReplayBuffer()

    optimizer = optim.Adam(q.parameters(), lr=LEARNING_RATE)
    loss = 0.0
    
    win, draw, lose = 0, 0, 0
    for epi in tqdm(range(1, n_episodes + 1), desc=f'Loss : {loss} : {win}/{draw}/{lose}'): 
        state = env.reset(opponent=RandomAgent(), turn=bool(random.getrandbits(1)))
        done = False

        while not done: 
            action = sample_action(q, state)
            next_state, reward, done, _ = env.step((action // 3, action % 3))
            done_mask = float(0 if done else 1)
            memory.put((flat_state(state), action, reward, flat_state(next_state), done_mask))
            state = next_state
        
        win += reward == 1
        draw += reward == 0
        lose += reward == -1

        if len(memory) > MIN_MEMORY_STACK: 
            loss = train(q, q_target, memory, optimizer)
            if epi % INTERVAL == 0: 
                q_target.load_state_dict(q.state_dict())
                win, draw, lose = 0, 0, 0
                torch.save(q.state_dict(), f'bin/model_ckpt{epi}.pt')

if __name__ == '__main__': 
    train_loop()
