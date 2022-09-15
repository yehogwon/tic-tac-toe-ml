import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm.auto import tqdm
from collections import deque
from typing import List, Tuple
import random

import numpy as np
import pickle
import datetime

from base import BaseAgent
from mcts import MCTSAgent
from game import TicTacToeState, self_play
from network import PolicyValueNet


class SelfPlayBuffer: 
    def __init__(self, capacity: int) -> None: 
        self._buffer = deque(maxlen=capacity)
        self.episode_len = 0

    # TODO: Implement a data augmentation function
    def augment_data(self, data: List[Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]]): 
        return data
    
    def correct_data(self, agent: BaseAgent, n_game: int): 
        for _ in tqdm(range(n_game), desc='Self-Play'): 
            reward, play_data = self_play(agent)
            self.episode_len += len(play_data)
            self._buffer.extend(play_data)
    
    def sample(self, size: int) -> List: 
        return random.sample(self._buffer, size)
    
    def save(self, path: str) -> None: 
        with open(path, 'wb') as f: 
            pickle.dump(self._buffer, f)

class TrainingPipeline: 
    def __init__(self, capacity: int, n_epoch: int, batch_size: int, optimizer: optim.Optimizer, interval: int, device: str) -> None:
        self.capacity = capacity
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.interval = interval
        self.device = device

        def _criteria(y_hat: Tuple[torch.Tensor, torch.Tensor], target: Tuple[torch.Tensor, torch.Tensor]) -> float: 
            print('shapes: ', y_hat[0].shape, target[0].shape, y_hat[1].shape, target[1].shape)
            policy_loss = F.cross_entropy(y_hat[0], target[0])
            value_loss = F.mse_loss(y_hat[1], target[1])
            return policy_loss + value_loss
        self.criteria = _criteria

    def train(
        self, 
        network: PolicyValueNet, 
        mcts_iteration: int, 
        n_game: int, 
        data_path: str = None, 
        data_only: bool = False) -> None: 

        if data_path: 
            with open(data_path, 'rb') as f: 
                buffer._buffer = pickle.load(f)
        else: 
            buffer = SelfPlayBuffer(self.capacity)
            buffer.correct_data(MCTSAgent(network, mcts_iteration), n_game)
            buffer.save('data/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.pkl')

        if data_only: 
            return

        for i in tqdm(range(1, self.n_epoch + 1), desc='Training'): 
            batch = buffer.sample(self.batch_size)

            state_batch = torch.tensor([data[0] for data in batch], dtype=torch.float32).unsqueeze(dim=1).to(self.device)
            policy_batch = torch.tensor([data[1] for data in batch], dtype=torch.float32).to(self.device)
            value_batch = torch.tensor([data[2] for data in batch], dtype=torch.float32).to(self.device)

            old_probs, old_v = network(state_batch)
            
            loss = self.criteria([old_probs, old_v], [policy_batch, value_batch])
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % self.interval == 0: 
                torch.save(network.state_dict(), f"model/model_{i}.pt")

if __name__ == '__main__': 
    network = PolicyValueNet()
    optimizer = optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-4)
    training = TrainingPipeline(10000, 1000, 32, optimizer, 1, 'cpu')
    training.train(network, 1000, 10000, data_only=True)
