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
import argparse

import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Pool

from base import BaseAgent
from mcts import MCTSAgent
from game import TicTacToeState, self_play
from network import PolicyValueNet
from config import device

try: 
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass

def time_stamp(): 
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def print_log(log: str): 
    print(f'{time_stamp()} : {log}')

class SelfPlayBuffer: 
    def __init__(self, capacity: int=10000) -> None: 
        self._buffer: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=capacity)
        
        self.episode_len = 0

    def augment_data(self, data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]: 
        _data = data.copy()
        for state, prob, z in data: 
            _data.append((np.fliplr(state), prob, z))
            _data.append((np.flipud(state), prob, z))
        return _data

    def _correct_data(self, idx: int, agent: BaseAgent, n_game: int, p_bar: bool) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]: 
        data_list = []
        it = tqdm(range(n_game), desc=f'Self-Play {idx}', position=idx) if p_bar else range(n_game)
        for _ in it: 
            reward, play_data = self_play(agent)
            self.episode_len += len(play_data)
            data_list.extend(play_data)
        return data_list
    
    def correct_data(self, agent: BaseAgent, n_game: int, p_bar: bool): 
        n_thread = multiprocessing.cpu_count() - 1
        counts = [n_game // n_thread] * n_thread
        if n_game % 4 != 0: 
            n_thread += 1
            counts.append(n_game % n_thread)
        with Pool() as p:
            data_list = p.starmap(self._correct_data, [list(item) for item in zip([i for i in range(n_thread)], [agent for _ in range(n_thread)], counts, [p_bar] * n_thread)])
        for data in data_list:
            self._buffer.extend(self.augment_data(data))
            self.episode_len += len(data)
    
    def sample(self, size: int) -> List: 
        return random.sample(self._buffer, size)

    def __len__(self) -> int:
        return len(self._buffer)
    
    def load(self, path: str) -> None: 
        with open(path, 'rb') as f: 
            self._buffer = pickle.load(f)
    
    def save(self, path: str) -> None: 
        with open(path, 'wb') as f: 
            pickle.dump(self._buffer, f)

class TrainingPipeline: 
    def __init__(self, n_epoch: int, batch_size: int, lr: float, interval: int) -> None:
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.interval = interval

        def _criteria(y_hat: Tuple[torch.Tensor, torch.Tensor], target: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor: 
            policy_loss = F.cross_entropy(y_hat[0], target[0])
            value_loss = F.mse_loss(y_hat[1], target[1])
            return policy_loss + value_loss
        self.criteria = _criteria

    def train(self, network: PolicyValueNet, buffer: SelfPlayBuffer, model_path: str) -> None: 
        optimizer = optim.Adam(network.parameters(), lr=self.lr)
        for i in range(1, self.n_epoch + 1): 
            batch = buffer.sample(self.batch_size)

            state_batch = torch.tensor(np.array([data[0] for data in batch]), dtype=torch.float32).unsqueeze(dim=1).to(device)
            policy_batch = torch.tensor(np.array([data[1] for data in batch]), dtype=torch.float32).to(device)
            value_batch = torch.tensor(np.array([data[2] for data in batch]), dtype=torch.float32).unsqueeze(dim=1).to(device)

            old_probs, old_v = network(state_batch)
            
            loss = self.criteria([old_probs, old_v], [policy_batch, value_batch])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print_log(f'{i}/{self.n_epoch} : {loss.item():.4f}')
            if i % self.interval == 0: 
                torch.save(network.state_dict(), model_path + f"/{time_stamp()}.pt")
                print_log(f'Model Saved : {model_path}/{time_stamp()}.pt')
        print_log(f'Training Finished')
        
    def self_train(self, network: PolicyValueNet, mcts: int, game: int, n_game: int, model_path: str) -> None: 
        optimizer = optim.Adam(network.parameters(), lr=self.lr)
        for game_count in range(1, n_game + 1): 
            agent = MCTSAgent(network, mcts)
            buffer = SelfPlayBuffer()
            for i in range(1, self.n_epoch + 1): 
                buffer.correct_data(agent, game, False)
                batch = buffer.sample(self.batch_size)

                state_batch = torch.tensor(np.array([data[0] for data in batch]), dtype=torch.float32).unsqueeze(dim=1).to(device)
                policy_batch = torch.tensor(np.array([data[1] for data in batch]), dtype=torch.float32).to(device)
                value_batch = torch.tensor(np.array([data[2] for data in batch]), dtype=torch.float32).unsqueeze(dim=1).to(device)

                old_probs, old_v = network(state_batch)
                
                loss = self.criteria([old_probs, old_v], [policy_batch, value_batch])
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print_log(f'{game_count}/{n_game} : {i}/{self.n_epoch} : {loss.item():.4f}')
                if i % self.interval == 0: 
                    torch.save(network.state_dict(), model_path + f'/{time_stamp()}.pt')
                    print_log(f'Model Saved : {model_path}/{time_stamp()}.pt')
            print_log(f'Training Finished : {game_count}/{n_game}')
        print_log('Training Finished')

if __name__ == '__main__': 
    print('device:', device)
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', type=str, help='train or data', default='train')

    parser.add_argument('--capacity', type=int, default=10000)
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--game_count', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--mcts', type=int, default=100)
    parser.add_argument('--game', type=int, default=100)
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--data', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--data_save', type=str)
    parser.add_argument('--model_save', type=str)

    args = parser.parse_args()

    network = PolicyValueNet().to(device)
    if args.model:
        network.load_state_dict(torch.load(args.model))

    if args.mode == 'train':
        # example
        # python src/train.py train --epoch 100 --batch_size 32 --lr 1e-3 --interval 10 --data data/ --model_save model
        if not args.data: 
            raise ValueError('Please specify the data path')
        buffer = SelfPlayBuffer(args.capacity)
        buffer.load(args.data)
        print('Dataset size:', len(buffer))
        optimizer = optim.Adam(network.parameters(), lr=args.lr)
        pipeline = TrainingPipeline(args.epoch, args.batch_size, optimizer, args.interval)
        pipeline.train(network, buffer, args.model_save)
    elif args.mode == 'data':
        # example
        # python src/train.py data --capacity 10000 --mcts 100 --game 10000 --data_save data
        buffer = SelfPlayBuffer(args.capacity)
        buffer.correct_data(MCTSAgent(network, args.mcts), args.game, True)
        buffer.save(args.data_save + '/' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.pkl')
    elif args.mode == 'auto': 
        # example
        # python src/train.py auto --epoch 100 --batch_size 32 --mcts 50 --game 20 --game_count 10 --lr 1e-3 --interval 50 --model_save model
        pipeline = TrainingPipeline(args.epoch, args.batch_size, args.lr, args.interval)
        pipeline.self_train(network, args.mcts, args.game, args.game_count, args.model_save)
    else: 
        raise ValueError('Invalid mode')
