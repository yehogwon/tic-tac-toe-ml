from __future__ import annotations

from abc import ABC
import copy
import math
import random
import time
from typing import Callable, List, NamedTuple, Optional, Set, Tuple
import numpy as np

import torch
import torch.nn.functional as F

import game
from game import TicTacToeState
from network import PolicyValueNet
from base import BaseAgent
from tqdm import tqdm

from config import device
import argparse


EPSILON = 1e-10

class Node: 
    def __init__(self, parent: Node, p_prob: float, c: float = math.sqrt(2)) -> None: 
        self.parent = parent
        self.p_prob = p_prob
        self.c = c

        self.children = {}
        self.n = 0
        self.w = 0
        self.u = 0
    
    def expand(self, action_prob: List[Tuple[int, float]]): 
        for action, prob in action_prob: 
            if action not in self.children: 
                self.children[action] = Node(self, prob)

    def select(self, explore: bool): 
        return max(self.children.items(), key=lambda node: node[1].get_uct(explore))
    
    def update(self, reward: int): 
        self.n += 1
        self.w += max(0, reward)
    
    def backpropagate(self, reward: int): 
        if self.parent: 
            self.parent.backpropagate(-reward)
        self.update(reward)

    def get_uct(self, explore: bool): 
        if self.n == 0: 
            self.u = float('inf')
        else:
            if explore: 
                self.u = self.w / (self.n) + self.c * self.p_prob * math.sqrt(self.parent.n) / (1 + self.n)
            else: 
                self.u = self.w / (self.n)
        return self.u
    
    def is_leaf(self): 
        return len(self.children) == 0
    
    def is_root(self): 
        return self.parent is None

class MCTS: 
    def __init__(self, policy_value_fn: Callable, c_puct=math.sqrt(2), n_iteration=10) -> None:
        self.root = Node(None, 1.0, c_puct)
        self.network = policy_value_fn
        self.c_puct = c_puct
        self.n_iteration = n_iteration
    
    def playout(self, state: TicTacToeState, explore: bool): 
        node = self.root
        while not node.is_leaf(): 
            action, node = node.select(explore)
            state.take_action(action)
        action_probs, leaf_value = self.network(torch.tensor(state.board, dtype=torch.float32).view(1, 1, 3, 3).to(device))
        if state.is_terminal(): 
            winner = state.get_reward()
            if winner == state.turn: 
                leaf_value = 1
            else: 
                leaf_value = 0
        else: 
            action_probs = action_probs.flatten()
            illegal_actions: List[int] = [x for x in list(range(9)) if x not in state.get_possible_actions()]
            for illegal_action in illegal_actions:
                action_probs[illegal_action] = float('-inf')

            _action_probs = F.softmax(action_probs, dim=0).cpu().data.numpy()
            action_prob_list = []
            for action, prob in enumerate(_action_probs): 
                action_prob_list.append((action, prob))
            node.expand(action_prob_list)
        node.backpropagate(-leaf_value)
    
    def get_move_probs(self, state: TicTacToeState, explore: bool = True) -> Tuple[List[int], np.typing.NDArray]: 
        for _ in range(self.n_iteration): 
            _state = copy.deepcopy(state)
            self.playout(_state, explore=explore)
        act_visits = [(act, node.n) for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        visits = list(visits)
        illegal_actions = [x for x in list(range(9)) if x not in state.get_possible_actions()]
        for illegal_action in illegal_actions:
            visits[illegal_action] = float('-inf')
        act_probs = F.softmax(torch.tensor(visits, dtype=torch.float32), dim=0).data.numpy()
        return acts, act_probs

    def update_with_move(self, last_move): 
        if last_move in self.root.children: 
            self.root = self.root.children[last_move]
            self.root.parent = None
        else: 
            self.root = Node(None, 1.0, self.c_puct)

class MCTSAgent(BaseAgent): 
    def __init__(self, policy_value_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]], n_iteration: int) -> None:
        self.network = policy_value_fn
        self.mcts = MCTS(policy_value_fn, n_iteration=n_iteration)

    def reset_agent(self):
        self.mcts.update_with_move(-1)
    
    def get_action(self, state: TicTacToeState) -> Tuple[int, np.ndarray]: 
        acts, probs = self.mcts.get_move_probs(state, explore=False)
        action = np.random.choice(acts, p=probs)
        self.mcts.update_with_move(action)
        return action, probs
    
    def __call__(self, state: TicTacToeState) -> int:
        return self.get_action(state)[0]

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='name of the file containing the model weights')
    parser.add_argument('--mcts', type=int, help='the number of iterations to search for each move')
    args = parser.parse_args()

    net = PolicyValueNet()
    net.load_state_dict(torch.load(f'model/{args.model}', map_location='cpu'))
    game.play([MCTSAgent(net, args.mcts), game.ManualAgent()])
