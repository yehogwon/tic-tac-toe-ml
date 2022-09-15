from __future__ import annotations

from abc import ABC
import copy
import math
import random
import time
from typing import Callable, List, Optional, Set, Tuple
import numpy as np

import torch
import torch.nn.functional as F

from network import PolicyValueNet
from tqdm import tqdm


EPSILON = 1e-10

class BaseState(ABC): 
    def get_current_player(self) -> int:
        raise NotImplementedError()

    def get_possible_actions(self) -> List[int]: 
        raise NotImplementedError()

    def take_action(self, action: int) -> BaseState:
        raise NotImplementedError()

    def is_terminal(self) -> bool: 
        raise NotImplementedError()

    def get_possible_actions(self) -> List[int]: 
        raise NotImplementedError()

    def get_reward(self) -> int: 
        raise NotImplementedError()

class TicTacToeState(BaseState): 
    def __init__(self, turn: int) -> None:
        self.board = np.zeros((3, 3), dtype=np.int_)
        self.turn = turn

    def get_current_player(self): 
        return self.turn
    
    def get_possible_actions(self) -> List[int]: 
        coords = np.argwhere(self.board == 0)
        coords = coords[:, 0] * 3 + coords[:, 1]
        return [item for item in coords.tolist()]

    def take_action(self, action: int) -> TicTacToeState:
        self.board[action // 3][action % 3] = self.turn
        self.turn -= 2 * self.turn
        return self

    def is_terminal(self) -> bool:
        return np.abs(self.board).sum() == 9 or self.get_reward() != 0

    def get_reward(self) -> int:
        sum_list = []
        for i in range(3): 
            sum_list.append(self.board[i, :].sum())
            sum_list.append(self.board[:, i].sum())
        sum_list.append(self.board.diagonal().sum())
        sum_list.append(np.fliplr(self.board).diagonal().sum())
        sum_list = [abs(x) for x in sum_list]
        if 3 in sum_list:
            return -1 * self.turn
        else: 
            return 0
    
    def __repr__(self) -> str:
        _board = self.board.tolist()
        for item in range(3): 
            for i in range(3): 
                if _board[item][i] == 1: 
                    _board[item][i] = 'X'
                elif _board[item][i] == -1: 
                    _board[item][i] = 'O'
                else: 
                    _board[item][i] = ' '
        return '\n'.join([str(item) for item in _board])

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

    def select(self): 
        return max(self.children.items(), key=lambda node: node[1].get_uct())
    
    def update(self, reward: int): 
        self.n += 1
        self.w += reward
    
    def update_recursive(self, reward: int): 
        if self.parent: 
            self.parent.update_recursive(-reward)
        self.update(reward)

    def get_uct(self): 
        self.u = self.w / (self.n + EPSILON) + self.c * self.p_prob * math.sqrt(self.parent.n) / (1 + self.n)
        return self.u
    
    def is_leaf(self): 
        return len(self.children) == 0
    
    def is_root(self): 
        return self.parent is None

class MCTS: 
    def __init__(self, policy_value_fn: Callable, c_puct=math.sqrt(2), n_iteration=100) -> None:
        self.root = Node(None, 1.0, c_puct)
        self.network = policy_value_fn
        self.c_puct = c_puct
        self.n_iteration = n_iteration
    
    def playout(self, state: TicTacToeState): 
        node = self.root
        while not node.is_leaf(): 
            action, node = node.select()
            state.take_action(action)
        action_probs, leaf_value = self.network(torch.tensor(state.board, dtype=torch.float32).view(1, 1, 3, 3))
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

            _action_probs = F.softmax(action_probs, dim=0).data.numpy()
            action_prob_list = []
            for action, prob in enumerate(_action_probs): 
                action_prob_list.append((action, prob))
            node.expand(action_prob_list)
        node.update_recursive(-leaf_value)
    
    def get_move_probs(self, state: TicTacToeState) -> Tuple[List[int], np.typing.NDArray]: 
        for _ in tqdm(range(self.n_iteration)): 
            _state = copy.deepcopy(state)
            self.playout(_state)
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

def mcts_agent(mcts: MCTS, state: TicTacToeState) -> int: 
    acts, probs = mcts.get_move_probs(state)
    action = np.random.choice(acts, p=probs)
    mcts.update_with_move(action)
    return action

def random_agent(state: TicTacToeState) -> int: 
    return random.choice(state.get_possible_actions())

# FIXME: Do only legal actions
if __name__ == '__main__': 
    state = TicTacToeState(-1)
    network = PolicyValueNet()
    mcts = MCTS(lambda x: network(x), n_iteration=200)

    agents = [lambda x: mcts_agent(mcts, x), lambda x: random_agent(x)]

    print('\033[2J')
    print(state)
    while not state.is_terminal():
        agent = agents[(state.turn + 1) // 2]
        action = agent(state)
        state.take_action(action)

        print('\033[2J')
        print(state)
        print(f'agent: {(state.turn + 1) // 2}, action: {action}')

        time.sleep(0.5)
    print(state.get_reward())
