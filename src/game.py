from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
import random

from base import BaseState, BaseAgent


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

def self_play(agent: BaseAgent) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]: 
    state = TicTacToeState(random.choice([1, -1]))
    states, probs, current_players = [], [], []
    while not state.is_terminal():
        action, action_probs = agent.get_action(state)
        states.append(state.board.copy())
        probs.append(action_probs)
        current_players.append(state.get_current_player())
        state = state.take_action(action)
    reward = state.get_reward()
    z = np.zeros(len(current_players), dtype=np.float32)
    if reward != 0: 
        z[np.array(current_players) == reward] = [1.0]
        z[np.array(current_players) != reward] = [-1.0]
    agent.reset_agent()
    return reward, list(zip(states, probs, z))
