from abc import ABCMeta, abstractmethod
from typing import OrderedDict, Tuple
import numpy as np
import random
import torch

from network import QNet
from utils import flat_state

class BaseAgent(metaclass=ABCMeta): 
    @abstractmethod
    def action(self, state: np.ndarray) -> Tuple[int, int]: 
        """
        Return the action of the agent.
        @param state: The current state of the game.
        @type state: np.ndarray
        @return: The action of the agent.
        @rtype: Tuple[int, int]
        """
        pass

class HumanAgent(BaseAgent): 
    def action(self, state: np.ndarray) -> Tuple[int, int]: 
        print(state)
        print('Enter the cell to place. Write row and column:', end=' ')
        r, c = input().split(' ')
        return (int(r), int(c))

class RandomAgent(BaseAgent): 
    def action(self, state: np.ndarray) -> Tuple[int, int]:
        action_list = [(i // 3, i % 3) for i in range(9) if state[i // 3][i % 3] == 0]
        return random.choice(action_list)

class LearningAgent(BaseAgent): 
    def __init__(self, state_dict: OrderedDict) -> None:
        super().__init__()
        self.q = QNet()
        self.q.load_state_dict(state_dict)
    
    def action(self, state: np.ndarray) -> Tuple[int, int]: 
        q_table = self.q(torch.FloatTensor(flat_state(state)))
        action = torch.argmax(q_table).item()
        return (action // 3, action % 3)
