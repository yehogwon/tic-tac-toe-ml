from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

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

class BaseAgent(ABC): 
    @abstractmethod
    def reset_agent(self) -> None:
        raise NotImplementedError()
        
    @abstractmethod
    def get_action(self, state: BaseState) -> Tuple[int, np.ndarray]:
        raise NotImplementedError()