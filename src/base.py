from abc import abstractmethod
from typing import List

import numpy as np

SIZE = 3

class BaseAgent():
    @abstractmethod
    def action(self, state) -> int: 
        raise NotImplementedError('BaseAgent.action() is not implemented')

# TODO: Implement the game environment
class TicTacToe(): 
    def __init__(self, opponent: BaseAgent): 
        self.reset()
        self._agent = opponent

    def reset(self, first: int = 0): # 0 is computer, 1 is opponent (computer: O, opponent X)
        self.status = np.zeros((SIZE, SIZE), dtype=np.int)

    def step(self, action: int) -> List[np.typing.NDArray, int, bool, dict]: # state, reward, done, info (for debug purpose)
        _action = self._agent.action(self.status)
        _done = self.done()
        if _done == -1: 
            self._forward()
            return self.status, 
        else: 
            return self.state, 
        return 

    def _forward(self): 
        _action = self._agent.action(self.status)
        self.status[_action // SIZE, _action % SIZE] = 1
    
    def _check(self, v: np.typing.NDArray) -> bool:
        pass
    
    def done(self) -> int: # check the winner of the game. If it's not done, return -1. 
        # check rows
        for row in self.state: 
            if not self._check(row): 
                return False
        # check columns
        for col in self.state.T: 
            if not self._check(col): 
                return False
        # check diagonals
        for diag in [self.state.diagonal(), self.state.flip().diagonal()]: 
            if not self._check(diag): 
                return False
        return -1

    def render(self): 
        for i in range(SIZE): 
            print(' '.join(['O' if self.status[i, j] == 0 else 'X' for j in range(SIZE)]))
