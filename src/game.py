from typing import List, Tuple
import numpy as np
from exception import *
from agent import BaseAgent

class TicTacToe: 
    def reset(self, opponent: BaseAgent, turn=True) -> np.ndarray: 
        self._opponent = opponent
        self._board = np.zeros((3, 3), dtype=int)
        if not turn:
            agent_action = self._opponent.action(self._board)
            self._board[agent_action[0]][agent_action[1]] = -1
        return self._board.copy()
    
    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, int, bool, dict]:
        '''
        Walks through the game with given action, returns the next state, reward, done, and info.
        @param Tuple[int, int] action: action to take
        @return: next state, reward, done, and info (for debugging purposes)
        @rtype: Tuple[np.ndarray, int, bool, dict]
        '''
        # First, place on the board for my turn
        # Assertive (the case where the action is out of range)
        if not (0 <= action[0] <= 2 and 0 <= action[1] <= 2): 
            print('Action is out of action space.')
            return self._board.copy(), -1, True, {}
        
        if not self._possible(action): 
            return self._board.copy(), -1, True, {}
        self._board[action[0]][action[1]] = 1
        if self._win(): 
            return self._board.copy(), 1, True, {}
        if self._all_occupied():
            return self._board.copy(), 0, True, {}
        
        # It's the opponent's turn
        opponent_action = self._opponent.action(-self._board)
        if not self._possible(opponent_action): 
            return self._board.copy(), -1, True, {}
        self._board[opponent_action[0]][opponent_action[1]] = -1
        if self._win():
            return self._board.copy(), -1, True, {}
        if self._all_occupied():
            return self._board.copy(), 0, True, {}
        
        return self._board.copy(), 0, False, {}
    
    def _possible(self, action: Tuple[int, int]) -> bool: 
        return self._board[action[0]][action[1]] == 0
    
    def _check(self, v: List[int]) -> int:
        _sum = sum(v)
        if _sum == 3: 
            return 1
        elif _sum == -3:
            return -1
        else:
            return 0
    
    def _win(self) -> int: # check the winner of the game. If it's not done, return -1. 
        vecs: list = [self._board.tolist()[0], self._board.T.tolist()[0], self._board.diagonal(), np.flip(self._board, 1).diagonal()]
        
        for vec in vecs: 
            checked = self._check(vec)
            if checked != 0:
                return checked
        return 0

    def _all_occupied(self) -> bool: 
        return np.square(self._board).sum() == 9
    
    def render(self) -> None: 
        print("+---+---+---+")
        for row in range(3):
            print("|", end = '')
            for col in range(3):
                shape = ' '
                if self._board[row][col] == 1:
                    shape = 'O'
                if self._board[row][col] == -1:
                    shape = 'X'
                print(" {} |".format(shape), end = '')
            print("\n+---+---+---+")
