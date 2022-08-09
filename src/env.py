import numpy as np

class Board(): # Empty: 0, X: 1, O: -1
    def __init__(self, size=3):
        assert size >= 3, "Size must be at least 3"
        assert type(size) is int, "Size must be an integer"
        
        self.size = size
        self.reset()
    
    def reset(self): 
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.status = 0 # 0: draw, 1: X, -1: O
    
    def put(self, x, y, player):
        assert player in [-1, 1], "Player must be -1 or 1"
        self.board[x][y] = player
        
        self.check()
        return self.status != 0
    
    def _check(self, vec): 
        s = sum(vec)
        if abs(s) == self.size:
            self.status = int(s / self.size)

    def check(self): 
        for row in self.board: 
            self._check(row)
        for col in self.board.T: 
            self._check(col)
        self._check(np.diag(self.board))
        self._check(np.diag(np.fliplr(self.board)))
    
    def can_placed(self, x, y): 
        return self.board[x][y] == 0
    
    def all_occupied(self): 
        return np.all(self.board != 0)
    
    def flatten(self): 
        return self.board.flatten()
    
    def show_board(self): 
        print(f'=== Board : {self.status} ===')
        for row in self.board:
            print(' '.join([str(item) for item in row]))
    
    def __call__(self, x, y, player):
        return self.put(x, y, player)