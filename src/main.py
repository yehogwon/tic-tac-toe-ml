import env
import random
from tqdm import tqdm

SIZE = 3

scoreboard = {-1: 0, 0: 0, 1: 0}
if __name__ == '__main__':
    board = env.Board(SIZE)
    for _ in tqdm(range(10000)): 
        board.reset()
        idx = 0
        while not board.all_occupied(): 
            x = random.randint(0, SIZE - 1)
            y = random.randint(0, SIZE - 1)
            if not board.can_placed(x, y):
                continue
            
            idx += 1
            if board(x, y, idx % 2 + 1): 
                break
        # board.show_board()
        winner = board.status
        scoreboard[winner] += 1

print(scoreboard)