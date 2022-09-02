import torch
import numpy as np

from game import TicTacToe
from agent import LearningAgent
from utils import flat_state
from config import device

MODEL_CKPT = 'bin/model_ckpt10000.pt'

env = TicTacToe()
state = env.reset(opponent=LearningAgent(state_dict=torch.load(MODEL_CKPT), device=device), turn=False)
done = False

while not done:
	env.render()
	r, c = input('Input your action (row, column): ').split(' ')
	r, c = int(r), int(c)
	state, reward, done, _ = env.step((r, c))

env.render()
print('You have got a reward of:', reward)