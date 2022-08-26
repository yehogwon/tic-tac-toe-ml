class OutofRangeException(Exception):
	def __init__(self):
		super().__init__('Given input is out of range. It should be between 0 and 2.')

class InvalidActionException(Exception):
	def __init__(self):
		super().__init__('Given cell is already full. You should choose empty cell.')
