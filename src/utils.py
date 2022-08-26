import numpy as np

def flat_state(state: np.ndarray) -> np.ndarray:
    _state = state.flatten()
    return np.concatenate([_state == 1, _state == -1])
