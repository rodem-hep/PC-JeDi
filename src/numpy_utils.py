import numpy as np


def undo_log_squash(data: np.ndarray) -> np.ndarray:
    """Undo the log squash function above."""
    return np.sign(data) * (np.exp(np.abs(data)) - 1)


def log_squash(data: np.ndarray) -> np.ndarray:
    """Apply a log squashing function for distributions with high tails."""
    return np.sign(data) * np.log(np.abs(data) + 1)
