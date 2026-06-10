import numpy as np


def make_windows(X, y=None, window_size=60, stride=10):
    Xs, ys = [], []
    n = len(X)
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        Xs.append(X[start:end].T[..., None])  # [N sensors, W, 1]
        if y is not None:
            ys.append(int(np.max(y[start:end]) > 0))
    Xs = np.asarray(Xs, dtype='float32')
    ys = np.asarray(ys, dtype='int64') if y is not None else None
    return Xs, ys
