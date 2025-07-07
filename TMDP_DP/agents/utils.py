import numpy as np

def softmax(x, beta=1.0):
    x = x - np.max(x)  # stability
    e_x = np.exp(beta * x)
    return e_x / np.sum(e_x)

def manhattan_distance(p, q):
    if len(p) != len(q):
        raise ValueError("Points must have the same dimension")
    return sum(abs(a - b) for a, b in zip(p, q))