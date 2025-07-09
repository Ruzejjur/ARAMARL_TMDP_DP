import numpy as np

def softmax(x: np.ndarray, beta: float=1.0) -> np.ndarray:
    """
    Computes the softmax distribution for a given array of scores.

    The softmax function converts a vector of arbitrary real-valued scores into
    a vector of probabilities, where each element is in the range (0, 1) and
    all elements sum to 1. A stability trick (subtracting the max value) is
    used to prevent potential overflow issues with large input values.

    Args:
        x (np.ndarray): A 1D array of numerical scores.
        beta (float, optional): The temperature parameter. A higher beta value
            results in a more deterministic (sharper) probability distribution,
            while a lower value results in a more uniform (softer) distribution.
            Defaults to 1.0.

    Returns:
        np.ndarray: The computed probability distribution.
    """
    # Subtract the maximum value for numerical stability before exponentiating.
    # This prevents overflow errors when dealing with large scores.
    x_stable = x - np.max(x)
    
    # Calculate the exponentiated values, scaled by beta.
    e_x = np.exp(beta * x_stable)
    
    # Normalize by the sum to get the final probability distribution.
    return e_x / np.sum(e_x)

def manhattan_distance(p: np.ndarray, q:np.ndarray):
    """
    Calculates the Manhattan distance (L1 norm) between two points.

    The Manhattan distance is the sum of the absolute differences of their
    Cartesian coordinates. It is the distance a car would drive in a city
    laid out in a grid.

    Args:
        p (np.ndarray): The first coordunates
        q (np.ndarra): The second coordinates

    Returns:
        float: The Manhattan distance between points p and q.

    Raises:
        ValueError: If the input points do not have the same dimension.
    """

    if len(p) != len(q):
        raise ValueError("Points must have the same dimension.")
    
    # Use a generator expression for a memory-efficient sum.
    return sum(abs(a - b) for a, b in zip(p, q))