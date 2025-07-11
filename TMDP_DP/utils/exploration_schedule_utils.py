import numpy as np

def linear_epsilon_decay(epsilon_begin, epsilon_end, n_iter): 
    """
    Generates a schedule of epsilon values that linearly decay over a specified
    number of iterations.

    This is commonly used in reinforcement learning to decrease the exploration
    rate (epsilon) from a starting value to an ending value over the course of
    training.

    Args:
        epsilon_begin (float): The initial value of epsilon at the start (iteration 0).
        epsilon_end (float): The final value of epsilon at the last iteration.
        n_iter (int): The total number of iterations for the decay. Must be at least 1.

    Returns:
        np.ndarray: A 1D array of size `n_iter` containing the linearly
            decaying epsilon values.

    Raises:
        ValueError: If n_iter is less than 1.

    Example:
        >>> linear_epsilon_decay(epsilon_begin=1.0, epsilon_end=0.1, n_iter=10)
        array([1. , 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    """
    if n_iter < 1:
        raise ValueError("Number of iterations (n_iter) must be at least 1.")
    
    # Handle the edge case where there is only one iteration.
    if n_iter == 1:
        return np.array([epsilon_begin])

    # Calculate the decay amount for each step.
    # The number of intervals is n_iter - 1.
    decay_step = (epsilon_end - epsilon_begin) / (n_iter - 1)
    
    # Generate the schedule.
    epsilon_schedule = np.arange(n_iter) * decay_step + epsilon_begin
    
    return epsilon_schedule