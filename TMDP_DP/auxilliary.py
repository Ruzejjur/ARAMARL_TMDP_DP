import numpy as np

def linear_epsilon_decay(epsilon_begin, epsilon_end, n_iter): 
    epsilon_begin = 1 
    epsilon_end = 0.1

    epsilon_arr =  np.arange(n_iter)*((epsilon_end-epsilon_begin)/(n_iter-1)) + epsilon_begin
    
    return epsilon_arr