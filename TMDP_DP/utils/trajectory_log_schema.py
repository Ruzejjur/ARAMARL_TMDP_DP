TRAJECTORY_LOG_COLUMN_MAP = {
    'experiment_num': 0, 'episode_num': 1, 'step_num': 2,
    'p0_loc_old_row': 3, 'p0_loc_old_col': 4, # Previous location of player 0
    'p1_loc_old_row': 5, 'p1_loc_old_col': 6, # Previous location of player 1
    'p0_loc_new_row': 7, 'p0_loc_new_col': 8, # New location of player 0
    'p1_loc_new_row': 9, 'p1_loc_new_col': 10, # New location of player 1
    'coin1_row': 11, 'coin1_col': 12, # Coin 1 location 
    'coin2_row': 13, 'coin2_col': 14, # Coin 2 location
    'p0_action_move': 15, 'p0_action_push': 16, # Player's 0 move and push actions chosen in previous state
    'p1_action_move': 17, 'p1_action_push': 18, # Player's 1 move and push actions chosen in previous state
    'p0_reward': 19, 'p1_reward': 20, # Player's 0 and Player's 1 respective rewards for reaching state new sate
    'p0_cum_reward': 21, 'p1_cum_reward': 22 # Player's 0 and Player's 1 respective cumulative rewards for the whole episode
}