"""
This module implements several environments, i.e., the simulators in which agents will interact and learn.
Any environment is characterized by the following two methods:
 * step : receives the actions taken by the agents, and returns the new state of the simulator and the rewards
          perceived by each agent, amongst other things.
 * reset : sets the simulator at the initial state.
"""

import numpy as np


class RMG():
    """
    A two-agent environment for a repeated matrix (symmetric) game.
    Possible actions for each agent are (C)ooperate (0) and (D)efect (1).
    The state is s_t = (a_{t-1}, b_{t-1}) with a_{t-1} and b_{t-1} the actions of the two players in the last turn,
    plus an initial state s_0.
    """
    # Possible actions
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = NUM_AGENTS*NUM_ACTIONS + 1   # we add the initial state.

    def __init__(self, max_steps, payouts, batch_size=1):
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.payout_mat = payouts
        self.available_actions = [
            np.ones((batch_size, self.NUM_ACTIONS), dtype=int)
            for _ in range(self.NUM_AGENTS)
        ]

        self.step_count = None

    def reset(self):
        self.step_count = 0
        init_state = np.zeros((self.batch_size, self.NUM_STATES))
        init_state[:, -1] = 1
        observations = [init_state, init_state]
        info = [{'available_actions': aa} for aa in self.available_actions]
        return observations, info

    def step(self, action):
        ac0, ac1 = action

        self.step_count += 1

        rewards = []

        # The state is a OHE vector indicating [CC, CD, DC, DD, initial], (iff NUM_STATES = 5)
        state0 = np.zeros((self.batch_size, self.NUM_STATES))
        state1 = np.zeros((self.batch_size, self.NUM_STATES))
        for i, (a0, a1) in enumerate(zip(ac0, ac1)):  # iterates over batch dimension
            rewards.append([self.payout_mat[a1][a0], self.payout_mat[a0][a1]])
            state0[i, a0 * 2 + a1] = 1
            state1[i, a1 * 2 + a0] = 1
        rewards = list(map(np.asarray, zip(*rewards)))
        observations = [state0, state1]

        done = (self.step_count == self.max_steps)
        info = [{'available_actions': aa} for aa in self.available_actions]

        return observations, rewards, done, info


class AdvRw():
    """
    A two-action stateless environment in which an adversary controls the reward
    """

    def __init__(self, mode='friend', p=0.5):
        self._mode = mode

        self._p = p  # probability for the neutral environment

        if self._mode == 'friend':
            self.reward_table_DM = np.array([[+50,-50],[-50,+50]])
            self.reward_table_ADV = np.array([[+50,-50],[-50,+50]])
        elif self._mode == 'adversary':
            self.reward_table_DM = np.array([[+50,-50],[-50,+50]])
            self.reward_table_ADV = np.array([[-50,+50],[+50,-50]])
        else: 
            raise ValueError('Invalid environment mode.')

        # TODO: Redefine the rewards for nutral case using a table
        # elif self._mode == 'neutral':
        #     box = np.random.rand() < self._p
        #     if int(box) == action_agent:
        #         reward = +50
        #     else:
        #         reward = -50 
    
    def get_reward_table_DM(self):
        return self.reward_table_DM
    
    def get_reward_table_ADV(self):
        return self.reward_table_ADV
    
    def reset(self):
        # self._policy = np.asarray([0.5, 0.5])
        return

    def step(self, action_agent, action_adversary):
        
        reward_DM = self.reward_table_DM[action_agent, action_adversary]
        reward_ADV = self.reward_table_ADV[action_agent, action_adversary]
        
        return None, (reward_DM, reward_ADV), True, None
   

class AdvRw2():
    """
    Friend or Foe modified to model adversary separately.
    """

    def __init__(self, max_steps, payout=50, batch_size=1):
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.payout = payout
        self.available_actions = np.array([0, 1])
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return

    def step(self, action):
        ac0, ac1 = action

        self.step_count += 1

        dm_reward = self.payout if ac0 == ac1 else -self.payout

        rewards = [dm_reward, -dm_reward]  # Assuming zero-sum...
        observations = None

        done = (self.step_count == self.max_steps)

        return observations, rewards, done
#


class AdvRwGridworld():
    """
    Friend or Foe modified to model adversary separately, with gridworld
    """

    def __init__(self, max_steps, batch_size=1):
        self.H = 4
        self.W = 3
        self.world = np.array([self.H, self.W])  # The gridworld

        self.targets = np.array([[0, 0], [0, 2]])  # Position of the targets
        self.DM = np.array([3, 1])  # Initial position of the DM

        self.max_steps = max_steps
        self.batch_size = batch_size
        self.available_actions_DM = np.array(
            [0, 1, 2, 3])  # Up, right, down, left
        self.available_actions_Adv = np.array([0, 1])  # Select target 1 or 2.
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        self.DM = np.array([3, 1])
        return

    def _coord2int(self, pos):
        return pos[0] + self.H*pos[1]

    def step(self, action):
        ac_DM, ac_Adv = action

        self.step_count += 1

        if ac_DM == 0:  # Up
            self.DM[0] = np.maximum(0, self.DM[0] - 1)
        elif ac_DM == 1:  # Right
            self.DM[1] = np.minimum(self.W - 1, self.DM[1] + 1)
        elif ac_DM == 2:  # Down
            self.DM[0] = np.minimum(self.H - 1, self.DM[0] + 1)
        elif ac_DM == 3:  # Left
            self.DM[1] = np.maximum(0, self.DM[1] - 1)

        done = False
        dm_reward = -1  # One step more
        adv_reward = 0

        # Check if DM is @ targets, then finish

        if np.all(self.DM == self.targets[0, :]):
            if ac_Adv == 0:
                dm_reward += 50
                adv_reward -= 50
            else:
                dm_reward -= 50
                adv_reward += 50
            done = True

        if np.all(self.DM == self.targets[1, :]):
            if ac_Adv == 1:
                dm_reward += 50
                adv_reward -= 50
            else:
                dm_reward -= 50
                adv_reward += 50
            done = True

        # Check if step limit, then finish

        if self.step_count == self.max_steps:
            done = True

        #dm_reward = self.payout if ac0 == ac1 else -self.payout

        # rewards = [dm_reward, -dm_reward] #Assuming zero-sum...
        #observations = None

        #done = (self.step_count == self.max_steps)

        return self._coord2int(self.DM), (dm_reward, adv_reward), done


class Blotto():
    """
    Blotto game with multiple adversaries
    """

    def __init__(self, max_steps, payout=50, batch_size=1, deterministic=True):
        self.max_steps = max_steps
        self.batch_size = batch_size
        #self.payout = payout
        self.available_actions = np.array([0, 1])
        self.step_count = 0
        self.deterministic = deterministic

    def reset(self):
        self.step_count = 0
        return

    def step(self, actions):
        """ action[0] is that of the defender """
        self.step_count += 1

        num_attackers = len(actions) - 1

        actions = np.asarray(actions)

        att_rew = np.sum(actions[1:, ], axis=0)
        tmp = actions[0, ] - att_rew

        draw_pos = tmp == 0
        if self.deterministic != True:
            tmp[tmp == 0] = np.random.choice(
                [-1, 1], size=len(tmp[tmp == 0]))*(actions[0, draw_pos] > 0)


        ind = np.sum(actions, axis=0) > 0 ## to see in which position there was at least one resource

        tmp = tmp*ind

        tmp[tmp < 0] = -1 # Defender looses corresponding position
        tmp[tmp > 0] = 1  # Defender wins corresponding position

        # print('tmp', tmp)

        reward_dm = tmp.sum()

        tmp2 = actions[1:, ] - actions[0, ]
        tmp2[tmp2 > 0] = 1
        tmp2[tmp2 < 0] = -1

        # print('tmp2', tmp2)

        # s = np.sum(actions[1:, draw_pos], axis=0)
        z = draw_pos & actions[1:, ]

        z_new = z/z.sum(axis=0)
        z_new = np.nan_to_num(z_new)
        z_new = z_new*ind

        # print('z_new', z_new)

        #z_new = np.zeros_like(z_new)
        z_new[:, draw_pos] = z_new[:, draw_pos]*np.sign(-tmp[draw_pos])

        tmp2[z == 1.] = 0

        # print('tmp2', tmp2)

        z_new = tmp2 + z_new

        # print('z-new', z_new)
        # print('tmp2', tmp2)

        rewards_atts = np.sum(z_new*(actions[1:, ] > 0), axis=1)

        rewards = [reward_dm]

        for r in rewards_atts:
            rewards.append(r)

        observations = None

        done = (self.step_count == self.max_steps)

        return observations, rewards, done


class modified_Blotto():
    """
    Modified Blotto game with multiple adversaries (we just care about positions
    where there has been some attack)
    """

    def __init__(self, max_steps, payout=50, batch_size=1, deterministic=True):
        self.max_steps = max_steps
        self.batch_size = batch_size
        #self.payout = payout
        self.available_actions = np.array([0, 1])
        self.step_count = 0
        self.deterministic = deterministic

    def reset(self):
        self.step_count = 0
        return

    def step(self, actions):
        """ action[0] is that of the defender """
        self.step_count += 1

        actions = np.asarray(actions)

        ## Defender's Reward
        att_rew = np.sum(actions[1:, ], axis=0)
        attacked_pos = att_rew > 0 ## indicates in which position attacks where performed

        tmp = actions[0, ] - att_rew
        tmp[np.logical_not(attacked_pos)] = 0.0

        # Code non-deterministic case ??

        tmp[tmp < 0] = -1 # Defender looses corresponding position
        tmp[tmp > 0] = 1  # Defender wins corresponding position
        reward_dm = tmp.sum()

        ## Attacker's Reward
        tmp_att = -tmp

        h = actions[1:] > 0
        units = tmp_att / np.sum(h, axis=0)
        units = np.nan_to_num(units)

        rewards_att = h*units
        rewards_atts = np.sum(rewards_att, axis=1)

        rewards = [reward_dm]

        for r in rewards_atts:
            rewards.append(r)

        observations = None

        done = (self.step_count == self.max_steps)

        return observations, rewards, done


class Urban():
    """
    A two-agent environment for a urban resource allocation problem.
    """

    def __init__(self):
        # The state is designated by s = (s_0, s_1, s_2, s_3)
        # s_0 represents wheter we are in the initial state or not
        # s_i, i>0 represent whether the attack was successful on the site i.
        self.state = np.array([1, 0, 0, 0])
        self.step_count = 0
        self.max_steps = 2  # as in the ARA for Urban alloc. paper
        self.payoffs = np.array([1., 0.75, 2.])  # v_i from the paper

        # Transition dynamics

        # p(s_1_i = 1 | d1_i, a_i)  for site i
        self.p_s1_d1_a = np.array([[0, 0.85, 0.95],
                                   [0, 0.6, 0.75],
                                   [0, 0.3, 0.5],
                                   [0, 0.05, 0.1],
                                   [0, 0,  0.05]])

        # p(s_2_i = 1 | s_1_i, d2_i) for site i
        self.p_s2_s1_d2 = np.array([[0, 0, 0, 0, 0],
                                    [1., 0.95, 0.8, 0.6, 0.4]])

        self.n_sites = 3
        self.k = 0.005
        self.rho = 0.1
        self.c_A = 10.
        self.c_D = 10.

        self.available_actions_DM = [i for i in range(5**self.n_sites)]   # up to four units in each site
        self.n_states = 2 ** (self.n_sites + 1)

    def state2idx(self, state):
        """
        In [19]: state = np.array([1, 0, 0, 1])
        In [20]: state2idx(state)
        Out[20]: 9
        """
        pows = np.array([1 << i for i in range(len(state))[::-1]])
        return np.dot(pows, state)

    def idx2state(self, idx):
        """
        In [28]: idx = 9
        In [30]: idx2state(idx)
        Out[30]: array([1, 0, 0, 1])
        """
        return (idx & (1 << np.arange(len(self.state))) > 0).astype(int)

    def actionDM2idx(self, a):
        """ Now we have 3 sites, in which we can defend with up to 5 units. """
        pows = np.array([5**i for i in range(self.n_sites)[::-1]])
        return np.dot(pows, a)

    def idx2actionDM(self, idx):
        return list(map(int, (list(np.base_repr(idx, 5, padding=3))[-self.n_sites:])))

    def valid_actionDM(self, state_idx, action_idx, prev_action_idx):

        action = self.idx2actionDM(action_idx)
        prev_action = self.idx2actionDM(prev_action_idx)
        state = self.idx2state(state_idx)

        if state[0] == 1: #initial state
            #print('a', action)
            return np.sum(action) == 4
        else:  # second move
            #print('b', action, prev_action)
            c1 = np.sum(action) == 4
            c2 = action[0] <= prev_action[0] + prev_action[1]
            c3 = action[1] <= prev_action[0] + prev_action[1] + prev_action[2]
            c4 = action[2] <= prev_action[1] + prev_action[2]
            return c1 & c2 & c3 & c4

    def reset(self):
        self.step_count = 0
        self.state = np.array([1, 0, 0, 0])
        return

    def step(self, action):

        # first action is that from the DM
        ac0, ac1 = action

        self.step_count += 1

        if self.step_count == 1:

            self.state = np.array([0, 0, 0, 0])
            for i in range(self.n_sites):
                p = self.p_s1_d1_a[ac0[i], ac1[i]]
                u = np.random.rand()
                if u <= p:
                    self.state[i + 1] = 1  # success

            rewards = [0., 0.]   # no rewards until end of episode
            observations = self.state

            done = False

            return observations, rewards, done

        elif self.step_count == 2:  # end of episode

            for i in range(self.n_sites):
                p = self.p_s2_s1_d2[self.state[i+1], ac0[i]]
                u = np.random.rand()
                if u <= p:
                    self.state[i + 1] = 1  # success

            done = True
            observations = self.state
            #print(np.dot(self.payoffs, self.state[1:]))
            rewards = [- np.exp(self.c_D * self.rho * np.dot(self.payoffs, self.state[1:])),
                       np.exp(self.c_A * np.dot(self.payoffs,self.state[1:])  - np.sum(ac1 * self.k))]  

            return observations, rewards, done


class SimpleCoin():
    """
    Simple Coin Game from LOLA paper, where state is just the color of the coin.
    """

    def __init__(self, max_steps, batch_size=1):
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.available_actions = np.array([0, 1])  # 1 pick coin.
        self.step_count = 0
        self.state = 0  # initially, coin is red (for first player)

    def reset(self):
        self.step_count = 0
        return

    def step(self, action):
        ac0, ac1 = action

        self.step_count += 1

        rewards = np.asarray([ac0, ac1])  # +1 point if thw agent picks coin.
        
        # conflict
        if ac0 and self.state == 1:
            rewards[1] -= 2
        
        if ac1 and self.state == 0:
            rewards[0] -= 2

        if np.random.rand() < 0.5:
            self.state = 0
        else:
            self.state = 1

        done = (self.step_count == self.max_steps)

        return self.state, rewards, done
#

class CoinGame():
    """
    Coin game environment for two agents on a grid. Who batlle for two coins.
    """

    def __init__(self, max_steps=5, size_square_grid=4, push_distance=1):
        
        self.max_steps = max_steps  # Maximum number of steps per episode
        self.step_count = 0  # Counter for steps taken in the current episode
        self.N = size_square_grid  # Number of rows and columns in a square grid
        
        self.push_distance = push_distance # Distance to push the players away from each other
        
        self.available_actions_DM = np.array([0, 1, 2, 3])  # Actions available to the decision-maker
        self.available_actions_Adv = np.array([0, 1, 2, 3])  # Actions available to the adversary
        
        self.actions_push = np.array([0,1]) # push or not push
        
        # Creating combined actions for both players
        grid_A, grid_B = np.meshgrid(self.available_actions_DM, self.actions_push)
        self.combined_actions_blue = np.column_stack([grid_A.ravel(), grid_B.ravel()])
        
        grid_A, grid_B = np.meshgrid(self.available_actions_Adv, self.actions_push)
        self.combined_actions_red = np.column_stack([grid_A.ravel(), grid_B.ravel()])

        
        self.blue_player_execution_prob = 0.8 # Probability of executing the intended action for the blue player
        self.red_player_execution_prob = 0.8 # Probability of executing the intended action for the red player
        
        # Player positions
        self.blue_player = np.array([self.N // 2, 0]) # Centered start at left edge
        self.red_player = np.array([self.N // 2, self.N -1])   # Centered start at right edge
        
        self.coin_1 = np.array([0, self.N // 2]) # Initial position [row, col] of the coin 1
        self.coin_2 = np.array([self.N - 1, self.N // 2]) # Initial position [row, col] of the coin 2
        
        # Keep track of whether coins are still available on the map
        self.coin1_available = True
        self.coin2_available = True
        
        # Seve initial position of players and coins for reseting the environment
        self.blue_player_initial = self.blue_player.copy()  # Initial position [row, col] of the blue player (DM)
        self.red_player_initial = self.red_player.copy()  # Initial position [row, col] of the red player (ADV)
        
        self.coin_1_initial = self.coin_1.copy() # Initial position [row, col] of the coin 1
        self.coin_2_initial = self.coin_2.copy() # Initial position [row, col] of the coin 2
        
        self.done = False # Flag to indicate if episode is done
        
        # Track who collected which coin
        self.blue_collected_coin1 = False
        self.blue_collected_coin2 = False
        self.red_collected_coin1 = False
        self.red_collected_coin2 = False
        

    def get_state(self):
        """
        Returns a unique integer representing the full state of the environment,
        based the positions of the blue player, red player, coin availability, and collected coin counts.

        Each entity’s 2D position on an N×N grid is flattened and combined using radix encoding,
        ensuring every state has a unique ID.
        """

        # Flatten 2D positions into 1D indices (column-major order)
        p1 = self.blue_player[0] + self.N * self.blue_player[1]  # Blue player's position
        p2 = self.red_player[0] + self.N * self.red_player[1]    # Red player's position
        
        # Coin availability on map
        c1_avail_val = 1 if self.coin1_available else 0
        c2_avail_val = 1 if self.coin2_available else 0
        
        # How many coins each player has collected *so far*
        # (0, 1, or 2 - though game ends at 2 for one player)
        # For state simplicity, let's use: 0 = 0 coins, 1 = 1 coin, 2 = 2 coins (winning state)
        blue_coins_collected_count = int(self.blue_collected_coin1) + int(self.blue_collected_coin2)
        red_coins_collected_count = int(self.red_collected_coin1) + int(self.red_collected_coin2)
        
        # Radix encoding:
        # Max values: p_flat=N*N-1, c_avail=1, collected_count=2
        # Order: P1_pos, P2_pos, C1_avail, C2_avail, P1_coll_count, P2_coll_count
        base_pos = self.N * self.N
        base_avail = 2 # (0 or 1)
        base_coll_count = 3 # (0, 1, or 2)

        state_id = p1
        state_id = state_id * base_pos + p2
        state_id = state_id * base_avail + c1_avail_val
        state_id = state_id * base_avail + c2_avail_val
        state_id = state_id * base_coll_count + blue_coins_collected_count
        state_id = state_id * base_coll_count + red_coins_collected_count
        
        return int(state_id)
    
    @property
    def n_states(self):
        # (N*N for P1_pos) * (N*N for P2_pos) * (2 for C1_avail) * (2 for C2_avail) *
        # (3 for P1_coll_count) * (3 for P2_coll_count)
        return (self.N**4) * (2**2) * (3**2)

    def reset(self):
        """
        Resets the environment to its initial state:
        - Step counter is set to 0
        - Player and coin positions are restored to their starting values
        """
        
        self.step_count = 0  # Reset step counter to start a new episode
        
        self.done = False # Reset done flag to indicate the episode is not finished

        # Set initial positions for the agents
        self.blue_player = self.blue_player_initial.copy()  # Starting position of the blue player
        self.red_player = self.red_player_initial.copy()  # Starting position of the red player

        # Set initial positions for the coins
        self.coin_1 = self.coin_1_initial.copy()   # Starting position of the coin 1
        self.coin_2 = self.coin_2_initial.copy()   # Starting position of the coin 2
        
        # Reset coin availability
        self.coin1_available = True
        self.coin2_available = True
        
        # Reset coin collection flags for each player
        self.blue_collected_coin1 = False
        self.blue_collected_coin2 = False
        self.red_collected_coin1 = False
        self.red_collected_coin2 = False

        return 
    
    def _are_players_adjacent(self, player1_pos, player2_pos):
        # Only checks if the players are horizontaly or verticaly adjacent, not diagonally
        return np.sum(np.abs(player1_pos - player2_pos)) == 1

    def step(self, action):
        """
        Executes one environment step given the actions of both agents.

        - Updates agent positions based on chosen actions
        - Handles coin collection and assigns corresponding rewards
        - Returns the new state, rewards for each agent, and whether the episode is done
        """
        
        # Unpack actions for both agents
        ac0, ac1 = action  # ac0: blue player's action, ac1: red player's action

        # Increment step count for this episode
        self.step_count += 1

        # Initialize step rewards
        reward_blue, reward_red = -0.1, -0.1
        
        # Store original positions for push logic
        original_blue_pos = self.blue_player.copy()
        original_red_pos = self.red_player.copy()
        
        # Check if players are adjacent 
        players_are_adjacent = self._are_players_adjacent(original_blue_pos, original_red_pos)
        
        # --- Effective actions ---
        
        # Define movement deltas for up, right, down, left
        deltas = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)])
        
        # TODO: Apply radix decoding instead of this.
        # Randomly select the actual action based on the execution probability of the blue player
        if np.random.rand() < self.blue_player_execution_prob:
            actual_action_blue = self.combined_actions_blue[ac0]
        else:
            num_total_blue_actions = len(self.combined_actions_blue)
            alternatives = [a for a in range(num_total_blue_actions) if a != ac0]
            actual_action_blue = self.combined_actions_blue[np.random.choice(alternatives)]   

        ## Red player movement 
        
        # Randomly select the actual action based on the execution probability of the red player
        if np.random.rand() < self.red_player_execution_prob:
            actual_action_red = self.combined_actions_red[ac1]
        else:
            num_total_red_actions = len(self.combined_actions_red)
            alternatives = [a for a in range(num_total_red_actions) if a != ac1]
            actual_action_red = self.combined_actions_red[np.random.choice(alternatives)]   
            
    
        # --- 1. Resolve Push Attempts ---
        
        if actual_action_blue[1] == 1 and actual_action_red[1] == 1:
            if players_are_adjacent:
               reward_blue = -0.05
               reward_red = -0.05
            else: 
               reward_blue = -0.05 # Small penalty for failed push (not adjacent)   
               reward_red = -0.05 # Small penalty for failed push (not adjacent) 
               
        elif actual_action_blue[1] == 1 and actual_action_red[1] != 1: 
            if players_are_adjacent: 
                push_direction = original_red_pos - original_blue_pos
                pushed_red_player_position = original_red_pos + self.push_distance * push_direction
                
                self.red_player = np.clip(pushed_red_player_position, 0, self.N-1)
                reward_blue = 0.2 # Reward blue for succesfull push
                reward_red = -0.2 # Penilize red for no being pushed
            else: 
                reward_blue = -0.05 # Penilize blue if push was chosen but agents were not adjacent
                
        elif actual_action_blue[1] != 1 and actual_action_red[1] == 1: 
            if players_are_adjacent: 
                push_direction = original_blue_pos - original_red_pos
                pushed_blue_player_position = original_blue_pos + self.push_distance * push_direction
                
                self.blue_player = np.clip(pushed_blue_player_position, 0, self.N-1)
                reward_blue = -0.2 # Reward blue for succesfull push
                reward_red = 0.2 # Penilize red for no being pushed
            else: 
                reward_red = -0.05 # Penilize red if push was chosen but agents were not adjacent
        
        # Apply movement
        if actual_action_blue[1] != 1:
            new_position = self.blue_player + deltas[actual_action_blue[0]]
            new_position = np.clip(new_position, 0, self.N - 1)
            self.blue_player = new_position
                
        # Apply movement
        if actual_action_red[1] != 1:
            new_position = self.red_player + deltas[actual_action_red[0]]
            new_position = np.clip(new_position, 0, self.N - 1)
            self.red_player = new_position

        # --- Coin collection logic ---
        
        # Coin 1
        if self.coin1_available:
            blue_on_coin_1 = np.array_equal(self.blue_player, self.coin_1)
            red_on_coin_1 = np.array_equal(self.red_player, self.coin_1)

            if blue_on_coin_1 and not red_on_coin_1:
                self.blue_collected_coin1 = True
                self.coin1_available = False
                reward_blue = 2.0 # Add to base penalty, or set directly
                reward_red = -0.5 # Optional: small penalty for opponent scoring
                
            elif red_on_coin_1 and not blue_on_coin_1:
                self.red_collected_coin1 = True
                self.coin1_available = False
                reward_red = 2.0
                reward_blue = -0.5

            #elif blue_on_coin_1 and red_on_coin_1: # Contested
                # reward_blue and reward_red keep their -0.1, or add a specific contest penalty

        # Coin 2
        if self.coin2_available:
            blue_on_coin_2 = np.array_equal(self.blue_player, self.coin_2)
            red_on_coin_2 = np.array_equal(self.red_player, self.coin_2)

            if blue_on_coin_2 and not red_on_coin_2:
                self.blue_collected_coin2 = True
                self.coin2_available = False
                reward_blue = 2.0
                reward_red = -0.5
                
            elif red_on_coin_2 and not blue_on_coin_2:
                self.red_collected_coin2 = True
                self.coin2_available = False
                reward_red = 2.0
                reward_blue = -0.5
                
            #elif blue_on_coin_2 and red_on_coin_2: # Contested
                

        # After all coin collection attempts for the step:
        if (self.blue_collected_coin1 and self.blue_collected_coin2) or \
            (self.red_collected_coin1 and self.red_collected_coin2):
            # Blue winds
            if (self.blue_collected_coin1 and self.blue_collected_coin2):
                reward_blue = 10 # Win bonus
                reward_red = -5  # Loss penalty
                
            # Red wins
            else: 
                reward_red = 10 # Win bonus
                reward_blue = -5 # Loss penalty
            self.done = True
        elif (self.blue_collected_coin1 and self.red_collected_coin2) or \
            (self.red_collected_coin1 and self.blue_collected_coin2 ):
                reward_blue = -1
                reward_red = -1
                self.done = True
        
        # Check if max step coun is reached
        if self.step_count == self.max_steps:
            self.done = True

        # Return new state, rewards, and done flag
        return self.get_state(), np.array([reward_blue, reward_red]), self.done