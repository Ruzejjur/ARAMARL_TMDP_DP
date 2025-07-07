import numpy as np
import copy
from tqdm.notebook import tqdm

from .dp import LevelKDPAgent_Stationary


class DPAgent_PerfectModel(LevelKDPAgent_Stationary):
    """
    Computes the optimal policy (a best response) against a known, fixed opponent
    policy using offline value iteration.

    This agent inherits the efficient, vectorized model-handling and Bellman update
    logic from LevelKDPAgent_Stationary and overrides the opponent model to be
    a "perfect" (i.e., known and fixed) model of a ManhattanAgent.
    """
    def __init__(self, action_space, enemy_action_space, n_states, gamma, player_id, env, enemy):
        # Initialize as a Level-1 DP Agent to leverage its pre-computation methods.
        # We pass epsilon=0 because the final policy will be deterministic.
        super().__init__(k=1, action_space=action_space, enemy_action_space=enemy_action_space,
                         n_states=n_states, epsilon=0, gamma=gamma, player_id=player_id, env=env)

        # Define the fixed opponent we are solving against.
        self.enemy = copy.deepcopy(enemy)

        # Pre-compute the opponent's full, fixed policy table.
        print("Pre-computing the perfect opponent policy table...")
        self.opponent_policy_table = self._precompute_opponent_policy()
        print("Opponent policy table finished.")

        # The final optimal policy π*(s) will be stored here.
        self.optim_policy_table = np.zeros(self.n_states, dtype=int)

        # Run offline value iteration to find the optimal value function V*(s, b)
        # and then extract the final policy.
        self.run_value_iteration()
        self.extract_optimal_policy()
        

    def _precompute_opponent_policy(self):
        """Computes the opponent's action probability distribution for every state."""
        policy_table = np.zeros((self.n_states, self.num_Adv_actions))
        for s in tqdm(range(self.n_states), desc="Pre-computing opponent policy table"):
            try:
                # The act method of ManhattanAgent sets its `possible_actions` attribute.
                self.enemy.act(s, None)
                possible_actions = self.enemy.possible_actions

                if possible_actions is not None and len(possible_actions) > 0:
                    policy_table[s, possible_actions] = 1.0 / len(possible_actions)
                else: # Handle cases where no specific action is determined (e.g., no coins left)
                    policy_table[s, :] = 1.0 / self.num_Adv_actions # Assume uniform random
            except (ValueError, IndexError): # Catch errors from invalid/unreachable states
                policy_table[s, :] = 1.0 / self.num_Adv_actions
        return policy_table

    def get_opponent_policy(self, obs):
        """
        Overrides the parent method. Instead of learning, it returns the
        pre-computed policy for the fixed opponent.
        """
        return self.opponent_policy_table[obs]

    def run_value_iteration(self, theta=1e-2, max_iters=10000):
        """
        Performs offline, in-place (asynchronous) value iteration on the V(s, b) table
        until convergence. This is more memory-efficient as it avoids copying the
        large V matrix in each iteration.
        """
        print("Starting offline in-place value iteration on V(s, b)...")
        
        progress_bar = tqdm(range(max_iters), desc="Value Iteration", unit="it")
        
        for i in progress_bar:
            delta = 0  # Initialize max change for this sweep to zero

            # Loop through all states to update their values
            for s in range(self.n_states):
                
                if self._is_terminal_state(s):
                    continue  # Skip updates for terminal states, their value is 0
                
                # Get outcomes for the current state from the lookup tables
                R_exec_obs = self.R_lookup[s, :, :]
                S_prime_exec_obs = self.S_prime_lookup[s, :, :]
                
                # Calculate expected future values for all outcomes
                future_V_values_executed = self._calculate_expected_future_values(S_prime_exec_obs)
                
                # Loop through all possible opponent actions `b`
                for b_opp in range(self.num_Adv_actions):
                    
                    # Store the value of V(s, b_opp) before the update
                    v_s_b_old = self.V[s, b_opp]
                    
                    # Extract transition probabilities conditioned on the opponent's TAKEN action
                    prob_exec_tensor_fixed_b = self.prob_exec_tensor[:, b_opp, :, :]
                    
                    # Calculate expected rewards and future values for each of our INTENDED actions
                    expected_rewards_all_idm = np.einsum('ikl,kl->i', prob_exec_tensor_fixed_b, R_exec_obs)
                    expected_future_V_all_idm = np.einsum('ikl,kl->i', prob_exec_tensor_fixed_b, future_V_values_executed)
                    
                    Q_values_for_dm_intentions = expected_rewards_all_idm + self.gamma * expected_future_V_all_idm
                    # The new value is the max Q-value
                    v_s_b_new = np.max(Q_values_for_dm_intentions)

                    # --- Update V in-place and track the change ---
                    self.V[s, b_opp] = v_s_b_new
                    delta = max(delta, np.abs(v_s_b_new - v_s_b_old))
                    
            # --- TQDM Update Logic ---
            # The progress bar updates automatically with each loop.
            # We simply set the postfix to show the latest delta value.
            progress_bar.set_postfix(delta=f"{delta:.6f}/{theta:.4f}", refresh=True)
            
            # --- Convergence Check ---
            if delta < theta:
                # When using tqdm as an iterator, a 'break' is sufficient.
                # It will automatically close the bar and stop iterating.
                print(f"\nValue iteration converged after {i + 1} iterations.")
                return

        print(f"Value iteration did not converge after {max_iters} iterations.")


    def extract_optimal_policy(self):
        """
        Extracts the optimal deterministic policy π*(s) after value iteration.
        This reuses the parent's `optim_act` method.
        """
        print("Extracting optimal policy...")
        for s in range(self.n_states):
            # The parent's optim_act function finds the best action 'a' that maximizes
            # the expected Q-value, marginalized over the opponent's policy at state 's'.
            # This is exactly what we need for policy extraction.
            self.optim_policy_table[s] = self.optim_act(s)
        print("Optimal policy extracted.")

    def act(self, obs, env=None):
        """
        Returns the pre-computed optimal action for the given state.
        """
        return self.optim_policy_table[obs]

    def update(self, obs, actions, rewards, new_obs):
        """
        This agent is an offline solver. It does not learn during episodes.
        """
        pass