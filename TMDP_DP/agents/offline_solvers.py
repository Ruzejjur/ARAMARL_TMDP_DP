import numpy as np
import copy
from tqdm.notebook import tqdm

from typing import Optional

from .level_k_dp import LevelKDPAgent_Stationary
from .heuristic import ManhattanAgent

# --- Type Aliases for Readability ---
State = int
Action = int
Reward = float
Policy = np.ndarray
ValueFunction = np.ndarray
ActionDetails = np.ndarray


class TMDDPAgent_PerfectModel(LevelKDPAgent_Stationary):
    """
    Computes the optimal policy (a best response) against a known, fixed opponent
    policy using offline value iteration.

    This agent acts as an offline solver. It first builds a "perfect" model of a
    specific, non-learning opponent (e.g., a hard-coded heuristic agent). Then, it
    runs value iteration until convergence to find the optimal value function
    against that opponent's fixed policy. Finally, it extracts a deterministic
    optimal policy.

    It inherits from `LevelKDPAgent_Stationary` to reuse the efficient,
    vectorized model-handling and Bellman update logic.

    Attributes:
        opponent (object): A deep copy of the fixed opponent agent this agent is
                           solving a best response for.
        opponent_policy_table (np.ndarray): A pre-computed table of shape
                                           (n_states, num_opponent_actions)
                                           storing the opponent's fixed policy.
        optim_policy_table (np.ndarray): A table of shape (n_states,) storing the
                                         final, optimal action for each state.
                                         
        For other atribute description check parent.
    """
    def __init__(self, action_space: np.ndarray, opponent_action_space: np.ndarray, n_states: int, gamma: float, initial_V_value: float, player_id: int,
                 termination_criterion: float, value_iteration_max_num_of_iter: int, env, opponent: ManhattanAgent):
        # Initialize as a Level-1 DP Agent to leverage its pre-computation,
        # lookup tables, and vectorized calculation methods.
        # k=1 is sufficient as we are pre-solving the optimal policy.
        # epsilon=0 is used because the final policy will be purely deterministic
        # and there is no exploration during online interaction.
        
        if not isinstance(opponent, ManhattanAgent):
            raise TypeError(f"The provided opponent must be a ManhattanAgent class instance, but got {type(opponent).__name__} instead.")
        
        super().__init__(k=1, action_space=action_space, opponent_action_space=opponent_action_space, lower_level_k_epsilon=0,
                         n_states=n_states, epsilon=0, gamma=gamma, initial_V_value=initial_V_value,
                         player_id=player_id, env=env)

        # Agent specific initialisation
        self.termination_criterion = termination_criterion
        self.value_iteration_max_num_of_iter = value_iteration_max_num_of_iter
        
        # Store a deep copy of the fixed opponent we are solving against.
        self.opponent = copy.deepcopy(opponent)

        # --- Offline Solving Pipeline ---
        # Pre-compute the opponent's full, fixed policy for every state.
        print("Pre-computing the perfect opponent policy table...")
        self.opponent_policy_table = self._precompute_opponent_policy()
        print("Opponent policy table finished.")

        # This table will store the final optimal policy π*(s).
        self.optim_policy_table = np.zeros(self.n_states, dtype=int)

        # Run value iteration to find the optimal value function V*(s, b).
        self.run_value_iteration(termination_criterion=self.termination_criterion, value_iteration_max_num_of_iter=self.value_iteration_max_num_of_iter)
        # Extract the final deterministic policy from the converged V-function.
        self.extract_optimal_policy()
        

    def _precompute_opponent_policy(self):
        """
        Computes and stores the opponent's action probability distribution for
        every state in the environment.

        It iterates through all states, queries the provided opponent agent for its
        action distribution, and stores it in a table.

        Returns:
            np.ndarray: A policy table of shape (n_states, num_opponent_actions).
        """
        policy_table = np.zeros((self.n_states, self.num_opponent_actions))
        for s in tqdm(range(self.n_states), desc="Pre-computing opponent policy table"):
            try:
                # The provided opponent's `act` method is expected to determine
                # its set of preferred actions. Here, we rely on the implementation
                # detail that it sets a `optimal_actions` attribute defining all
                # equaly optimal actions of the opponent.
                self.opponent.act(s, None)
                optimal_actions = self.opponent.optimal_actions_cache

                if optimal_actions is not None and len(optimal_actions) > 0:
                    # Assume the opponent chooses uniformly among its possible optimal actions.
                    policy_table[s, optimal_actions] = 1.0 / len(optimal_actions)
                else: 
                    # Handle cases where no specific action is determined (e.g., in a terminal state).
                    # Assume a uniform random policy as a fallback.
                    policy_table[s, :] = 1.0 / self.num_opponent_actions
            except (ValueError, IndexError): # Catch errors from invalid/unreachable states
                policy_table[s, :] = 1.0 / self.num_opponent_actions
        return policy_table

    def get_opponent_policy(self, obs: State):
        """
        Overrides the parent method. Instead of learning or using a recursive
        model, it simply returns the pre-computed policy for the fixed opponent
        from the lookup table.

        Args:
            obs: The current state.

        Returns:
            An array representing the opponent's action probability distribution.
        """
        return self.opponent_policy_table[obs]

    def run_value_iteration(self, termination_criterion: float, value_iteration_max_num_of_iter: int):
        """
        Performs offline, in-place (asynchronous) value iteration on the V(s, b)
        table until convergence.

        "In-place" means the V-table is updated directly in each sweep, which is
        more memory-efficient than creating a new copy of V for each iteration.

        Args:
            termination_criterion (float): The convergence threshold. Iteration stops when the
                           maximum change in the value function is less than termination_criterion.
            value_iteration_max_num_of_iter (int): The maximum number of iterations to perform.
        """
        print("Starting offline in-place value iteration on V(s, b)...")
        
        progress_bar = tqdm(range(value_iteration_max_num_of_iter), desc="Value Iteration", unit="it")
        
        for i in progress_bar:
            delta = 0  # Tracks the maximum change in V(s,b) during a sweep.

            # Loop through all states to update their values.
            for s in range(self.n_states):
                
                if self._is_terminal_state(s):
                    continue  # The value of terminal states is always 0.
                
                # Get pre-computed outcomes from the lookup tables for state 's'.
                rewards_executed = self.r_lookup[s, :, :]
                s_primes_executed = self.s_prime_lookup[s, :, :]
                
                # Calculate expected future values for all possible resulting states.
                # Note: This step uses the perfect opponent model via get_opponent_policy.
                future_V_values_executed = self._calculate_expected_future_values(s_primes_executed)
                
                # Loop through all possible opponent actions `b` to update V(s,b).
                for opponent_action in range(self.num_opponent_actions):
                    
                    # Store the value of V(s, opponent_action) before the update
                    v_s_b_old = self.V[s, opponent_action]
                    
                    # Get transition probabilities conditioned on the opponent's INTENDED action.
                    prob_exec_tensor_fixed_opp_action = self.prob_exec_tensor[:, opponent_action, :, :]
                    
                    # Calculate expected rewards and future values for each of agents INTENDED actions.
                    expected_rewards_per_action = np.einsum('ikl,kl->i', prob_exec_tensor_fixed_opp_action, rewards_executed, optimize=True)
                    expected_future_V_per_action = np.einsum('ikl,kl->i', prob_exec_tensor_fixed_opp_action, future_V_values_executed, optimize=True)
                    
                    # Q(s, a | b) = E[R(s,a,b)] + gamma * E[V(s')]
                    q_values_for_actions = expected_rewards_per_action + self.gamma * expected_future_V_per_action
                    
                    # The new value is the maximum Q-value over our possible actions.
                    v_s_b_new = np.max(q_values_for_actions)

                    # --- Update V in-place and track the change ---
                    self.V[s, opponent_action] = v_s_b_new
                    delta = max(delta, np.abs(v_s_b_new - v_s_b_old))
                    
            # Update the progress bar with the current delta.       
            progress_bar.set_postfix(delta=f"{delta:.6f}/{termination_criterion:.4f}", refresh=True)
            
            # --- Convergence Check ---
            if delta < termination_criterion:
                print(f"\nValue iteration converged after {i + 1} iterations.")
                return

        print(f"Value iteration did not converge after {value_iteration_max_num_of_iter} iterations.")


    def extract_optimal_policy(self):
        """
        Extracts the optimal deterministic policy π*(s) after value iteration.

        For each state, it calculates the best action for the agent, assuming the
        opponent plays according to its pre-computed fixed policy. This reuses
        the parent's `optim_act` method, which performs exactly this calculation.
        """
        
        print("Extracting optimal policy...")
        for s in range(self.n_states):
            # The parent's optim_act function finds the action 'a' that maximizes
            # the expected Q-value, marginalized over the opponent's policy at state 's'.
            # This is exactly what we need to determine the best response at each state.
            self.optim_policy_table[s] = self.optim_act(s)
        print("Optimal policy extracted.")

    def act(self, obs: State, env=None) -> Action:
        """
        Returns the pre-computed optimal action for the given state.

        Args:
            obs: The current state observation.
            env: The environment (not used in this offline agent).

        Returns:
            The optimal action for the given state.
        """
        
        return self.optim_policy_table[obs]

    def update(self, obs: State, actions: tuple[Action, Action], new_obs: State, rewards: Optional[tuple[Reward, Reward]] = None):
        """
        This agent is an offline solver and does not learn during online interaction.
        This method is implemented to satisfy the agent interface but performs no action.
        """
        pass