from abc import ABC, abstractmethod
from typing import Optional

# --- Type Aliases for Readability ---
State = int
Action = int
Reward = float


class BaseAgent(ABC):
    """
    The minimal interface for any agent.
    
    Every agent in the simulation must be able to select an action based on an observation.
    """
    @abstractmethod
    def act(self, obs: State, env=None) -> Action:
        """
        Selects an action given the current observation.
        
        Args:
            obs: The current state observation.
            env: The environment instance. This is optional and only used by agents
                 that need to query the environment's current state (e.g., for
                 non-stationary transition probabilities).
                 
        Returns:
            The integer ID of the chosen action.
        """
        pass

class LearningAgent(BaseAgent):
    """
    The interface for agents that learn from experience during training.
    
    These agents must implement methods for updating their models/policies
    and for managing their exploration rate.
    """
    @abstractmethod
    def update(self, obs: State, actions: tuple[Action, Action], new_obs: State, rewards: Optional[tuple[Reward, Reward]]):
        """Updates the agent's internal model based on a transition."""
        pass
    
    @abstractmethod
    def update_epsilon(self, new_epsilon_agent: float, new_epsilon_lower_k_level: Optional[float]):
        """Updates the agent's exploration parameter."""
        pass