import logging

# --- Utility Functions ---
# Expose helper functions if they are needed externally,
# otherwise they can be imported within the modules themselves.
from .utils import softmax, manhattan_distance

# --- Base Agents ---
# Import all the Base agents
from .base import (
    BaseAgent,
    LearningAgent
)

# --- Heuristic Agents ---
# Import all the Manhattan agents
from .heuristic import (
    ManhattanAgent,
    ManhattanAgent_Passive,
    ManhattanAgent_Aggressive,
    ManhattanAgent_Ultra_Aggressive
)

# --- Q-Learning Agents ---
# Import the independent Q-learning agents
from .q_learning import (
    IndQLearningAgent,
    IndQLearningAgentSoftmax
)

# --- Level-K Q-Learning Agents ---
# Import the Level-K Q-learning agents
from .level_k_q import (
    LevelKQAgent,
    LevelKQAgentSoftmax
)

# --- Level-K MDP Dynamic Programming Agents ---
# Import the MDP DP agent family
from .mdp_dp import (
    LevelK_MDP_DP_Agent_Stationary,
    LevelK_MDP_DP_Agent_NonStationary,
    LevelK_MDP_DP_Agent_Dynamic
)

# --- Level-K TMDP Dynamic Programming Agents ---
# Import the TMDP DP agent family
from .tmdp_dp import (
    LevelK_TMDP_DP_Agent_Stationary,
    LevelK_TMDP_DP_Agent_NonStationary,
    LevelK_TMDP_DP_Agent_Dynamic
)

# --- Offline Solver Agents ---
# Import the perfect model solver
from .offline_solvers import (
    TMDP_DPAgent_PerfectModel,
    DPAgent_PerfectModel
)

logging.info("Agents package successfully initialized.")