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

# --- Level-K Dynamic Programming Agents ---
# Import the DP agent family
from .level_k_dp import (
    LevelKDPAgent_Stationary,
    LevelKDPAgent_NonStationary,
    LevelKDPAgent_Dynamic
)

# --- Offline Solver Agents ---
# Import the perfect model solver
from .offline_solvers import (
    TMDP_DPAgent_PerfectModel,
    DPAgent_PerfectModel
)

print("Agents package successfully initialized.")