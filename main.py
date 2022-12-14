import scipy.io

from ComputeStageCosts import ComputeStageCosts
from ComputeTerminalStateIndex import *
from ComputeTransitionProbabilities import *
from GenerateWorld import *
from MakePlots import *
from Solution import *

"""
Set to true to generate a random map of size mapSize, else set to false
to load the pre-existing example map
"""
generateRandomWorld = False

"""
Generate map
map(m,n) represents the cell type at indices (m,n) according to the axes
specified in the PDF.
"""

print('Generate map')
if generateRandomWorld:
    map_world = GenerateWorld(Constants.M, Constants.N)
else:
    # We can load a pre-generated map_world
    data = scipy.io.loadmat('test.mat')
    map_world = data["map"]
MakePlots(map_world)
"""
 Generate state space
 Generate a (K x 4)-matrix 'stateSpace', where each accessible cell is
 represented by 4 rows.
"""

print('Generate state space')
stateSpace = []
for m in range(0, len(map_world)):
    for n in range(0, len(map_world[0])):
        if map_world[m][n] != Constants.OBSTACLE:
            stateSpace.extend([[m, n, Constants.EMPTY, Constants.UPPER],
                               [m, n, Constants.GEMS, Constants.UPPER],
                               [m, n, Constants.EMPTY, Constants.LOWER],
                               [m, n, Constants.GEMS, Constants.LOWER]])

# State space size
K = len(stateSpace)

# Set the following to True as you progress with the files
terminalStateIndexImplemented = False
transitionProbabilitiesImplemented = False
stageCostsImplemented = False
SolutionImplemented = False

# Compute the terminal state index
if terminalStateIndexImplemented:

    # TODO: Question a)
    TERMINAL_STATE_INDEX = ComputeTerminalStateIndex(stateSpace, map_world)

# Compute transition probabilities
if transitionProbabilitiesImplemented:
    print('Compute transition probabilities')

    """
        Compute the transition probabilities between all states in the
        state space for all control inputs.
        The transition probability matrix has the dimension (K x K x L), i.e.
        the entry P(i, j, l) represents the transition probability from state i
        to state j if control input l is applied.
    """

    # TODO: Question b)
    P = ComputeTransitionProbabilities(stateSpace, map_world, K)

# Compute stage costs
if stageCostsImplemented:
    print("Compute stage costs")

    """
        Compute the stage costs for all states in the state space for all
        control inputs.
        The stage cost matrix has the dimension (K x L), i.e. the entry G(i, l)
        represents the cost if we are in state i and apply control input l.
    """

    # TODO: Question c)  
    G = ComputeStageCosts(stateSpace, map_world, K)

# Solve the stochastic shortest path problem
if SolutionImplemented:
    print('Solve stochastic shortest path problem')

    # TODO: Question d)
    [J_opt, u_opt_ind] = Solution(P, G, K, TERMINAL_STATE_INDEX)

    if len(J_opt) != K or len(u_opt_ind) != K:
        print('[ERROR] the size of J and u must be K')

# Plot results
print('Plot results')
if SolutionImplemented:
    MakePlots(map_world, stateSpace, J_opt, u_opt_ind, TERMINAL_STATE_INDEX, "Solution")

# Terminated
print('Terminated')
