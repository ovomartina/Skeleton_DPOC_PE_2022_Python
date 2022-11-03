from ComputeTerminalStateIndex import ComputeTerminalStateIndex


class Constants:
    M = 19
    N = 11
    mapSize = [M, N]

    # Plotting options
    PLOT_POLICY = True
    PLOT_COST = False

    # Global problem parameters
    SF = 0.5           # Shielding factor against radiation in the lower dimension
    NA = 3             # Time steps required to fight with alien
    NB = 10            # Time steps required to bring robot to base when damaged
    P_DISTURBED = 0.2  # Probability that the robot is disturbed due to radiation
    P_PROTECTED = 0.6  # Probability that the robot successfully protects its gems against aliens

    # IDs of elements in the map matrix
    FREE = 0
    OBSTACLE = 1
    PORTAL = 2
    MINE = 3
    FACTORY = 4
    BASE = 5
    ALIEN = 6

    # Index of each action in the P and G matrices. Use this ordering
    NORTH = 0
    SOUTH = 1
    WEST = 2
    EAST = 3
    STAY = 4

    # Index of whether the robot is carrying gems
    EMPTY = 0
    GEMS = 1

    # Index of dimensions
    UPPER = 0
    LOWER = 1

    a = ComputeTerminalStateIndex()