import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cmx

from Constants import *


def MakePlots(map_world, *args):

    """
    Plot the map of the world. If extra args are passed to the function, plot the cost-to-go and the control action
    associated to each state.

    @type  map_world: (M x N)-matrix
    @param  map_world: A matrix describing the terrain.
          With values: FREE OBSTACLE PORTAL ALIEN MINE
          FACTORY BASE

    @param args:
        (optional)  Input argument list:
            0: 	A (K x 2)-matrix 'stateSpace', where each row
                  represents an element of the state space.
            1:  A (K x 1 )-matrix 'J' containing the optimal cost-to-go
                for each element of the state space.
            2:  A (K x 1 )-matrix containing the index of the optimal
                control input for each element of the state space.
            3:  A integer representing the terminal state index
            4:  Title
    """
    if len(args) < 2:
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(18.5, 10.5, forward=True)
        PlotMap(map_world, Constants.UPPER, axs[Constants.UPPER])
        axs[Constants.UPPER].set_title('Upper World', fontsize=20)
        axs[Constants.UPPER].set_aspect('equal')

        PlotMap(map_world, Constants.LOWER, axs[Constants.LOWER])
        axs[Constants.LOWER].set_title('Lower World', fontsize=20)
        axs[Constants.LOWER].set_aspect('equal')

        fig.suptitle(f'Map (width={map_world.shape[0]}, height={map_world.shape[1]})', fontsize=40)
        plt.show()

    else:
        stateSpace = args[0]
        stateSpace = np.array(stateSpace)
        J_opt = args[1]
        u = args[2]
        terminal_state_index = args[3]
        title = args[4]
        fig, axs = plt.subplots(2, 2)
        fig.set_size_inches(18.5, 18.5, forward=True)

        # remove terminal state from plotted results

        stateSpace = np.delete(stateSpace, terminal_state_index, 0)
        J_opt = np.delete(J_opt, terminal_state_index)
        u = np.delete(u, terminal_state_index)

        # split states
        with_gems_upper = \
            np.where(np.equal(stateSpace[:, 2], Constants.GEMS) & np.equal(stateSpace[:, 3], Constants.UPPER))[0]
        with_gems_lower = \
            np.where(np.equal(stateSpace[:, 2], Constants.GEMS) & np.equal(stateSpace[:, 3], Constants.LOWER))[0]
        without_gems_upper = \
            np.where(np.equal(stateSpace[:, 2], Constants.EMPTY) & np.equal(stateSpace[:, 3], Constants.UPPER))[0]
        without_gems_lower = \
            np.where(np.equal(stateSpace[:, 2], Constants.EMPTY) & np.equal(stateSpace[:, 3], Constants.LOWER))[0]

        # Call the plotting function for states with/without gems, upper/lower world
        # separately

        #  sub-plot without gems upper world
        PlotMap(map_world, Constants.UPPER, axs[Constants.EMPTY][Constants.UPPER], stateSpace[without_gems_upper, :],
                J_opt[without_gems_upper], u[without_gems_upper], 'Upper World without Gems')

        # sub-plot with gems upper world
        PlotMap(map_world, Constants.UPPER, axs[Constants.GEMS][Constants.UPPER], stateSpace[with_gems_upper, :],
                J_opt[with_gems_upper], u[with_gems_upper], 'Upper World with Gems')

        #  sub-plot without gems lower world
        PlotMap(map_world, Constants.LOWER, axs[Constants.EMPTY][Constants.LOWER], stateSpace[without_gems_lower, :],
                J_opt[without_gems_lower], u[without_gems_lower], 'Lower World without Gems')

        # sub-plot with gems lower world
        PlotMap(map_world, Constants.LOWER, axs[Constants.GEMS][Constants.LOWER], stateSpace[with_gems_lower, :],
                J_opt[with_gems_lower], u[with_gems_lower], 'Lower World with Gems')

        fig.suptitle(title, fontsize=40)
        plt.show()


def PlotMap(map_world, *args):
    """
      Plot a map, the costs for each cell and the control action in
      each cell.

    @type  map_world: (M x N)-matrix
    @param  map_world:      A matrix describing the terrain.
          With values: FREE OBSTACLE PORTAL ALIEN MINE
          FACTORY BASE

    @param *args (optional):
        Input argument list:
            0:  Constant representing in which world we are (UPPER or LOWER)
            1:  Plotting axis
            2: 	A (K x 2)-matrix 'stateSpace', where each row
                  represents an element of the state space.
            3:  A (K x 1 )-matrix 'J' containing the optimal cost-to-go
                for each element of the state space.
            4:  A (K x 1 )-matrix containing the index of the optimal
                control input for each element of the state space.
            5:  Title
    """

    # plot parameters
    # obstacles
    obstacleColor = [100 / 255, 100 / 255, 100 / 255]
    # shooter
    portalColor = [50 / 255, 205 / 255, 50 / 255]
    # alien
    alienColor = [242 / 255, 61 / 255, 0 / 255]
    # pick-up
    mineColor = [255 / 255, 230 / 255, 0 / 255]
    # drop-off
    factoryColor = [212 / 255, 42 / 255, 255 / 255]
    # base
    baseColor = [112 / 255, 181 / 255, 255 / 255]

    world = args[0]
    ax = args[1]

    # Color cells
    cMap = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cMap)

    if len(args) > 2:
        x = args[2]
        J = args[3]
        u_opt_ind = args[4]
        title = args[5]
        maxJ = np.max(J)
        for i in range(0, len(J)):
            if world == Constants.UPPER:
                xCorner, yCorner = getUpperTriangle(x[i, 0], x[i, 1])
            else:
                xCorner, yCorner = getLowerTriangle(x[i, 0], x[i, 1])
            vertices = np.array([xCorner, yCorner]).transpose()
            colorVal = scalarMap.to_rgba(J[i] / maxJ)
            triangle = plt.Polygon(vertices, color=[0.5 * i for i in colorVal], linewidth=0)
            ax.add_patch(triangle)
        ax.set_title(title, fontsize=20)

        #  plot arrows and expected costs
        for i in range(0, len(J)):
            x_i = x[i, :]
            if Constants.PLOT_POLICY:
                center = [x_i[0] + 1, x_i[1] * 2 + 1]
                match u_opt_ind[i]:
                    case Constants.NORTH:
                        u_i = np.array([0, -1])
                    case Constants.SOUTH:
                        u_i = np.array([0, 1])
                    case Constants.EAST:
                        u_i = np.array([1, 0])
                    case Constants.WEST:
                        u_i = np.array([-1, 0])
                    case Constants.STAY:
                        u_i = np.array([0, 0])
                startPt = np.copy(center)
                endPt = center + 0.4 * u_i
                arrow(startPt, endPt, ax)

            if Constants.PLOT_COST:
                if world == Constants.UPPER:
                    if x_i[0] % 2 == 0:
                        if x_i[1] % 2 == 0:
                            ax.text(x_i[0] + 0.5, x_i[1] * 2 + 0.2, round(J[i], 1), fontsize=8, color="black")
                        else:
                            ax.text(x_i[0] + 0.5, x_i[1] * 2 + 1.5, round(J[i], 1), fontsize=8, color="black")
                    else:
                        if x_i[1] % 2 == 0:
                            ax.text(x_i[0] + 0.5, x_i[1] * 2 + 1.5, round(J[i], 1), fontsize=8, color="black")
                        else:
                            ax.text(x_i[0] + 0.5, x_i[1] * 2 + 0.2, round(J[i], 1), fontsize=8, color="black")
                else:
                    if x_i[0] % 2 == 0:
                        if x_i[1] % 2 == 0:
                            ax.text(x_i[0] + 0.5, x_i[1] * 2 + 1.5, round(J[i], 1), fontsize=8, color="black")
                        else:
                            ax.text(x_i[0] + 0.5, x_i[1] * 2 + 0.2, round(J[i], 1), fontsize=8, color="black")
                    else:
                        if x_i[1] % 2 == 0:
                            ax.text(x_i[0] + 0.5, x_i[1] * 2 + 0.2, round(J[i], 1), fontsize=8, color="black")
                        else:
                            ax.text(x_i[0] + 0.5, x_i[1] * 2 + 1.5, round(J[i], 1), fontsize=8, color="black")

    # Plot obstacles
    obstacle = np.where(map_world == Constants.OBSTACLE)
    for i in range(len(obstacle[0])):
        if world == Constants.UPPER:
            [xCorner, yCorner] = getUpperTriangle(obstacle[0][i], obstacle[1][i])
        else:
            [xCorner, yCorner] = getLowerTriangle(obstacle[0][i], obstacle[1][i])
        vertices = np.array([xCorner, yCorner]).transpose()
        triangle = plt.Polygon(vertices, color=obstacleColor)
        ax.add_patch(triangle)

    # Plot portals
    portals = np.where(map_world == Constants.PORTAL)
    for i in range(len(portals[0])):
        if world == Constants.UPPER:
            [xCorner, yCorner] = getUpperTriangle(portals[0][i], portals[1][i])
        else:
            [xCorner, yCorner] = getLowerTriangle(portals[0][i], portals[1][i])
        vertices = np.array([xCorner, yCorner]).transpose()
        triangle = plt.Polygon(vertices, color=portalColor)
        ax.add_patch(triangle)

    # Plot aliens
    aliens = np.where(map_world == Constants.ALIEN)
    for i in range(len(aliens[0])):
        if world == Constants.LOWER:
            [xCorner, yCorner] = getLowerTriangle(aliens[0][i], aliens[1][i])
            vertices = np.array([xCorner, yCorner]).transpose()
            triangle = plt.Polygon(vertices, color=alienColor)
            ax.add_patch(triangle)

    # Plot mine point
    mine = np.where(map_world == Constants.MINE)
    if world == Constants.LOWER:
        [xCorner, yCorner] = getLowerTriangle(mine[0][0], mine[1][0])
        vertices = np.array([xCorner, yCorner]).transpose()
        triangle = plt.Polygon(vertices, color=mineColor)
        ax.add_patch(triangle)

    # Plot factory point
    factory = np.where(map_world == Constants.FACTORY)
    if world == Constants.UPPER:
        [xCorner, yCorner] = getUpperTriangle(factory[0][0], factory[1][0])
        vertices = np.array([xCorner, yCorner]).transpose()
        triangle = plt.Polygon(vertices, color=factoryColor)
        ax.add_patch(triangle)

    # Plot base
    base = np.where(map_world == Constants.BASE)
    if world == Constants.UPPER:
        [xCorner, yCorner] = getUpperTriangle(base[0][0], base[1][0])
        vertices = np.array([xCorner, yCorner]).transpose()
        triangle = plt.Polygon(vertices, color=baseColor)
        ax.add_patch(triangle)

    # Add letters to cells with features
    if not Constants.PLOT_COST or len(args) <= 2:
        [xPortal, yPortal] = np.where(map_world == Constants.PORTAL)
        for i in range(0, len(xPortal)):
            if world == Constants.UPPER:
                if (xPortal[i] % 2 == 0 and yPortal[i] % 2 == 0) or (xPortal[i] % 2 != 0 and yPortal[i] % 2 != 0):
                    ax.text(xPortal[i] + 0.7, yPortal[i] * 2 + 0.2, 'P', fontsize=20)
                else:
                    ax.text(xPortal[i] + 0.7, yPortal[i] * 2 + 1.1, 'P', fontsize=20)
            else:
                if (xPortal[i] % 2 == 0 and yPortal[i] % 2 == 0) or (xPortal[i] % 2 != 0 and yPortal[i] % 2 != 0):
                    ax.text(xPortal[i] + 0.7, yPortal[i] * 2 + 1.1, 'P', fontsize=20)
                else:
                    ax.text(xPortal[i] + 0.7, yPortal[i] * 2 + 0.2, 'P', fontsize=20)

        if world == Constants.UPPER:
            [xBase, yBase] = np.where(map_world == Constants.BASE)
            [xFactory, yFactory] = np.where(map_world == Constants.FACTORY)
            if (xBase % 2 == 0 and yBase % 2 == 0) or (xBase % 2 != 0 and yBase % 2 != 0):
                ax.text(xBase[0] + 0.6, yBase[0] * 2 + 0.2, 'B', fontsize=20)
            else:
                ax.text(xBase[0] + 0.6, yBase[0] * 2 + 1.1, 'B', fontsize=20)

            if xFactory % 2 == 0 and yFactory % 2 == 0 or (xFactory % 2 != 0 and yFactory % 2 != 0):
                ax.text(xFactory[0] + 0.6, yFactory[0] * 2 + 0.2, 'F', fontsize=20)
            else:
                ax.text(xFactory[0] + 0.6, yFactory[0] * 2 + 1.1, 'F', fontsize=20)
        else:
            [xMine, yMine] = np.where(map_world == Constants.MINE)
            [xAlien, yAlien] = np.where(map_world == Constants.ALIEN)
            if (xMine % 2 == 0 and yMine % 2 == 0) or (xMine % 2 != 0 and yMine % 2 != 0):
                ax.text(xMine[0] + 0.6, yMine[0] * 2 + 1.1, 'M', fontsize=20)
            else:
                ax.text(xMine[0] + 0.6, yMine[0] * 2 + 0.2, 'M', fontsize=20)

            for i in range(0, len(xAlien)):
                if (xAlien[i] % 2 == 0 and yAlien[i] % 2 == 0) or (xAlien[i] % 2 != 0 and yAlien[i] % 2 != 0):
                    ax.text(xAlien[i] + 0.6, yAlien[i] * 2 + 1.1, 'A', fontsize=20)
                else:
                    ax.text(xAlien[i] + 0.6, yAlien[i] * 2 + 0.2, 'A', fontsize=20)

    # Plot (outer) boundaries
    mapSize = map_world.shape
    if world == Constants.UPPER:
        ax.plot(np.array([*range(0, mapSize[1] + 1)]) % 2,
                np.array([*range(0, 2 * mapSize[1] + 1, 2)]), 'k', linewidth=2)
        if mapSize[0] % 2 == 0:
            ax.plot([0, mapSize[0]], [0, 0], c='k', linewidth=2)
            ax.plot(np.array([*range(0, mapSize[1] + 1)]) % 2 + mapSize[0],
                    np.array([*range(0, 2 * mapSize[1] + 1, 2)]), 'k', linewidth=2)
            if mapSize[1] % 2 == 0:
                ax.plot([mapSize[0], 0], [2 * mapSize[1], 2 * mapSize[1]], 'k', linewidth=2)
            else:
                ax.plot([mapSize[0] + 1, 1], [2 * mapSize[1], 2 * mapSize[1]], 'k', linewidth=2)
        else:
            ax.plot([0, mapSize[0] + 1], [0, 0], 'k', linewidth=2)
            ax.plot(-(np.array([*range(0, mapSize[1] + 1)]) % 2) + mapSize[0] + 1,
                    np.array([*range(0, 2 * mapSize[1] + 2, 2)]),
                    'k', linewidth=2)
            if mapSize[1] % 2 == 0:
                ax.plot([mapSize[0] + 1, 0], [2 * mapSize[1], 2 * mapSize[1]], 'k', linewidth=2)
            else:
                ax.plot([mapSize[0], 1], [2 * mapSize[1], 2 * mapSize[1]], 'k', linewidth=2)
    else:
        ax.plot((-(np.array([*range(0, mapSize[1] + 1)]) % 2) + 1),
                np.array([*range(0, 2 * mapSize[1] + 2, 2)]), 'k', linewidth=2)

        if mapSize[0] % 2 == 0:
            ax.plot([1, mapSize[0] + 1], [0, 0], 'k', linewidth=2)
            ax.plot((-(np.array([*range(0, mapSize[1] + 1)]) % 2)) + mapSize[0] + 1,
                    np.array([*range(0, 2 * mapSize[1] + 2, 2)]), 'k', linewidth=2)
            if mapSize[1] % 2 == 0:
                ax.plot([mapSize[0] + 1, 1], [2 * mapSize[1], 2 * mapSize[1]], 'k', linewidth=2)
            else:
                ax.plot([mapSize[0], 0], [2 * mapSize[1], 2 * mapSize[1]], 'k', linewidth=2)
        else:
            ax.plot([1, mapSize[0]], [0, 0], 'k', linewidth=2)
            ax.plot(np.array([*range(0, mapSize[1] + 1)]) % 2 + mapSize[0],
                    np.array([*range(0, 2 * mapSize[1] + 2, 2)]), 'k', linewidth=2)
            if mapSize[1] % 2 == 0:
                ax.plot([mapSize[0], 1], [2 * mapSize[1], 2 * mapSize[1]], 'k', linewidth=2)
            else:
                ax.plot([mapSize[0] + 1, 0], [2 * mapSize[1], 2 * mapSize[1]], 'k', linewidth=2)

    # Set proper indices on axes
    ax.set_xticks(range(1, mapSize[0] + 1), range(0, mapSize[0]))
    ax.set_yticks(range(1, 2*mapSize[1]+1, 2), range(0, mapSize[1]))

    # set aspect ratio to 1
    ratio = 1.0
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)


def getUpperTriangle(column, row):
    if column % 2 == row % 2:
        xCorner = [column, column + 1, column + 2]
        yCorner = [2 * row, 2 * row + 2, 2 * row]
    else:
        xCorner = [column, column + 1, column + 2]
        yCorner = [2 * row + 2, 2 * row, 2 * row + 2]
    return [xCorner, yCorner]


def getLowerTriangle(column, row):
    if column % 2 != row % 2:
        xCorner = [column, column + 1, column + 2]
        yCorner = [2 * row, 2 * row + 2, 2 * row]
    else:
        xCorner = [column, column + 1, column + 2]
        yCorner = [2 * row + 2, 2 * row, 2 * row + 2]
    return [xCorner, yCorner]


def arrow(startPt, endPt, ax):
    color = [100 / 255, 100 / 255, 100 / 255]
    # color = np.array([0, 0, 0]) + 0.8

    # If policy is stay, don't plot arrow
    if endPt[1] == startPt[1] and endPt[0] == startPt[0]:
        # stay - plot point
        ax.plot(endPt[0], endPt[1], marker="o", markersize=2, markeredgecolor=color, markerfacecolor=color)
    else:
        # Compute orientation if arrow head
        alpha = np.arctan2(endPt[1] - startPt[1], endPt[0] - startPt[0])
        R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])

        # define lines that draw the arrow head
        arrowHead = np.array([[0, 0],
                              [-0.1, 0.1],
                              [0, 0],
                              [-0.1, -0.1]])
        for i in range(0, arrowHead.shape[0]):
            arrowHead[i, :] = np.transpose(np.matmul(R, np.transpose(arrowHead[i, :])))
            arrowHead[i, :] = arrowHead[i, :] + endPt

        # define line that draws the arrow
        arrowLines = np.array([[startPt[0], startPt[1]],
                               [endPt[0], endPt[1]]])
        # plot
        ax.plot(arrowLines[:, 0], arrowLines[:, 1], color=color, linewidth=1.5)
        ax.plot(arrowHead[:, 0], arrowHead[:, 1], color=color, linewidth=1.5)
