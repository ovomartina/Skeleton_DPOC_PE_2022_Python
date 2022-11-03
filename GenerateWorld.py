from random import randrange
import numpy as np
from Constants import *


def GenerateWorld(width, height):
    """
    	@width:
                   Integer describing the width of the map, M.

               height:
                   Integer describing the length of the map, N.

    	Output arguments:

        	map:
               A (M x N) matrix describing the terrain of the map. map(m,n)
               represents the cell at indices (m,n) according to the axes
               specified in the PDF.
   """

    # check input feasibility
    if not isinstance(width, int) or not isinstance(height, int):
        print('Warning: Width or height not an integer!')
        width = int(width)
        height = int(height)
        print(f'New width: {width}')
        print(f'New height: {height}')

    if width < 8 or height < 8:
        print('Error: Minimum width and height is 8!')
        print('Exiting function')
        return

    # Generate map

    # obstacle parameters
    obstacleDensity = 6 / 100  # 8 obstacle groups per 100 cells
    obstacleScalingWidth = 1.0
    obstacleScalingHeight = 0.4

    # Portal parameters
    portalDensity = 1 / 100  # 1 portal per 100 cells

    #  alien parameters
    alienDensity = 3 / 100  # 3 aliens per 100 cells

    feasible = False
    while not feasible:
        map = np.zeros((width, height))
        # add obstacles
        for k in range(round(obstacleDensity * width * height)):
            # pick center of obstacle group
            obstacleCenter = np.array([round(np.random.uniform() * (width - 1)),
                                       round(np.random.uniform() * (height - 1))])
            # choose group size
            obstacleSize = np.array([round(abs(np.random.normal()) * obstacleScalingWidth + 1),
                                     round(abs(np.random.normal()) * obstacleScalingHeight + 1)])
            # insert building into map
            mLow = max(0, round(obstacleCenter[0] - obstacleSize[0] / 2))
            mHigh = min(
                width - 1, round(obstacleCenter[0] + obstacleSize[0] / 2))
            nLow = max(0, round(obstacleCenter[1] - obstacleSize[1] / 2))
            nHigh = min(
                height - 1, round(obstacleCenter[1] + obstacleSize[1] / 2))

            map[mLow:mHigh, nLow:nHigh] = Constants.OBSTACLE

        # add portals
        for k in range(round(portalDensity * width * height)):
            while True:
                portal = np.array([round(np.random.uniform() * (width - 1)),
                                   round(np.random.uniform() * (height - 1))])
                # check if portal is in a free spot
                if map[portal[0], portal[1]] == Constants.FREE:
                    break

            map[portal[0], portal[1]] = Constants.PORTAL

        # add aliens
        for k in range(round(alienDensity * width * height)):
            while True:
                alien = np.array([round(np.random.uniform() * (width - 1)),
                                  round(np.random.uniform() * (height - 1))])
                # check if alien is in a free spot
                if map[alien[0], alien[1]] == Constants.FREE:
                    break

            map[alien[0], alien[1]] = Constants.ALIEN

        # add mine
        while True:
            mine = np.array([round(np.random.uniform() * (width - 1)),
                             round(np.random.uniform() * (height - 1))])
            # check if mine is in a free spot
            if map[mine[0], mine[1]] == Constants.FREE:
                break

        map[mine[0], mine[1]] = Constants.MINE

        # add factory
        while True:
            factory = np.array([round(np.random.uniform() * (width - 1)),
                                round(np.random.uniform() * (height - 1))])
            # check if drop_off is in a free spot
            if map[factory[0], factory[1]] == Constants.FREE:
                break

        map[factory[0], factory[1]] = Constants.FACTORY

        # add base
        while True:
            base = np.array([round(np.random.uniform() * (width - 1)),
                             round(np.random.uniform() * (height - 1))])

            # stick the base on the edges of the map
            r = randrange(4)
            match r:
                case 0:
                    base[0] = 0
                case 1:
                    base[1] = 0
                case 2:
                    base[0] = width - 1
                case 3:
                    base[1] = height - 1

            # check if drop_off is in a free spot
            if map[base[0], base[1]] == Constants.FREE:
                break

        map[base[0], base[1]] = Constants.BASE

        # make map feasible: check that their exists a path from every
        # non-building cell to the pizzeria and drop-off points
        # feasible = check_map(map, pizzeria) and check_map(map, drop_off)  # Why check both? useless?
        feasible = check_map(map, factory)
    return map


def check_map(map, start):
    feasible_normal = False
    feasible_inverted = False

    # Check normal map
    stack = [start]
    visited_cells = np.zeros_like(map)
    visited_cells[start[0], start[1]] = 1
    while stack:
        current_cell = stack.pop()
        unvisited_neighbors = find_unvisited_neighbors_normal(
            current_cell, visited_cells)
        for i in range(len(unvisited_neighbors)):
            cell = unvisited_neighbors[i]
            if map[cell[0], cell[1]] == Constants.OBSTACLE:
                continue
            visited_cells[cell[0], cell[1]] = 1
            stack.append(cell)

    if visited_cells.sum() == (map != Constants.OBSTACLE).sum():
        feasible_normal = True

    # Check inverted map
    stack = [start]
    visited_cells = np.zeros_like(map)
    visited_cells[start[0], start[1]] = 1
    while stack:
        current_cell = stack.pop()
        unvisited_neighbors = find_unvisited_neighbors_inverted(
            current_cell, visited_cells)
        for i in range(len(unvisited_neighbors)):
            cell = unvisited_neighbors[i]
            if map[cell[0], cell[1]] == Constants.OBSTACLE:
                continue
            visited_cells[cell[0], cell[1]] = 1
            stack.append(cell)
    if visited_cells.sum() == (map != Constants.OBSTACLE).sum():
        feasible_inverted = True
    return feasible_normal and feasible_inverted


def find_unvisited_neighbors_normal(current_cell, visited_cells):
    unvisited_neighbors = []
    m = current_cell[0]
    n = current_cell[1]

    # above
    if m % 2 == n % 2:
        if n - 1 >= 0:
            if visited_cells[m, n - 1] == 0:
                unvisited_neighbors.append([m, n - 1])

    # below
    if m % 2 != n % 2:
        if n + 1 < visited_cells.shape[1]:
            if visited_cells[m, n + 1] == 0:
                unvisited_neighbors.append([m, n + 1])

    # left
    if m - 1 >= 0:
        if visited_cells[m - 1, n] == 0:
            unvisited_neighbors.append([m - 1, n])

    # right
    if m + 1 < visited_cells.shape[0]:
        if visited_cells[m + 1, n] == 0:
            unvisited_neighbors.append([m + 1, n])

    return unvisited_neighbors


def find_unvisited_neighbors_inverted(current_cell, visited_cells):
    unvisited_neighbors = []
    m = current_cell[0]
    n = current_cell[1]

    # above
    if m % 2 != n % 2:
        if n - 1 >= 0:
            if visited_cells[m, n - 1] == 0:
                unvisited_neighbors.append([m, n - 1])

    # below
    if m % 2 == n % 2:
        if n + 1 < visited_cells.shape[1]:
            if visited_cells[m, n + 1] == 0:
                unvisited_neighbors.append([m, n + 1])

    # left
    if m - 1 >= 0:
        if visited_cells[m - 1, n] == 0:
            unvisited_neighbors.append([m - 1, n])

    # right
    if m + 1 < visited_cells.shape[0]:
        if visited_cells[m + 1, n] == 0:
            unvisited_neighbors.append([m + 1, n])

    return unvisited_neighbors
