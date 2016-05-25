# astar implementation needs to go here

import numpy as np
import math as m
import rospy
import heapq

from cse_190_assi_3.msg import AStarPath
from copy import deepcopy

from read_config import read_config

def solve():
    config = read_config()
    
    row_size = config['map_size'][0]
    col_size = config['map_size'][1]

    forward_cost_map = np.zeros((row_size, col_size))
    walls = config['walls']
    pits = config['pits']

    start = config['start']
    goal = config['goal']

    move_list = config['move_list']

    ###############
    # Set up grid #
    ###############
    for wall in walls:
        forward_cost_map[wall[0]][wall[1]] = -1000

    for pit in pits:
        forward_cost_map[pit[0]][pit[1]] = -1000

    for x in range(0, row_size):
        for y in range(0, col_size):
            if forward_cost_map[x][y] == -1000:
                continue
            forward_cost_map[x][y] = abs(goal[0]-x) + abs(goal[1]-y)

    ############
    # Solve A* #
    ############
    move_count = 0
    move_queue = []
    prev_map = [[[-1,-1] for i in xrange(col_size)] for j in xrange(row_size)]
    heapq.heappush(move_queue, (0, 0, start, []))

    while len(move_queue) > 0:
        move_set = heapq.heappop(move_queue)
        backward_cost = move_set[1] + 1
        cur_pos = move_set[2]
        path = deepcopy(move_set[3])
        path.append(cur_pos)
        if(cur_pos == goal):
            return path
        x = cur_pos[0]
        y = cur_pos[1]
        for move in move_list:
            new_x = x + move[0]
            new_y = y + move[1]
            if(new_x < 0 or new_x >= row_size or new_y < 0 or new_y >= col_size):
                continue
            if(forward_cost_map[new_x][new_y] == -1000):
                continue
            cost = forward_cost_map[new_x][new_y] + backward_cost
            prev_map[new_x][new_y] = cur_pos
            heapq.heappush(move_queue, (cost, backward_cost, [new_x, new_y], path))
        
    return [] 
