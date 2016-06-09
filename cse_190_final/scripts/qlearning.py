import heapq
import sys
import copy
import random
import bisect

from read_config import read_config

class Cell():
    def __init__(self, position, reward):
        self.position = position
        self.reward   = reward
        self.value    = 0.0
        self.pre      = 0.0
        self.q_values = {(0,1): 0.0, (0,-1): 0.0, (1,0): 0.0, (-1,0): 0.0}
        self.policy   = "NONE"

class Util():
    def __init__(self, utility):
        self.utility = utility
    def __cmp__(self, other):
        return -cmp(self.utility, other.utility)

def weighted_choice(distribution):
    moves, weights = zip(*distribution)
    total = 0.0
    cummulative_weights = []

    for weight in weights:
        total += weight
        cummulative_weights.append(total)

    r = random.uniform(0.0, total)
    return moves[bisect.bisect(cummulative_weights, r)]

def solve():

    random.seed(0)
    map_graph = {}

    config    = read_config()

    row_size  = config['map_size'][0]
    col_size  = config['map_size'][1]

    walls     = config['walls']
    pits      = config['pits']

    start     = tuple(config['start'])
    goal      = tuple(config['goal'])

    move_list = config['move_list']

    max_iterations       = config['max_iterations']
    threshold_difference = config["threshold_difference"]

    reward_for_each_step      = config["reward_for_each_step"]
    reward_for_hitting_wall   = config["reward_for_hitting_wall"]
    reward_for_reaching_goal  = config["reward_for_reaching_goal"]
    reward_for_falling_in_pit = config["reward_for_falling_in_pit"]

    discount_factor    = config["discount_factor"]
    learning_rate      = config["learning_rate"]
    learning_rate_fast = config["learning_rate_fast"]

    epsilon = config["epsilon"]

    prob_move_forward  = config["prob_move_forward"]
    prob_move_left     = config["prob_move_left"]
    prob_move_right    = config["prob_move_right"]
    prob_move_backward = config["prob_move_backward"]

    distribution = (("FORWARD", prob_move_forward), ("LEFT", prob_move_left), ("RIGHT", prob_move_right), ("BACKWARDS", prob_move_backward))

    ###############
    # Set up grid #
    ###############
    for x in range(0, row_size):
        for y in range(0, col_size):
            map_graph[(x, y)] = Cell((x, y), reward_for_each_step)

    for wall in walls:
        wall_cell        = map_graph[(wall[0], wall[1])]
        wall_cell.policy = "WALL"
        wall_cell.reward = reward_for_hitting_wall

    for pit in pits:
        pit_cell        = map_graph[(pit[0], pit[1])]
        pit_cell.policy = "PIT"
        pit_cell.reward = reward_for_falling_in_pit

    goal_cell        = map_graph[tuple(goal)]
    goal_cell.policy = "GOAL"
    goal_cell.reward = reward_for_reaching_goal

    policies = {(0,1):"E", (0,-1):"W", (1,0):"S", (-1,0):"N"}

    ####################
    # Solve Q-Learning #
    ####################
    iteration = 0

    list_policies = []

    curr_cell = map_graph[start]

    # start x, y
    x = start[0]
    y = start[1]

    while iteration < max_iterations:

        relative_cell = None
        action        = None

        forward    = None
        right      = None
        down       = None
        left       = None

        threshold  = 0.0
        utility    = 0.0
        alpha      = 0.0

        action_util = []
        curr_q      = []
        max_q       = []
        min_q       = []
        neighbours  = []
        policy      = []

        # if pit or goal, reset robot to the start
        if curr_cell.policy == "PIT" or curr_cell.policy == "GOAL":
            x = start[0]
            y = start[1]
            curr_cell = map_graph[start]

        stationary_cell        = copy.deepcopy(curr_cell)
        stationary_cell.reward = reward_for_hitting_wall

        # pre-compute neighbouring cells
        for move in move_list:
            new_x    = x + move[0]
            new_y    = y + move[1]

            # out of bounds
            if new_x < 0 or new_x >= row_size or new_y < 0 or new_y >= col_size:
                neighbours.append(stationary_cell)
                continue

            neighbour_cell = map_graph[(new_x, new_y)]

            # hit a wall
            if neighbour_cell.policy == "WALL":
                neighbours.append(stationary_cell)
            else:
                neighbours.append(neighbour_cell)

        # pick a random action based on epsilon
        if random.random() < epsilon:
            for m, q in curr_cell.q_values.iteritems():
                heapq.heappush(curr_q, (Util(q), m))
                heapq.heappush(min_q, (q, m))

            mag = max(abs(heapq.heappop(curr_q)[0].utility), abs(heapq.heappop(min_q)[0]))
            curr_q = []

            for m, q in curr_cell.q_values.iteritems():
                heapq.heappush(curr_q, (Util(q + (random.random() - .5)*mag), m))

            action = heapq.heappop(curr_q)[1]
        else:
            action = tuple(move_list[random.randint(0,3)])

        # movement resulted from uncertainty
        relative_action = weighted_choice(distribution)

        # Move EAST
        if action == (0,1):
            forward = neighbours[0]
            right   = neighbours[2]
            down    = neighbours[1]
            left    = neighbours[3]
        # Move WEST
        elif action == (0,-1):
            forward = neighbours[1]
            right   = neighbours[3]
            down    = neighbours[0]
            left    = neighbours[2]
        # Move SOUTH
        elif action == (1,0):
            forward = neighbours[2]
            right   = neighbours[1]
            down    = neighbours[3]
            left    = neighbours[0]
        # Move NORTH
        else:
            forward = neighbours[3]
            right   = neighbours[0]
            down    = neighbours[2]
            left    = neighbours[1]

        # set relative cell based on uncertainty
        if relative_action == "FORWARD":
            relative_cell = forward
        elif relative_action == "LEFT":
            relative_cell = left
        elif relative_action == "RIGHT":
            relative_cell = right
        else:
            relative_cell = down

        # disastrous result(s)
        if relative_cell.policy == "PIT" or relative_cell.policy == "WALL":
            alpha = learning_rate_fast
        # normal exploration
        else:
            alpha = learning_rate

        # learn new Q
        for move, q in relative_cell.q_values.iteritems():
            heapq.heappush(action_util, (Util(q), move))

        maxQ   = heapq.heappop(action_util)
        sample = alpha * (relative_cell.reward + discount_factor * maxQ[0].utility)
        curr_cell.q_values[action] *= (1.0 - alpha)
        curr_cell.q_values[action] += sample

        # update policy
        for move, q in curr_cell.q_values.iteritems():
            heapq.heappush(max_q, (Util(q), move))

        curr_cell.policy = policies[heapq.heappop(max_q)[1]]
        curr_cell        = relative_cell
        x                = curr_cell.position[0]
        y                = curr_cell.position[1]

        iteration += 1

        for _x in range(0, row_size):
            for _y in range(0, col_size):
                policy.append(map_graph[(_x, _y)].policy)

        list_policies.append(copy.deepcopy(policy))

    return list_policies
