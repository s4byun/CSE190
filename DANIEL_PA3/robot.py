#!/usr/bin/env python

import rospy
import numpy as np
import math as m
import random as r

from read_config import read_config
from std_msgs.msg import Bool
from cse_190_assi_3.msg import AStarPath

import astar

class Robot():
    def __init__(self):
        rospy.init_node('robot')
        self.astar_pub = rospy.Publisher(
                "/results/path_list",
                AStarPath,
                queue_size = 10
        )
        self.simulation_complete_pub = rospy.Publisher(
                "/map_node/sim_complete",
                Bool,
                queue_size = 1
        )
        self.config = read_config()
        rospy.sleep(1)
        self.perform_astar()
        self.simulation_complete_pub.publish(True)
        rospy.sleep(1)
        rospy.signal_shutdown("Done.")

    def perform_astar(self):
        astar_result = astar.solve()
        for result in astar_result:
            self.astar_pub.publish(result)
        
if __name__ == '__main__':
    rb = Robot()
