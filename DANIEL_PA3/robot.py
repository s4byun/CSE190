#!/usr/bin/env python

import rospy
import numpy as np
import math as m
import random as r

from read_config import read_config
import astar

class Robot():
    def __init__(self):
        rospy.init_node('robot')
        self.config = read_config()
        
        astar_result = astar.solve()
        print astar_result 
        rospy.spin() 

if __name__ == '__main__':
    rb = Robot()
