#!/usr/bin/env python

import rospy
import numpy as np
import math as m
import random as r
import helper_functions as hf

from read_config import read_config

from nav_msgs.msg import OccupancyGrid, MapMetaData
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray

from map_utils import Map

class Particle():
    def __init__(self,pose,t,w):
        self.pose = pose
        self.theta = t
        self.weight = w

class Robot():
    def __init__(self):
        rospy.init_node('robot')
        self.config = read_config()
        self.size = self.config["num_particles"]
        r.seed(self.config["seed"])

        self.particle_array = [] 
        self.pose_array = PoseArray() 
        self.pose_array.header.stamp = rospy.Time.now()
        self.pose_array.header.frame_id = 'map'
        self.pose_array.poses = []

        rospy.Subscriber("/map", OccupancyGrid, self.handle_mapserver) 
        rospy.Subscriber("base_scan_with_error", LaserScan, self.handle_scan)
        self.particle_pub = rospy.Publisher("/particlecloud", PoseArray, queue_size = 10)

        self.rate = rospy.Rate(1)
        rospy.spin()

    def handle_scan(self, resp):
        poses = self.pose_array.poses
        for i in xrange(self.size):
            if(poses[i].position.x < self.width - 1):
                poses[i].position.x += 1
            else:
                poses[i].position.x = 0

        self.pose_array.poses = poses
        self.particle_pub.publish(self.pose_array)

    def handle_mapserver(self, resp):
        self.map = Map(resp)

        self.width = self.map.width
        self.height = self.map.height

        for i in xrange(self.size):
            x = r.randint(0,self.width)
            y = r.randint(0,self.height)
            t = r.random()*m.pi*2
            w = 1./self.size
            pose = hf.get_pose(x,y,t)
            self.pose_array.poses.append(pose)
            self.particle_array.append(Particle(pose,t,w))


if __name__ == '__main__':
    rb = Robot()
