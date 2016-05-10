#!/usr/bin/env python

import rospy
import numpy as np
import math as m
import random as r
import helper_functions as hf

from read_config import read_config

from nav_msgs.msg import OccupancyGrid, MapMetaData
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Twist
from sklearn.neighbors import KDTree
from map_utils import Map

class Particle():
    def __init__(self,x,y,t,w):
        self.x = x
        self.y = y
        self.theta = t
        self.weight = w

class Robot():
    def __init__(self):
        rospy.init_node('robot')
        self.config = read_config()
        r.seed(self.config["seed"])

        self.rate = rospy.Rate(1)
        self.num_moves = 0
        self.moved = False

        self.bootstrap()
        rospy.spin()

    def bootstrap(self):
        rospy.Subscriber("/map", OccupancyGrid, self.handle_mapserver) 
        rospy.Subscriber("base_scan_with_error", LaserScan, self.handle_scan)
        self.particle_pub = rospy.Publisher("/particlecloud", PoseArray, queue_size = 10)
        self.likelihood_pub = rospy.Publisher("/likelihood_field", OccupancyGrid, queue_size = 10, latch = True) 
        

    def update_likelihood(self):
        obstacle_loc = []
        for x in xrange(self.width):
            for y in xrange(self.height):
                if self.map.get_cell(x,y) == 1.0:
                    obstacle_loc.append([x,y])

        kdt = KDTree(obstacle_loc)

        for x in xrange(self.width):
            for y in xrange(self.height):
                dist, index = kdt.query([x,y], k = 1)
                self.map.set_cell(x, y, self.normpdf(0, self.config["laser_sigma_hit"], dist[0])) 
        
        self.likelihood_pub.publish(self.map.to_message())


    def initialize_particles(self):
        self.size = self.config["num_particles"]
        self.particles = [] 
        self.pose_array = PoseArray() 
        self.pose_array.header.stamp = rospy.Time.now()
        self.pose_array.header.frame_id = 'map'
        self.pose_array.poses = []

        for i in xrange(self.size):
            x = r.randint(0,self.width)
            y = r.randint(0,self.height)
            t = r.random()*m.pi*2
            w = 1./self.size
            pose = hf.get_pose(x,y,t)
            self.pose_array.poses.append(pose)
            self.particles.append(Particle(x,y,t,w))

        self.particle_pub.publish(self.pose_array)

    def first_move(self):
        move_list = self.config["move_list"][0]
        a = move_list[0] * m.pi / 180.0
        d = move_list[1]
        n = move_list[2]
        hf.move_function(a, 0)
        p = self.particles
        for i in xrange(self.size):
            p[i].theta += a + r.gauss(0, self.config["first_move_sigma_angle"])
            if(p[i].theta >= 2*m.pi):
                p[i].theta -= 2*m.pi
            elif(p[i].theta < 0):
                p[i].theta += 2*m.pi

        for _ in xrange(n):
            hf.move_function(0,d)
            for i in xrange(self.size):
                dx = d*m.cos(p[i].theta)+r.gauss(0,self.config["first_move_sigma_x"])
                dy = d*m.sin(p[i].theta)+r.gauss(0,self.config["first_move_sigma_y"])
                p[i].x += dx
                p[i].y += dy
                self.pose_array.poses[i] = hf.get_pose(p[i].x, p[i].y, p[i].theta)

        self.particles = p 
        self.particle_pub.publish(self.pose_array)
        self.moved = True
        self.num_moves += 1
    
    def handle_scan(self, resp):
        if self.moved == True:
            self.moved = False


    def handle_mapserver(self, resp):
        self.map = Map(resp)

        self.width = self.map.width
        self.height = self.map.height

        self.update_likelihood()
        self.initialize_particles()
        self.first_move()
       
        
    def normpdf(self, mean, sd, x):
        var = float(sd)**2
        denom = (2*m.pi*var)**.5
        num = m.exp(-(float(x)-float(mean))**2/(2*var))
        return num  #/denom

        
        

if __name__ == '__main__':
    rb = Robot()
