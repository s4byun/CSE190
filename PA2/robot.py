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
                val = self.normpdf(0, self.config["laser_sigma_hit"], dist[0])
                self.map.set_cell(x, y, val)
        
        self.likelihood_pub.publish(self.map.to_message())


    def initialize_particles(self):
        self.size = self.config["num_particles"]
        self.particles = [] 
        poses = []
        
        for i in xrange(self.size):
            x = r.randint(0,self.width)
            y = r.randint(0,self.height)
            t = r.random()*2*m.pi 
            w = 1./self.size
            poses.append(hf.get_pose(x,y,t))
            self.particles.append(Particle(x,y,t,w))

        self.update_pose_and_publish(poses)

    def move(self):
        sdt = self.config["resample_sigma_angle"]
        sdx = self.config["resample_sigma_x"]
        sdy = self.config["resample_sigma_y"]
        if(self.num_moves == 0):
            sdt = self.config["first_move_sigma_angle"]
            sdx = self.config["first_move_sigma_x"]
            sdy = self.config["first_move_sigma_y"]

        move_list = self.config["move_list"][self.num_moves]
        a = move_list[0] * m.pi / 180.0
        d = move_list[1]
        n = move_list[2]
        hf.move_function(a, 0)
        p = self.particles

        for i in xrange(self.size):
            p[i].theta += r.gauss(a, sdt) 
            if(p[i].theta >= 2*m.pi):
                p[i].theta -= 2*m.pi
            elif(p[i].theta < 0):
                p[i].theta += 2*m.pi

        for step in xrange(n):
            hf.move_function(0,d)
            poses = []
            for i in xrange(self.size):
                dx = d*m.cos(p[i].theta)+r.gauss(0,sdx)
                dy = d*m.sin(p[i].theta)+r.gauss(0,sdy)
                p[i].x += dx
                p[i].y += dy
                poses.append(hf.get_pose(p[i].x, p[i].y, p[i].theta))
            self.update_pose_and_publish(poses)
                    
        self.particles = p 
        self.num_moves += 1
        self.moved = True
    
    def update_pose_and_publish(self,poses):
        pose_array = PoseArray() 
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = 'map'
        pose_array.poses = poses 
        self.particle_pub.publish(pose_array)

    def handle_scan(self, resp):
        if self.moved == True:
            self.moved = False
            p = self.particles
            total_weight = 0.0
            for i in xrange(self.size):
                Ptot = 0.0
                coord = self.map.get_cell(p[i].x, p[i].y)
                if(coord == 1.0 or coord != coord):
                    p[i].weight = 0.0
                else:
                    for d in resp.ranges:
                        x = p[i].x + d*m.cos(p[i].theta)
                        y = p[i].y + d*m.sin(p[i].theta)
                        Lp = self.map.get_cell(x,y)
                        pz = self.config["laser_z_hit"]*Lp + self.config["laser_z_rand"]
                        Ptot *= pz

                    p[i].weight *= Ptot
                    total_weight += p[i].weight

            p_accumulator = 0.0
            wheel = []
            for i in xrange(self.size):
                p[i].weight /= total_weight
                p_accumulator += p[i].weight
                wheel.append(p_accumulator)

            self.particles = p
            self.resample(wheel)
            self.move()

    def resample(self, wheel):
        p = self.particles
        poses = []
        for i in xrange(self.size):
            picker = r.random()
            for j in xrange(self.size):
                if wheel[j] >= picker:
                    self.particles[i] = p[j]
                    poses.append(hf.get_pose(p[j].x, p[j].y, p[j].theta))
                    break

        self.update_pose_and_publish(poses)

    def handle_mapserver(self, resp):
        self.map = Map(resp)

        self.width = self.map.width
        self.height = self.map.height

        self.update_likelihood()
        self.initialize_particles()
        self.move()
       
    def normpdf(self, mean, sd, x):
        var = float(sd)**2
        denom = (2*m.pi*var)**.5
        num = m.exp(-(float(x)-float(mean))**2/(2*var))
        return num

if __name__ == '__main__':
    rb = Robot()
