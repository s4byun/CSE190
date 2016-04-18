#!/usr/bin/env python

import rospy
import time
import numpy as np
import copy
from math import sqrt
from math import pi
from math import exp
from std_msgs.msg import Bool, String, Float32
from cse_190_assi_1.msg import temperatureMessage, RobotProbabilities
from cse_190_assi_1.srv import requestMapData, moveService, requestTexture
from read_config import read_config

class Robot():
    def __init__(self):
        rospy.init_node('robot')
        self.config = read_config()
        self.texture_map = [item for sublist in self.config["texture_map"] for item in sublist]
        self.pipe_map = [item for sublist in self.config["pipe_map"] for item in sublist]
        self.move_list = self.config["move_list"]
        self.temp_dict = {
                'H': 40.0,
                'C': 20.0,
                '-': 25.0
        }
        self.temp_std_dev = self.config["temp_noise_std_dev"]
        self.move_num = 0
        self.move_max = len(self.move_list)
        self.activation_pub = rospy.Publisher(
                "/temp_sensor/activation",
                Bool,
                queue_size = 10
        )
        self.temperature_sub = rospy.Subscriber(
                "/temp_sensor/data",
                temperatureMessage,
                self.handle_temperature_data
        )
        self.temperature_pub = rospy.Publisher(
                "/results/temperature_data", Float32, queue_size = 10)
        self.texture_pub = rospy.Publisher(
                "/results/texture_data", String, queue_size = 10)
        self.probabilities_pub = rospy.Publisher(
                "/results/probabilities", RobotProbabilities, queue_size = 10)
        self.sim_complete = rospy.Publisher(
                "/map_node/sim_complete", Bool, queue_size=1)
        self.belief = np.full(20, 1./20.)

    def handle_temperature_data(self, message):
        print message
        texture_requester = rospy.ServiceProxy('requestTexture', requestTexture)
        move_requester = rospy.ServiceProxy('moveService', moveService)
        tex = texture_requester()
        print tex
        self.perceptual_tex(tex.data)
        self.texture_pub.publish(tex.data)
        self.perceptual_temp(message.temperature)
        self.temperature_pub.publish(message.temperature)
        if self.move_num == self.move_max:
            self.probabilities_pub.publish(self.belief)
            self.sim_complete.publish(True)
            #rospy.spin()
            rospy.sleep(1)
            rospy.signal_shutdown("Done.")
            #self.activate_temperature(False)
            #rospy.spin()
            print self.belief
            print np.sum(self.belief)
        self.action(self.move_list[self.move_num])
        print self.belief
        print np.sum(self.belief)
        self.probabilities_pub.publish(self.belief)
        #rospy.spin()
        mov = move_requester(self.move_list[self.move_num])
        print mov
        self.move_num += 1
        rospy.sleep(1)
        #print mov

    def activate_temperature(self, isActive):
        self.activation_pub.publish(isActive)
        rospy.spin()

    def perceptual_tex(self, z):
        for i in range(len(self.belief)):
            if self.texture_map[i] == z:
                self.belief[i] *= self.config["prob_tex_correct"]
            else:
                self.belief[i] *= 1. - self.config["prob_tex_correct"]
        normalizer = np.sum(self.belief)
        self.belief /= normalizer
        #print self.belief
        #print np.sum(self.belief)

    def norm(self, mu, sigma, x):
        return (1./(sigma*sqrt(2*pi)))*exp(-(x-mu)**2/(2*sigma**2))

    def perceptual_temp(self, z):
        for i in range(len(self.belief)):
            normalizer = 0.
            pdf = self.norm(self.temp_dict[self.pipe_map[i]], \
                self.temp_std_dev, z)
            normalizer = self.norm(self.temp_dict["-"], self.temp_std_dev, z) \
                        +self.norm(self.temp_dict["C"], self.temp_std_dev, z) \
                        +self.norm(self.temp_dict["H"], self.temp_std_dev, z)
            self.belief[i] *= (pdf/normalizer)
        self.belief /= np.sum(self.belief)
        #print self.belief
        #print np.sum(self.belief)

    def action(self, u):
        q = copy.deepcopy(self.belief)
        for i in range(len(self.belief)):
            if u == [0, 0]:
                self.belief[i] = q[i]*(self.config["prob_move_correct"]) + ((1. - self.config["prob_move_correct"])/4.)*(q[(i-5)%20] + \
                        q[(i+5)%20] + q[(i+1)%5+5*(i/5)] + \
                        q[(i-1)%5+5*(i/5)])
            elif u == [1, 0]:
                self.belief[i] = q[(i-5)%20]*(self.config["prob_move_correct"]) + ((1. - self.config["prob_move_correct"])/4.)*(q[i] + \
                        q[(i+5)%20] + q[(i+1)%5+5*(i/5)] + \
                        q[(i-1)%5+5*(i/5)])
            elif u == [-1, 0]:
                self.belief[i] = q[(i+5)%20]*(self.config["prob_move_correct"]) + ((1. - self.config["prob_move_correct"])/4.)*(q[(i-5)%20] + \
                        q[i] + q[(i+1)%5+5*(i/5)] + \
                        q[(i-1)%5+5*(i/5)])
            elif u == [0, 1]:
                self.belief[i] = q[(i-1)%5+5*(i/5)]*(self.config["prob_move_correct"]) + ((1. - self.config["prob_move_correct"])/4.)*(q[(i-5)%20] + \
                        q[(i+5)%20] + q[(i+1)%5+5*(i/5)] + \
                        q[i])
            elif u == [0, -1]:
                self.belief[i] = q[(i+1)%5+5*(i/5)]*(self.config["prob_move_correct"]) + ((1. - self.config["prob_move_correct"])/4.)*(q[(i-5)%20] + \
                        q[(i+5)%20] + q[i] + \
                        q[(i-1)%5+5*(i/5)])
            else:
                self.belief[i] = q[i]

if __name__ == '__main__':
    robot = Robot()
    rospy.sleep(2)
    robot.activate_temperature(True)
#    robot.loop()
#    robot.activation_pub.publish(True)
#    rospy.spin()
