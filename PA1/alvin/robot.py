#!/usr/bin/env python

import rospy
import numpy as np
from copy import deepcopy
from math import sqrt, pi, exp
from std_msgs.msg import Bool, String, Float32
from cse_190_assi_1.msg import temperatureMessage, RobotProbabilities
from cse_190_assi_1.srv import requestMapData, moveService, requestTexture
from read_config import read_config

def norm(mu, sigma, x):
    return (1./(sigma*sqrt(2*pi)))*exp(-(x-mu)**2/(2*sigma**2))

class Robot():

    def __init__(self):
        rospy.init_node('robot')
        self.bootstrap()
        self.subscribe()
        self.publish()

    def bootstrap(self):
        self.config      = read_config()
        self.texture_map = [item for sublist in \
                self.config["texture_map"] for item in sublist]

        self.pipe_map    = [item for sublist in \
                self.config["pipe_map"] for item in sublist]

        self.rows        = len(self.config["texture_map"])
        self.cols        = len(self.config["texture_map"][0])
        self.grid_size   = self.rows*self.cols

        self.temp_dict   = {
                'H': 40.0,
                'C': 20.0,
                '-': 25.0
        }

        self.prob_move_correct   = self.config["prob_move_correct"]
        self.prob_move_incorrect = (1. - self.prob_move_correct) / 4.

        self.prob_tex_correct    = self.config["prob_tex_correct"]
        self.prob_tex_incorrect  = 1. - self.prob_tex_correct

        self.temp_std_dev        = self.config["temp_noise_std_dev"]

        self.move_list           = self.config["move_list"]
        self.move_num            = 0
        self.move_max            = len(self.move_list)

        self.belief              = np.full(self.grid_size, 1./self.grid_size)

    def publish(self):
        self.activation_pub = rospy.Publisher(
                "/temp_sensor/activation",
                Bool,
                queue_size = 1
        )
        self.temperature_pub = rospy.Publisher(
                "/results/temperature_data",
                Float32,
                queue_size = 10
        )
        self.texture_pub = rospy.Publisher(
                "/results/texture_data",
                String,
                queue_size = 10
        )
        self.probabilities_pub = rospy.Publisher(
                "/results/probabilities",
                RobotProbabilities,
                queue_size = 10
        )
        self.sim_complete = rospy.Publisher(
                "/map_node/sim_complete",
                Bool,
                queue_size = 1
        )

    def subscribe(self):
        self.temperature_sub = rospy.Subscriber(
                "/temp_sensor/data",
                temperatureMessage,
                self.handle_temperature_data
        )

    def handle_temperature_data(self, message):
        texture_requester = \
                rospy.ServiceProxy('requestTexture', requestTexture)
        move_requester    = \
                rospy.ServiceProxy('moveService', moveService)

        tex = texture_requester()

        print "temp:\t", message.temperature
        print "tex :\t", tex.data

        self.texture_pub.publish(tex.data)
        self.temperature_pub.publish(message.temperature)

        self.discrete_bayes_filter(0, tex.data)
        self.discrete_bayes_filter(1, message.temperature)

        if self.move_num == self.move_max:
            self.probabilities_pub.publish(self.belief)
            print "move:\tnone"
            print self.belief, "\n"
            self.sim_complete.publish(True)
            rospy.sleep(1)
            rospy.signal_shutdown("Done.")
        else:
            self.discrete_bayes_filter(2, self.move_list[self.move_num])
            self.probabilities_pub.publish(self.belief)
            print "move:\t", self.move_list[self.move_num]
            print self.belief, "\n"
            move_requester(self.move_list[self.move_num])
            self.move_num += 1

    def activate_temperature(self, isActive):
        self.activation_pub.publish(isActive)

    def discrete_bayes_filter(self, t, z):
        if   t == 0:
            self.perceptual_tex(z)
        elif t == 1:
            self.perceptual_temp(z)
        elif t == 2:
            self.action(z)
        else:
            print "invalid operation"

    def perceptual_tex(self, z):
        for i in range(self.grid_size):
            if self.texture_map[i] == z:
                self.belief[i] *= self.prob_tex_correct
            else:
                self.belief[i] *= self.prob_tex_incorrect
        normalizer = np.sum(self.belief)
        self.belief /= normalizer

    def perceptual_temp(self, z):
        for i in range(self.grid_size):
            normalizer = 0.
            pdf = norm(self.temp_dict[self.pipe_map[i]], \
                    self.temp_std_dev, z)
            normalizer = norm(self.temp_dict["-"], \
                    self.temp_std_dev, z) + \
                    norm(self.temp_dict["C"], self.temp_std_dev, z) + \
                    norm(self.temp_dict["H"], self.temp_std_dev, z)

            self.belief[i] *= pdf / normalizer

        self.belief /= np.sum(self.belief)

    def action(self, u):
        q = deepcopy(self.belief)
        for i in range(self.grid_size):

            top   = (i - self.cols) % self.grid_size
            bot   = (i + self.cols) % self.grid_size
            left  = (i - 1) % self.cols + self.cols * (i / self.cols)
            right = (i + 1) % self.cols + self.cols * (i / self.cols)

            if u == [0, 0]:
                self.belief[i] = q[i] * self.prob_move_correct + \
                        self.prob_move_incorrect * \
                        (q[top] +  q[bot] + q[left] + q[right])
            elif u == [1, 0]:
                self.belief[i] = q[top] * self.prob_move_correct + \
                        self.prob_move_incorrect * \
                        (q[i]   +  q[bot] + q[left] + q[right])
            elif u == [-1, 0]:
                self.belief[i] = q[bot] * self.prob_move_correct + \
                        self.prob_move_incorrect * \
                        (q[top] +  q[i]   + q[left] + q[right])
            elif u == [0, 1]:
                self.belief[i] = q[left] * self.prob_move_correct + \
                        self.prob_move_incorrect * \
                        (q[top] +  q[bot] + q[i]    + q[right])
            elif u == [0, -1]:
                self.belief[i] = q[right] * self.prob_move_correct + \
                        self.prob_move_incorrect * \
                        (q[top] +  q[bot] + q[left] + q[i])
            else:
                self.belief[i] = q[i]

if __name__ == '__main__':
    robot = Robot()
    rospy.sleep(1)
    robot.activate_temperature(True)
    rospy.spin()
