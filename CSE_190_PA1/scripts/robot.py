#!/usr/bin/env python

import rospy
import numpy as np
import math as m
from copy import deepcopy
from cse_190_assi_1.srv import requestMapData, requestTexture, moveService 
from cse_190_assi_1.msg import temperatureMessage
from cse_190_assi_1.msg import RobotProbabilities
from read_config import read_config
from std_msgs.msg import String
from std_msgs.msg import Float32
from std_msgs.msg import Bool

class Robot():
    def __init__(self):
        self.config = read_config()
        rospy.init_node('robot')

        self.temp_service = rospy.Publisher("/temp_sensor/activation", Bool, queue_size = 10)
        rospy.Subscriber("/temp_sensor/data", temperatureMessage, self.temp_callback)

        self.texService = self.request_texture_service() 
        self.movService = self.request_move_service() 

        self.pos_out = rospy.Publisher("/results/probabilities", RobotProbabilities, queue_size = 10)
        self.temp_out = rospy.Publisher("/results/temperature_data", Float32, queue_size = 10)
        self.tex_out = rospy.Publisher("/results/texture_data", String, queue_size = 10)
        self.sim_complete = rospy.Publisher("/map_node/sim_complete", Bool, queue_size = 1)
        
        self.num_rows = len(self.config['pipe_map'])
        self.num_cols = len(self.config['pipe_map'][0])
        self.belief_p = np.full(self.num_rows * self.num_cols, 1./(self.num_rows*self.num_cols))
        self.belief = np.full(self.num_rows * self.num_cols, 1./(self.num_rows*self.num_cols))

        self.move_count = 0 
        self.total_move_count = len(self.config['move_list'])
        self.p_move = self.config['prob_move_correct']

        self.rate = rospy.Rate(1)

        self.loop()
        rospy.rate.sleep()
        rospy.signal_shutdown("simulation complete")

    def temp_callback(self, data):

        tex = self.texService().data
        temp = data.temperature
        sd = self.config['temp_noise_std_dev']
        temp_dict = {'H':40.0, '-':25.0, 'C':20.0}


        for r in xrange(0,self.num_rows):
            for c in xrange(0,self.num_cols):
                index = r*self.num_cols + c
                    
                p_temp_x = self.normpdf(temp_dict[self.config['pipe_map'][r][c]], sd, temp)
                p_tex_x = 0.0

                if tex == self.config['texture_map'][r][c]:
                    p_tex_x = self.config['prob_tex_correct']
                else:
                    p_tex_x = 1. - self.config['prob_tex_correct']
                
                self.belief_p[index] = p_temp_x * p_tex_x * self.belief[index]

        K = sum(self.belief_p)
        self.belief_p[:] = [x / K for x in self.belief_p]

        if self.move_count < self.total_move_count:
            for r in xrange(0,self.num_rows):
                for c in xrange(0,self.num_cols):
                    index = r*self.num_cols + c
                    move = self.config['move_list'][self.move_count]
                    inc_move_list = deepcopy(self.config['possible_moves'])
                    inc_move_list.remove(move)

                    c_index = ((r-move[0])%self.num_rows)*self.num_cols + (c-move[1])%self.num_cols

                    self.belief[index] = self.p_move * self.belief_p[c_index]

                    for x in inc_move_list:
                        c_index = ((r-x[0])%self.num_rows)*self.num_cols + (c-x[1])%self.num_cols
                        self.belief[index] += ((1.0-self.p_move)/4.0)*self.belief_p[c_index]

            mov = self.movService(self.config['move_list'][self.move_count])
            self.temp_out.publish(temp)
            self.tex_out.publish(tex)
            self.pos_out.publish(self.belief)

        elif self.move_count == self.total_move_count:
            self.temp_out.publish(temp)
            self.tex_out.publish(tex)
            self.pos_out.publish(self.belief_p)
            self.temp_service.publish(False)
            self.sim_complete.publish(True)

        self.move_count += 1


    def request_texture_service(self):
        rospy.wait_for_service("requestTexture")
        return rospy.ServiceProxy("requestTexture", requestTexture)

    def request_move_service(self):
        rospy.wait_for_service("requestTexture")
        return rospy.ServiceProxy("moveService", moveService)

    def loop(self):
        while not rospy.is_shutdown() and self.move_count <= self.total_move_count:
            self.temp_service.publish(True)
            self.rate.sleep()
    
    def normpdf(self, mean, sd, x):
        var = float(sd)**2
        denom = (2*m.pi*var)**.5
        num = m.exp(-(float(x)-float(mean))**2/(2*var))
        return num/denom

if __name__ == '__main__':
    rb = Robot()
