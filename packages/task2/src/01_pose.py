#!/usr/bin/env python3
import numpy as np
import os
import rospy
import math
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelEncoderStamped , Twist2DStamped, Pose2DStamped
from std_msgs.msg import Float32

flag1 = 1
flag2 = 1

init_left = 0 
init_right = 0 

dist_left = 0 
dist_right = 0

ticks_left = 0 
ticks_right = 0 

x = 0
y = 0
theta = 0 

class PoseNode(DTROS):
	def __init__(self, node_name):

        # Initialize the DTROS parent class

		super(PoseNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
		self.veh_name = rospy.get_namespace().strip("/")

        # Get static parameters
		self._radius = rospy.get_param(f'/{self.veh_name}/kinematics_node/radius', 350)

        # Subscribing to the wheel encoders
		self.sub_encoder_ticks_left = rospy.Subscriber('/csc22945/left_wheel_encoder_node/tick', WheelEncoderStamped, self.cb_encoder_left,queue_size = 1 )
		self.sub_encoder_ticks_right = rospy.Subscriber('/csc22945/right_wheel_encoder_node/tick', WheelEncoderStamped, self.cb_encoder_right, queue_size = 1 )

        # Publishers
		self.pub_pose = rospy.Publisher('/pose', Pose2DStamped, queue_size = 1)
		self.pub_coord_cmd = rospy.Publisher('/csc22945/car_cmd_switch_node/cmd', Twist2DStamped, queue_size=1)

		self.log("Initialized")
		
	#CALCULATION OF POSE FROM TICKS 
	def cb_encoder_left(self, msg):
		global flag1
		global init_left
		global dist_left
		global ticks_left
		
		if (flag1 == 1) :
			init_left = msg.data 
			dist_left = (2 * math.pi * 0.0318 * (init_left)) / 135 
			flag1 = 0 
		else:
			ticks_left = msg.data
			dist_left = (2 * math.pi * 0.0318 * (ticks_left-init_left)) / 135  
			rospy.loginfo("wheel encoder left init ticks are %f current ticks are %f and distance is %f", init_left, ticks_left , dist_left)
			flag1 = 0

	def cb_encoder_right(self, msg):

		global flag2
		global init_right
		global dist_right
		global ticks_right

		if (flag2 == 1) :
			init_right = msg.data
			dist_right = (2 * math.pi * 0.0318 * (init_right)) / 135
			flag2 = 0
		else:
			ticks_right = msg.data
			dist_right = (2 * math.pi * 0.0318 * (ticks_right-init_right)) / 135
			rospy.loginfo( "wheel encoder right init ticks are %f and current ticks are %f and distance is %f", init_right, ticks_right , dist_right)
			flag2 = 0 

	def calculate_theta(self):
		global dist_right 
		global dist_left
		global theta
		theta = ( dist_right - dist_left ) / (2 * 0.05)
		
	def calculate_loc(self):
		global dist_right
		global dist_left
		global x 
		global y 
		

		dA = (dist_left + dist_right) / 2
		x = dA * math.cos(theta)
		y = dA * math.sin(theta)
		
	def run(self):
    	
		robot_pose = Pose2DStamped()
		rate = rospy.Rate(10)
		while not rospy.is_shutdown():
		
			self.calculate_loc()
			self.calculate_theta()
			robot_pose.x = x 
			robot_pose.y = y
			robot_pose.theta = theta
			print("pose",robot_pose)
			self.pub_pose.publish(robot_pose)
			rate.sleep()
		
if __name__ == '__main__':
		
	node = PoseNode(node_name='robot_pose_node')
	node.run()



    	
    

