#!/usr/bin/env python3
import os
import rospy
import math
import rosbag
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Pose2DStamped, WheelEncoderStamped, WheelsCmdStamped
from std_msgs.msg import Header, Float32

x_robot = 0 
y_robot = 0 

x_world = 0 
y_world = 0 

x_init = 0 
y_init = 0 

theta = 0 




class TransformationNode(DTROS):

	def __init__(self, node_name):
    
        # Initialize the DTROS parent class

		super(TransformationNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
		self.veh_name = rospy.get_namespace().strip("/")


	#Subscribers 
		self.sub_theta = rospy.Subscriber('/pose', Pose2DStamped, self.cb_pose, queue_size =1 ) 
		

        # Publishers
		self.pub = rospy.Publisher('/pose_tf',Pose2DStamped, queue_size = 1 )
		
		self.name = 'csc2245/robot_data.bag'
		self.bag = rosbag.Bag(self.name)

		self.log("Initialized")
	
	def cb_pose(self, msg):
		global x_robot
		global y_robot 
		global theta 
		
		x_robot = msg.x 
		y_robot = msg.y
		theta = msg.theta 
		

	def transformation(self):
		global x_world
		global y_world 
		global x_robot 
		global y_robot
		global x_init
		global y_init 
		global theta 
		
		x_world = x_init + x_robot*math.cos(theta) + y_robot*math.sin(theta)
		y_world = y_init - x_robot*math.sin(theta) + y_robot*math.sin(theta)
		
		x_init = x_world 
		y_init = y_world

	def bag_write(self):
		global x_world 
		global y_world 
		global theta 
		
		try:
			self.bag.write('x', x_world)
			self.bag.write('y', y_world)
			self.bag.write('theta', theta)
			
		except Exception as e:
			print(f'Error: {e}')
			self.bag.close()

	def run(self):
		t_pose = Pose2DStamped()
		t_pose.header.stamp = rospy.Time.now()
		t_pose.header.frame_id = "/t_pos"
		t_pose.x = x_world
		t_pose.y = y_world 
		t_pose.theta = theta 

		while not rospy.is_shutdown():
			self.pub.publish(t_pose)
			self.bag_write(t_pose)
			rospy.spin()
			rospy.loginfo("wheel_encoder_node is up and running...")
		
if __name__ == '__main__':
	
	node = TransformationNode(node_name='my_transformation_node')
	node.run()

	
	
	
