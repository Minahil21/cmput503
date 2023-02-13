#!/usr/bin/env python3

import rospy
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage, CameraInfo, Image

import numpy as np

image_data = 0 

class CameraNode(DTROS):

	def __init__(self, node_name):

		super(CameraNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
		self.sub_info = rospy.Subscriber('/csc22943/camera_node/camera_info', CameraInfo, self.cb_info)
		self.sub_image = rospy.Subscriber('/csc22943/camera_node/image/compressed', CompressedImage, self.cb_image)
		self.pub = rospy.Publisher('/robot_image/compressed', CompressedImage, queue_size=10)
		self.new_image = CompressedImage()

	def cb_info(self, msg):
		rospy.loginfo("Size of the image subscribed is: %f height x %f width", msg.height, msg.width)


	def cb_image(self, data):
		global image_data
		image_data = data.data
		
	def run(self):
		rate = rospy.Rate(1)
		self.new_image.header.frame_id = "/new_image"
		self.new_image.format = "jpeg"
		while not rospy.is_shutdown():
			self.new_image.header.stamp = rospy.Time.now()
			self.new_image.data = image_data 
			self.pub.publish(self.new_image)
			rate.sleep()
            

if __name__ == '__main__':
    node = CameraNode(node_name='robot_camera_node')
    node.run()
    rospy.spin() 

