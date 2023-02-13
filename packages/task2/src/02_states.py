#!/usr/bin/env python3
import numpy as np
import os
import rospy
import math
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Pose2DStamped , WheelsCmdStamped, Twist2DStamped, LEDPattern
from std_msgs.msg import Float32, String
from duckietown_msgs.srv import ChangePattern,  SetCustomLEDPattern, ChangePattern, ChangePatternResponse

x = 0

y = 0 

theta = 0 

e_dist = 0 

state = 0 

ang_vel = 0

prev_error = 0 

angle = 0 

flag = 1

init_x = 0
init_y = 0
init_theta = 0

class StateNode(DTROS):

	def __init__(self, node_name):
	# Initialize the DTROS parent class
		super(StateNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
		self.veh_name = rospy.get_namespace().strip("/")

        # Get static parameters
		self._radius = rospy.get_param('/csc22945kinematics_node/radius', 318)

        # Subscribing to the wheel encoders
		self.sub_pose = rospy.Subscriber('/pose', Pose2DStamped, self.cb_pose, queue_size = 1 )

        # Publishers
		self.pub_vel= rospy.Publisher("/csc22945/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)
		self.pub_wheel_vel = rospy.Publisher('/csc22945/wheels_driver_node/wheels_cmd', WheelsCmdStamped, queue_size = 1)
		self.pub_led = rospy.Publisher(f'/csc22945/led_emitter_node/led_pattern',LEDPattern, queue_size=1) 


        # Proxy
		s = '/csc22945/led_node/led_pattern'.format(self.veh_name)
		rospy.wait_for_service(s)
		self.led_colour = rospy.ServiceProxy(s, ChangePattern)

	#Variables 
		self.velocity = Twist2DStamped()
		self.wheel_velocity = WheelsCmdStamped()
		
		self.log("Initialized")
	
	def led(self, colour):
	
		name = String()
		name.data = colour
		self.led_colour(name)		
		
	def cb_pose(self,msg):
		global x 
		global y
		global theta 
		global flag
		global init_x
		global init_y
		global init_theta

		
		if (flag == 1) :
			init_x = msg.x
			init_y = msg.y
			init_theta = msg.theta
			flag = 0
		else:
			x = msg.x - init_x
			y = msg.y - init_y
			theta = msg.theta - init_theta
		
	
	def turn_left(self,v):
		self.led('pink')
		self.wheel_velocity.vel_left  = -v
		self.wheel_velocity.vel_right = v
		self.pub_wheel_vel.publish(self.wheel_velocity)
		
	def turn_right(self,v):
		self.led('pink')
		self.wheel_velocity.vel_left  = v
		self.wheel_velocity.vel_right = -v
		self.pub_wheel_vel.publish(self.wheel_velocity)
		
	def forward(self,v,w):
		global theta 

		self.led('blue')
		if (theta < 0 ):
			self.wheel_velocity.vel_left  = v
			self.wheel_velocity.vel_right = v + 0.05
			self.pub_wheel_vel.publish(self.wheel_velocity)
		elif (theta > 0 ):
			self.wheel_velocity.vel_left  = v + 0.05
			self.wheel_velocity.vel_right = v 
			self.pub_wheel_vel.publish(self.wheel_velocity)
		
		
	def stop(self):
		self.wheel_velocity.vel_left  = 0
		self.wheel_velocity.vel_right = 0
		self.velocity.v  = 0
		self.velocity.omega = 0
		self.led('red')
		for i in range(50):
			print("stopped")
			self.pub_wheel_vel.publish(self.wheel_velocity)
			self.pub_vel.publish(self.velocity)
		
		
	def move_in_circle(self):
		self.velocity.v = 2
		self.velocity.omega = -5
		self.pub_vel.publish(self.velocity) 
		self.led('green')
		
		
	def euclidean_dist(self,x_goal,y_goal):
		e_dist = math.sqrt((x_goal - x)**2+(y_goal - y)**2)
		return e_dist
		
        
	def pid(self,desired_angle,actual_angle):
		global angle
		global prev_error 
		global ang_vel 
        
		Kp = 0.1
		Ki = 0.3
		
		error = desired_angle - actual_angle
		sum_error = error + prev_error
		prev_error = error 
		ang_vel = Kp*error + Ki*sum_error 
		
	def run(self):
		global flag

		self.turn_right(0.1)

		while not rospy.is_shutdown():
			global state 
			global theta 
			global ang_vel 
			global x
			global y
				

			if state == 0:
		
				while( abs(theta) < (((math.pi)/2)- 1)):
					print("in state 0")
					print("theta", theta)
					self.turn_right(0.25)
				state = 1
				flag = 1

			
			elif state == 1: 
				print("in state 1")
				while(abs(x) < 1):
					print("in state 1.1")
					print("x",x)
					self.pid(0,theta)
					e_dist = self.euclidean_dist(1,0)
					lin_vel = 0.3*e_dist
					print("lin_vel and ang_vel are in state 1.1 are:", lin_vel, ang_vel)
					self.forward(lin_vel, -ang_vel)
				self.stop()
				flag = 1 
			
				while(abs(theta) < (((math.pi)/2))-1):
					print("in state 1.2")
					#self.pid(-math.pi,theta_end)
					self.turn_left(0.35)
				self.stop()
				flag = 1 
			
				while(abs(x) < 1):
					print("in state 1.3")
					self.pid(0,theta)
					e_dist = self.euclidean_dist(1,0)
					lin_vel = 0.3*e_dist
					print("lin_vel and ang_vel are in state 1.3 are:", lin_vel, ang_vel)
					self.forward(lin_vel, -ang_vel)
				self.stop()
				flag = 1 
				
				while(abs(theta) < (((math.pi)/2)-1)):
					print("in state 1.4")
					self.turn_left(0.3)
				self.stop()
				flag = 1 
				
				while(abs(x) < 1):
					print("in state 1.4")
					self.pid(0,theta)
					e_dist = self.euclidean_dist(1,0)
					lin_vel = 0.3*e_dist
					print("lin_vel and ang_vel in state 1.4 are:", lin_vel, ang_vel)
					self.forward(lin_vel, -ang_vel)
				self.stop()
				flag = 1 					
			
				while(abs(theta) < (((math.pi)/2)- 1)):
					print("in state 1.5")
					self.turn_left(0.3)
				self.stop()
				flag = 1 

				while(abs(x) < 1):
					print("in state 1.6")
					self.pid(0,theta)
					e_dist = self.euclidean_dist(1,0)
					lin_vel = 0.3*e_dist
					print("lin_vel and ang_vel in state 1.6 are:", lin_vel, ang_vel)
					self.forward(lin_vel, -ang_vel)
				self.stop()
				flag = 1 					
			
				while(abs(theta) < ((math.pi) - 1)):
					print("in state 1.7")
					self.turn_left(0.3)
				self.stop()	
				flag = 1 		
				state = 2
			
			elif state == 2:
				while(theta < (math.pi)):
					print("in state 2")
					self.move_in_circle()
				break 
				


if __name__ == '__main__':
	
	robot = StateNode(node_name='robot_state_node')
	robot.run()
	robot.stop()


    	
    

