#!/usr/bin/env python3

import rospy

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32
from turbojpeg import TurboJPEG
import cv2
import numpy as np
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped, BoolStamped, VehicleCorners, WheelEncoderStamped
import os
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from geometry_msgs.msg import Point32
import time
from math import pi

ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
DEBUG = False
ENGLISH = False
RED_MASK = [(151, 155, 84), (179, 255, 255)]


class LaneFollowNode(DTROS):

    def __init__(self, node_name):
        super(LaneFollowNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        self.veh_name = rospy.get_namespace().strip("/")
        self.host = str(os.environ['VEHICLE_NAME'])
        self.veh = rospy.get_param("~veh")

        # Publishers
        self.pub = rospy.Publisher("/" + self.veh + "/output/image/mask/compressed",
                                   CompressedImage,
                                   queue_size=1)
        self.vel_pub = rospy.Publisher("/" + self.veh + "/car_cmd_switch_node/cmd",
                                       Twist2DStamped,
                                       queue_size=1)
        
        # Subscribers
        self.sub = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed",CompressedImage,self.callback,queue_size=1,buff_size="20MB")
        self.sub_avg = rospy.Subscriber("/{}/duckiebot_detection_node/x".format(self.host), Float32, self.cb_avg, queue_size=1)
        self.right_tick = rospy.Subscriber(f'/{self.veh_name}/right_wheel_encoder_node/tick', WheelEncoderStamped, self.cb_right_tick,  queue_size = 1)
        self.left_tick = rospy.Subscriber(f'/{self.veh_name}/left_wheel_encoder_node/tick', WheelEncoderStamped, self.cb_left_tick,  queue_size = 1)
        self.sub_detection = rospy.Subscriber("/{}/duckiebot_detection_node/detection".format(self.host), BoolStamped, self.cb_detection, queue_size=1)
        self.sub_leader = rospy.Subscriber("/{}/duckiebot_distance_node/distance".format(self.host), Float32, self.cb_distance, queue_size=1)

        self.radius = rospy.get_param(f'/{self.veh_name}/kinematics_node/radius', 100)
        
        
        self.move = True
        self.intersection = False
        self.leader= False
        self.safe_dist = 30
        self.detection = False  
        self.distance = 500000

        
        
        # Assistant module
        self.bridge = CvBridge()
        self.jpeg = TurboJPEG()

        self.loginfo("Initialized")

        # PID Variables
        self.proportional = None
        if ENGLISH:
            self.offset = -220
        else:
            self.offset = 220
            
        self.velocity = 0.23
        self.move_velocity = self.velocity
        self.twist = Twist2DStamped(v=self.velocity, omega=0)

        self.P = 0.049
        self.D = -0.004
        self.last_error = 0
        self.last_time = rospy.get_time()
        self.right_init = False
        self.right_dist = 0
        self.right_init_dist = 0
        self.left_init = False
        self.left_dist = 0
        self.left_init_dist = 0
        self.prv_rt = 0
        self.prv_lt = 0
        self.dist = 0

        # Shutdown hook
        rospy.on_shutdown(self.hook)

    def cb_right_tick(self, msg):
        if not self.right_init:
            self.right_init = True
            self.right_init_dist = msg.data
        self.right_dist = msg.data - self.right_init_dist

    def cb_left_tick(self, msg):
        if not self.left_init:
            self.left_init = True
            self.left_init_dist = msg.data
        self.left_dist = msg.data - self.left_init_dist


         
    def cb_distance(self, distance):

        self.distance = distance.data
        
        
    
        if self.distance*95 < self.safe_dist:
            self.leader = True
        else:
            self.leader = False


        if self.distance*95 < self.safe_dist:
            self.leader = True
        else:
            self.leader = False

        
    def cb_avg(self,msg):
        self.avg = msg.data

    
    def cb_detection(self, bool_msg):
        self.detection = bool_msg.data 


    def lane_follow(self, image_hsv, width, crop):

        mask = cv2.inRange(image_hsv, ROAD_MASK[0], ROAD_MASK[1])
        crop = cv2.bitwise_and(crop, crop, mask=mask)
        contours, hierarchy = cv2.findContours(mask,
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
        max_area = 20
        max_idx = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_idx = i
                max_area = area

        if max_idx != -1:
            M = cv2.moments(contours[max_idx])
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                first = M['m10']
                sec = M['m00']
                # rospy.loginfo(f'cx:{cx}, first: {first}, sec:{sec}')
                
                self.proportional = cx - int(width / 2) + self.offset
                if DEBUG:
                    cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                pass
        else:
            self.proportional = None
        
        self.drive()
            
    def auto(self, image_hsv, width, crop, center=-1):

        self.velocity = self.move_velocity

        if not self.intersection:
            lower_red = np.array([0,50,50])
            upper_red = np.array([10,255,255])
            mask_init = cv2.inRange(image_hsv, lower_red, upper_red)

  
            lower_red = np.array([170,50,50])
            upper_red = np.array([180,255,255])
            mask1 = cv2.inRange(image_hsv, lower_red, upper_red)
            mask = mask_init+mask1
            size = np.sum(mask/255.) / mask.size


            if size > 0.15:
                self.move = False
                self.proportional = int(width / 2)
                self.drive()
                rospy.sleep(3)
                self.move = True
                self.drive()
                self.intersection = True
                self.intersection_dist = self.dist + 0.6
                if center < 200: 
                    rospy.loginfo('left')
                elif center < 300:
                    rospy.loginfo('straight')
                else:
                    rospy.loginfo('right')

        if self.intersection and self.dist > self.intersection_dist:
            self.intersection = False

        if self.intersection:
            if center != -1:
                self.proportional = ((center) - int(width / 2)) / 3.5
                return
        
        self.lane_follow(image_hsv, width, crop)

    def callback(self, msg):

        img = self.jpeg.decode(msg.data)
        crop = img[300:-1, :, :]
        width = crop.shape[1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        delta_right = self.right_dist - self.prev_right
        delta_left = self.lleft_dist - self.prev_left

        self.prev_right = self.right_dist
        self.prev_left = self.left_dist

        delta_right = (2 * pi * self.radius * delta_right) / 135
        delta_left= (2 * pi * self.radius * delta_left) / 135
        self.dist = self.dist + (delta_right + delta_left)/2

        if not self.detection:
            self.move = True
            self.auto(hsv,width, crop)
        elif self.leader:
            self.move = False
            self.proportional = int(width / 2)
            self.drive()
        else:
            self.move = True
            self.auto(hsv, width, crop, center=self.avg)
            
        self.drive()
            

    def drive(self):
        """
        use PID to drive
        """
        if self.proportional is None:
            self.twist.omega = 0
        else:
            # P Term
            P = -self.proportional * self.P

            # D Term
            d_error = (self.proportional - self.last_error) / (rospy.get_time() - self.last_time)
            self.last_error = self.proportional
            self.last_time = rospy.get_time()
            D = d_error * self.D

            if self.move:
                self.twist.v = self.velocity
                self.twist.omega = P + D
            else:
                self.twist.v = 0
                self.twist.omega = 0
                
            rospy.loginfo(f'v:{self.twist.v}, omega: {self.twist.omega}')
                
            if DEBUG:
                self.loginfo(self.proportional, P, D, self.twist.omega, self.twist.v)

        self.vel_pub.publish(self.twist)

    def hook(self):
        print("SHUTTING DOWN")
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        for i in range(8):
            self.vel_pub.publish(self.twist)


if __name__ == "__main__":
    node = LaneFollowNode("lanefollow_node")
    rate = rospy.Rate(8)  # 8hz
    while not rospy.is_shutdown():
        # node.drive()
        rate.sleep()
