#!/usr/bin/env python3
import rospy
from duckietown_msgs.srv import SetCustomLEDPattern, ChangePattern, ChangePatternResponse
from duckietown_msgs.msg import LEDPattern
from duckietown.dtros import DTROS, NodeType, TopicType



class LEDNode(DTROS):
    def __init__(self, node_name):
        
        super(LEDNode, self).__init__(node_name=node_name, node_type=NodeType.DRIVER)

        self.veh_name = rospy.get_namespace().strip("/")

        # Proxies
        self.setCustomPattern = rospy.ServiceProxy('/csc22945/led_emitter_node/set_custom_pattern'.format(self.veh_name), SetCustomLEDPattern)
            
        # Servers
        self.server = rospy.Service('/csc22945/led_node/led_pattern'.format(self.veh_name), ChangePattern, self.callback)
        
        # Publishers
        self.pub_LED = rospy.Publisher("~led_pattern", LEDPattern, queue_size=1, dt_topic_type=TopicType.DRIVER)



        self.set_color = {
            
            "red": [1, 0, 0],
            "pink": [1, 0, 0.5],
            "blue": [0, 0, 1],
            "green": [0, 1, 0]
        }

    def callback(self, msg: ChangePattern):

        led = LEDPattern()
        led.color_list = [msg.pattern_name.data] * 5
        led.color_mask = [1, 1, 1, 1, 1]
        led.frequency = 0.0
        led.frequency_mask = [0, 0, 0, 0, 0]
        self.setCustomPattern(led)

        return ChangePatternResponse()


if __name__ == "__main__":
    node = LEDNode(node_name="robot_led_node")
    rospy.spin()
    
