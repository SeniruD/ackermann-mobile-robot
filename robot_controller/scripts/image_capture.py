#!/usr/bin/env python3
# license removed for brevity

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def image_callback( data): 
    rospy.loginfo("Image recieved")
    bridge = CvBridge()
    try:
        # Convert the ROS Image message to an OpenCV image
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        cv2.imshow("Image window", cv_image)
            
        (h, w, channels) = cv_image.shape
        y = (w-h)//2
        rospy.loginfo("Image size: %d x %d", h, w)
        rospy.loginfo(cv_image)
        leftImage = cv_image[:, y:y+h]
        leftImage = cv2.resize(leftImage,(224,224))
        rospy.loginfo(leftImage.shape)
        
        cv2.imshow("cropped window", leftImage)
        cv2.waitKey(0) 
        cv2.destropAllWindows()
    except Exception as e:
        rospy.logerr("Error processing image: %s", str(e))
    image_sub.unregister()



def take_image():
    global image_sub
    image_sub = rospy.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image, image_callback)
    rospy.init_node('image_taker')
    rospy.spin()

if __name__ == '__main__':
    try:
        take_image()
    except rospy.ROSInterruptException:
        pass
