#!/usr/bin/env python3
# license removed for brevity

import rospy
from std_msgs.msg import String, Float64
import rosbag

def play_bagfile(command):

    bag_path = "/home/senirud/bag_files/"+ command +".bag"
    bag = rosbag.Bag(bag_path)
    rate = rospy.Rate(50) # 50hz
    for topic, msg, t in bag.read_messages():
        pub = rospy.Publisher(topic, type(msg), queue_size=10)
        pub.publish(msg)
        rate.sleep()
        print('Published topic and message: ', topic, msg)
    bag.close()
    print(f'{command} executed successfully!')

def callback(data): 
    rospy.loginfo("VLN model output: %s recieved", data.data) 

    if data.data == 'go_forward':
        print('Moving Forward')
        play_bagfile("go_forward")
    elif data.data == 'turn_left':
        print('Turning Left')
        play_bagfile("turn_left")

    
    elif data.data == 'turn_right':  
        print('Turning Right')
        play_bagfile("turn_right")
  
    elif data.data == 'stop':
        print('Stopping the robot')
    else: 
        print('Invalid Command!')  


def robot_mover():
    # pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.Subscriber('/vln_out', String, callback)
    rospy.init_node('robot_controller')
    rospy.spin()

if __name__ == '__main__':
    try:
        robot_mover()
    except rospy.ROSInterruptException:
        pass
