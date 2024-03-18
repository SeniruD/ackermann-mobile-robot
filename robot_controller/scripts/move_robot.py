#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Float64
import rosbag

import roslib
roslib.load_manifest('robot_controller')
import rospy
import actionlib

from robot_controller.msg import MoveRobotAction


class MoverRobotServer:
  def __init__(self):
    self.server = actionlib.SimpleActionServer('move_robot', MoveRobotAction, self.execute, False)
    self.server.start()

  def execute(self, goal):
    rospy.loginfo("VLN model output: %s recieved", goal) 

    if goal.action_id == 1: # move forward
        rospy.loginfo("Moving Forward")
        self.play_bagfile("go_forward")

    elif goal.action_id == 2: # turn left
        rospy.loginfo("Turning Left")
        self.play_bagfile("turn_left")

    
    elif goal.action_id == 3: # turn right
        rospy.loginfo("Turning Right")
        self.play_bagfile("turn_right")
  
    elif goal.action_id == 0: # stop
        rospy.loginfo("Stopping the robot")

    else: 
        rospy.loginfo("Invalid Command!")
         
    self.server.set_succeeded()

  def play_bagfile(self, command):

    bag_path = "/home/senirud/bag_files/"+ command +".bag"
    bag = rosbag.Bag(bag_path)
    rate = rospy.Rate(50) # 50hz
    for topic, msg, t in bag.read_messages():
        pub = rospy.Publisher(topic, type(msg), queue_size=10)
        pub.publish(msg)
        rate.sleep()
        print('Published topic and message: ', topic, msg)
    bag.close()
    # print(f'{command} executed successfully!')
    rospy.loginfo('Command Executed Successfully!')

if __name__ == '__main__':
  rospy.init_node('robot_action_server')
  server = MoverRobotServer()
  rospy.spin()


# ------------------------


# def handle_action(req): 
#     rospy.loginfo("VLN model output: %s recieved", goal) 

#     if goal == 'MOVE_FORWARD':
#         print('Moving Forward')
#         play_bagfile("go_forward")

#     elif goal == 'TURN_LEFT':
#         print('Turning Left')
#         play_bagfile("turn_left")

    
#     elif goal == 'TURN_RIGHT':  
#         print('Turning Right')
#         play_bagfile("turn_right")
  
#     elif goal == 'STOP':
#         print('Stopping the robot')

#     else: 
#         print('Invalid Command!')  


# def robot_mover():
#     # rospy.Subscriber('/vln_out', String, callback)
#     # s = rospy.Service('execute_action', ModelAction, handle_action)
#     rospy.init_node('robot_controller')
#     mover = actionlib.SimpleActionServer('execute_action', ModelAction, handle_action, False)
#     rospy.spin()

# if __name__ == '__main__':
#     try:
#         robot_mover()
#     except rospy.ROSInterruptException:
#         pass
