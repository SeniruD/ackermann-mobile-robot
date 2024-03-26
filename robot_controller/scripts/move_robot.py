#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Float64
import rosbag
import os
# import datetime



import roslib
roslib.load_manifest('robot_controller')
import rospy
import actionlib

from robot_controller.msg import MoveRobotAction

class MoverRobotServer:
  def __init__(self):
    rospy.init_node('action_server')
    self.motor_pub = rospy.Publisher('commands/motor/unsmoothed_speed', Float64, queue_size=10)
    self.servo_pub = rospy.Publisher('commands/servo/unsmoothed_position', Float64, queue_size=10)
    self.rate = rospy.Rate(10)  # 10hz
    self.server = actionlib.SimpleActionServer('move_robot', MoveRobotAction, self.execute, False)
    self.server.start()
    self.speed = Float64()
    self.angle = Float64()
    self.speed.data = 0
    self.angle.data = 0

  def execute(self, goal):
    rospy.loginfo("VLN model output: %s recieved", goal) 

    if goal.action_id == 1: # move forward
        rospy.loginfo("Moving Forward")
        # self.play_bagfile("go_forward")
        # os.system('rosbag play --bags=/home/senirud/bag_files/go_forward.bag')
        self.set_motor_speed(150, 1.6)

    elif goal.action_id == 2: # turn left
        rospy.loginfo("Turning Left")
        self.set_servo_angle(0.15)
        self.set_motor_speed(150,1.35)
        rospy.sleep(1)
        self.set_servo_angle(0.85)
        self.set_motor_speed(-150,1.35)
        rospy.sleep(1)
        self.stop_robot()
        # self.play_bagfile("turn_left")
        # os.system('rosbag play --bags=/home/senirud/bag_files/turn_left.bag')


    
    elif goal.action_id == 3: # turn right
        rospy.loginfo("Turning Right")
        self.set_servo_angle(0.85)
        self.set_motor_speed(150,1.35)
        rospy.sleep(1)
        self.set_servo_angle(0.15)
        self.set_motor_speed(-150,1.35)
        rospy.sleep(1)
        self.stop_robot()



        # self.stop_robot()
        # self.play_bagfile("turn_right")
        # os.system('rosbag play --bags=/home/senirud/bag_files/turn_right.bag')

  
    elif goal.action_id == 0: # stop
        rospy.loginfo("Stopping the robot")

    else: 
        rospy.loginfo("Invalid Command!")
    rospy.loginfo('Command Executed Successfully!')     
    self.server.set_succeeded()

  def play_bagfile(self, command):

    bag_path = "/home/senirud/bag_files/"+ command +".bag"
    bag = rosbag.Bag(bag_path)
    rate = rospy.Rate(50) # 50hz
    for topic, msg, t in bag.read_messages():
        # timestamp = datetime.datetime.fromtimestamp(t.to_sec())
        # formatted_timestamp = timestamp.strftime('%b %d %Y %H:%M:%S.%f')[:-3]
        pub = rospy.Publisher(topic, type(msg), queue_size=10)
        pub.publish(msg)
        rate.sleep()
        print('Published topic and message: ',t, topic, msg)
    bag.close()
    # print(f'{command} executed successfully!')
    rospy.loginfo('Command Executed Successfully!')

  def set_motor_speed(self, speed, duration):
      self.speed.data = speed
      start_time = rospy.Time.now()
      end_time = start_time + rospy.Duration(duration)
      while rospy.Time.now() < end_time:
          self.motor_pub.publish(self.speed)
          self.rate.sleep()
      self.stop_robot()

  def set_servo_angle(self, angle):
    self.angle.data = angle
    start_time = rospy.Time.now()
    end_time = start_time + rospy.Duration(1.2)
    while rospy.Time.now() < end_time:
      self.servo_pub.publish(self.angle)
      self.rate.sleep()
    # start_time = rospy.Time.now()
    # end_time = start_time + rospy.Duration(duration)
    # while rospy.Time.now() < end_time:
    #     self.pub.publish(speed_data)
    #     self.rate.sleep()
   
  def stop_robot(self):
      self.motor_pub.publish(0.0)
      self.servo_pub.publish(0.5)
      self.rate.sleep()

def callback(event):
    rospy.loginfo('Timer called at' + str(event.current_real))

if __name__ == '__main__':
    try:
        server = MoverRobotServer()
    except rospy.ROSInterruptException:
        pass

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
