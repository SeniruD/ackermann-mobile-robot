#!/usr/bin/env python3

import roslib
roslib.load_manifest('robot_controller')
import rospy
import actionlib

from robot_controller.msg import MoveRobotAction, MoveRobotGoal

if __name__ == '__main__':
    rospy.init_node('action_client')
    client = actionlib.SimpleActionClient('move_robot', MoveRobotAction)
    client.wait_for_server()

    goal = MoveRobotGoal(action_id=2)
    client.send_goal(goal)
    client.wait_for_result(rospy.Duration.from_sec(5.0))
    rospy.spin()