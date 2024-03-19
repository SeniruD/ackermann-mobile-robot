# Ackermann Robot For VLN
Implementation of ROS navigation stack on ackermann steered traxxas mobile robot


## Setting up the sensors and motors
roslaunch racecar teleop.launch joystick_enable:=false

## Setting up command execution module
rosrun robot_controller move_robot.py

## Setting up the VLNCE model
export PATH=/home/senirud/miniconda3/bin:$PATH
source activate vlnce
cd catkin_ws1/src/robot_controller/scripts/
rosrun robot_controller run.py 
