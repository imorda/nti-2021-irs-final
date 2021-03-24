#!/bin/bash
cd ~/catkin_ws/src
rm -rf robot_library
git clone https://gitlab.com/iu-robotics-assignments/gazebo-assignments/robot_library
cd ~/catkin_ws
catkin build
