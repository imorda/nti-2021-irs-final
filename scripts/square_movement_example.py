#!/usr/bin/env python3

import sys
from robot_library.robot import *
import cv2
import rospy
import numpy as np
import math

# initialize robot
robot = Robot()
# defining constant from simulator
WHEEL_RADIUS = 0.09

def turn(add_deg_rad: float) -> float:
    """
    function turns robot at given angle in radians
    returns predicted position after turn (calculated position, real position may differ)
    """
    global robot

    # defining some constants
    MAX_TURN_SPEED = 0.4
    P_COEF = 0.4

    current_dir = robot.getDirection()
    # calculate target direction of robot after turn
    target_dir = current_dir + add_deg_rad

    # calculate error of rotation (nobody knows how it works, but it does)
    e = (target_dir - current_dir + np.pi * 5) % (np.pi * 2) - (np.pi)

    # accepting threshold after turn is 0.01
    while abs(e - np.sign(e) * np.pi) > 0.01:
        current_dir = robot.getDirection()
        e = (target_dir - current_dir + np.pi * 5) % (np.pi * 2) - (np.pi)
        turn_speed = -(e - np.sign(e) * np.pi) * P_COEF + np.sign(e) * 0.1

        # limit our speed with MAX_TURN_SPEED bound
        turn_speed = np.sign(turn_speed) * np.maximum(np.abs(turn_speed), MAX_TURN_SPEED)
        # equivalent to bottom line
        # turn_speed = (turn_speed if turn_speed > -MAX_TURN_SPEED else -MAX_TURN_SPEED) if turn_speed < MAX_TURN_SPEED else MAX_TURN_SPEED

        robot.setVelosities(0, turn_speed)

        # some delay for don't overload computation
        robot.sleep(0.001)

    robot.setVelosities(0,0)
    return target_dir

def moveDist(dist, target_dir = robot.getDirection()):
    """
    function moves robot on desired distance in meters
    """
    global WHEEL_RADIUS

    # calculating how many radians to the required position left
    dist_left = (dist / WHEEL_RADIUS)

    # save initial number of encoders
    initial_enc = robot.getEncoders().get("left")

    while abs(dist_left) > 0.005:
        # calculating how many radians to the required position left
        enc = robot.getEncoders().get("left")
        dist_left = initial_enc + (dist * 1.0 / WHEEL_RADIUS) - enc
        up = 0.1 * dist_left
        up = (up if up > -0.3 else -0.3) if up < 0.3 else 0.3


        # trave to the angle
        current_dir = robot.getDirection()
        e = (target_dir - current_dir + 9.42) % 6.28 - 3.14
        up_ang = e * 0.5
        up_ang = (up_ang if up_ang > -0.3 else -0.3) if up_ang < 0.3 else 0.3

        robot.setVelosities(up, up_ang)

        # some delay for don't overload computation
        robot.sleep(0.001)

    robot.setVelosities(0, 0)

if __name__ == "__main__":
    robot = Robot()
    # just move square 1x1m
    moveDist(1,turn(-1.57))
    robot.sleep(1)
    moveDist(1,turn(1.57))
    robot.sleep(1)
    moveDist(1,turn(1.57))
    robot.sleep(1)
    moveDist(1,turn(1.57))
    robot.sleep(1)

    exit(1)
