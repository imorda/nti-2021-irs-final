#!/usr/bin/env python3

import sys
from robot_library.robot import *
import cv2
import rospy
import numpy as np
from time import time


if __name__ == "__main__":
    # initialize robot
    robot = Robot()

    img = robot.getImage()
    
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
