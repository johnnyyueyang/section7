#!/usr/bin/env python3

import typing as T
import numpy as np
import scipy.interpolate
from scipy.interpolate import splev

import rclpy
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from asl_tb3_lib.grids import StochOccupancyGrid2D
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.math_utils import wrap_angle

from P1_astar import AStar

def compute_smooth_plan(path, v_desired=0.15, spline_alpha=0.05) -> TrajectoryPlan:
        # Ensure path is a numpy array

        # Compute and set the following variables:
        #   1. ts: 
        #      Compute an array of time stamps for each planned waypoint assuming some constant 
        #      velocity between waypoints. 
        #
        #   2. path_x_spline, path_y_spline:
        #      Fit cubic splines to the x and y coordinates of the path separately
        #      with respect to the computed time stamp array.
        #      Hint: Use scipy.interpolate.splrep
        
        ##### YOUR CODE STARTS HERE #####
        ts_n = np.shape(path)[0]
        ts = np.zeros(ts_n)
        for i in range(ts_n-1):
            ts[i+1] = np.linalg.norm(path[i+1] - path[i]) / v_desired 
            ts[i+1] = ts[i+1] + ts[i]
        # print(ts)
        # print(path[: ,0])
        path_x_spline = scipy.interpolate.splrep(ts, path[: ,0], k=3, s=spline_alpha)
        path_y_spline = scipy.interpolate.splrep(ts, path[: ,1], k=3, s=spline_alpha)
        ###### YOUR CODE END HERE ######
        
        return TrajectoryPlan(
            path=path,
            path_x_spline=path_x_spline,
            path_y_spline=path_y_spline,
            duration=ts[-1],
        )