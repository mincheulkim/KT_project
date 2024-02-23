# 디버그 모드임당

from configparser import Interpolation
from difflib import diff_bytes
import math
import os
import random
import subprocess
import time
import copy
import planner  # 220928
import planner_warehouse  # 221102
import planner_U  # 221117
import astar.pure_astar  # 230213
import astar.pure_astar_RAL  # 240206
from os import path

import numpy.matlib


import dwa_pythonrobotics as dwa_pythonrobotics
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState
#from geometry_msgs.msg import Twist
from geometry_msgs.msg import Twist, Point   # 240214, final_goal output용
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from pedsim_msgs.msg import AgentStates, TrackedPersons, TrackedPerson
from sensor_msgs.msg import LaserScan

from tf.transformations import euler_from_quaternion

# 220915 image 처리
from sensor_msgs.msg import Image, CameraInfo
import cv2
from cv_bridge import CvBridge, CvBridgeError    # CvBridge: connection btwn ROS and OpenCV interface

GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.35
TIME_DELTA = 0.1

DYNAMIC_GLOBAL = True  # 221003    # global path replanning과 관련

PATH_AS_INPUT = False # 221014      # false
#PATH_AS_INPUT = True # 221019      # waypoint(5개)를 input으로 쓸것인지 결정

PARTIAL_VIEW = False ## 221114 TD3(아래쪽 절반), warehouse(아래쪽 절반) visible

SCENARIO = 'DWA'    # TD3, warehouse, U, DWA, warehouse_RAL (240205 for rebuttal stage)
#SCENARIO = 'warehouse_RAL'    

PURE_GP = False # 231020  pure astar planner(IS 및 social cost 미고려)
#PURE_GP = True # SimpleGP 트리거
#time_interval = 20
time_interval = 20

viz_flow_map = False

debug = False    # evaluate단에서 활성화할 시 시점과 종점을 대칭으로 생성해줌

evaluate = False   # qual figure 용. True로 할시, 각 시나리오 별 정해진 위치 + robot, ped_traj.txt 루트에 생성


# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
    goal_ok = True

    if -3.8 > x > -6.2 and 6.2 > y > 3.8:
        goal_ok = False

    if -1.3 > x > -2.7 and 4.7 > y > -0.2:
        goal_ok = False

    if -0.3 > x > -4.2 and 2.7 > y > 1.3:
        goal_ok = False

    if -0.8 > x > -4.2 and -2.3 > y > -4.2:
        goal_ok = False

    if -1.3 > x > -3.7 and -0.8 > y > -2.7:
        goal_ok = False

    if 4.2 > x > 0.8 and -1.8 > y > -3.2:
        goal_ok = False

    if 4 > x > 2.5 and 0.7 > y > -3.2:
        goal_ok = False

    if 6.2 > x > 3.8 and -3.3 > y > -4.2:
        goal_ok = False

    if 4.2 > x > 1.3 and 3.7 > y > 1.5:
        goal_ok = False

    if -3.0 > x > -7.2 and 0.5 > y > -1.5:
        goal_ok = False

    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
        goal_ok = False

    return goal_ok

def check_pos_U(x, y):
    goal_ok = True

    if 3.0 > x > -3.0 and 3+0.35 > y > 3-0.35:  # up
        goal_ok = False

    if 3+0.35 > x > 3-0.35 and 3 > y > -3:   # E
        goal_ok = False

    if 3.0 > x > -3.0 and -3+0.35 > y > -3-0.35:  # S
        goal_ok = False

    if -3+0.35 > x > -3-0.35 and 3 > y > -3:   # W
        goal_ok = False

    if -1.3 > x > -3.7 and -0.8 > y > -2.7:
        goal_ok = False
        
    if x < -4.5 or x > 4.5 or y < -4.5 or y > 4.5:
        goal_ok = False
        
    ## 221121 중간 사각형 안에 안생기게
    if -3 < x < 3 and -3 < y < 3:
        goal_ok = False

    return goal_ok


def check_pos_warehouse(x, y):   # 221102
    goal_ok = True
    # wall2
    if -9.0 > x > -1.0 and -6.0 > y > -10.0:
        goal_ok = False
    # wall3
    if -9.0 > x > -10.0 and 9.0 > y > -10.0:
        goal_ok = False
    # NW three pannels
    if -0.0 > x > -7 and 7.3 > y > 5.5:
        goal_ok = False
    # NW two pannels
    if -8.8 > x > -4.2 and 3 > y > 1:
        goal_ok = False
    # control panel
    if 2.0701+0.4336 > x > 2.0701-0.4336 and 1.1622+0.3 > y > 1.1622-0.3:
        goal_ok = False
    # rack1
    if 7.35056+0.4222 > x > 7.35056-0.4222 and 1.7443+0.9779 > y > 1.7443-0.9779:
        goal_ok = False
    # rack2
    if 5.36795+0.4222 > x > 5.36795-0.4222 and 1.7443+0.9779 > y > 1.7443-0.9779:
        goal_ok = False
    # rack3
    if 7.001420+0.4222 > x > 7.001420-0.4222 and -7.646700+1.95581 > y > -7.646700-1.95581:
        goal_ok = False
    # rack4
    if 4.97338+0.4222 > x > 4.97338-0.4222 and -7.646700+1.95581 > y > -7.646700-1.95581:
        goal_ok = False
    # rack5
    if 2.669490+0.4222 > x > 2.669490-0.4222 and -7.646700+1.95581 > y > -7.646700-1.95581:
        goal_ok = False
    # pole1
    if 5.02174+0.245805 > x > 5.02174-0.245805 and -2.39478+0.245805 > y > -2.39478-0.245805:
        goal_ok = False
    # pole2
    if 0.470710+0.245805 > x > 0.470710-0.245805 and -2.39478+0.245805 > y > -2.39478-0.245805:
        goal_ok = False
    # pole3
    if -3.93786+0.245805 > x > -3.93786-0.245805 and -2.39478+0.245805 > y > -2.39478-0.245805:
        goal_ok = False
    # 전체 지도 [-10, 10] 넘어가는 경우    # 221108
    if x < -10 or x > 10 or y < -10 or y > 10:
        goal_ok = False
        
    return goal_ok

def check_pos_warehouse_RAL(x, y):   # 240205 RA-L Rebuttal stage
    # Buffer 추가 240205
    buffer_length = 0.5
    goal_ok = True
    # 1번
    if -10-buffer_length < x < -9+buffer_length and -10-buffer_length < y < 8+buffer_length:
        goal_ok = False
    elif -10-buffer_length < x < -1+buffer_length and -10-buffer_length < y < -5+buffer_length:
        goal_ok = False
    elif -1-buffer_length < x < 10+buffer_length and -10-buffer_length < y < -9.7+buffer_length:
        goal_ok = False
    elif 2.2-buffer_length < x < 7.3+buffer_length and -10-buffer_length < y < -5.9+buffer_length:
        goal_ok = False
    elif -7.5-buffer_length < x < 0+buffer_length and 5.5-buffer_length < y < 7.5+buffer_length:
        goal_ok = False
    elif -10-buffer_length < x < -4.6+buffer_length and 1-buffer_length < y < 2.8+buffer_length:
        goal_ok = False
    elif 5-buffer_length < x < 7.7+buffer_length and 0.7-buffer_length < y < 2.8+buffer_length:
        goal_ok = False
    elif 2.5-buffer_length < x < 8.5+buffer_length and 5.5-buffer_length < y < 6+buffer_length:
        goal_ok = False
    elif 1.7-buffer_length < x < 3.1+buffer_length and 0.8-buffer_length < y < 1.5+buffer_length:
        goal_ok = False
    elif -4.2-buffer_length < x < -3.6+buffer_length and -2.6-buffer_length < y < -2.1+buffer_length:
        goal_ok = False
    elif 0.1-buffer_length < x < 0.6+buffer_length and -2.6-buffer_length < y < -2.1+buffer_length:
        goal_ok = False
    elif 4.7-buffer_length < x < 5.3+buffer_length and -2.6-buffer_length < y < -2.1+buffer_length:
        goal_ok = False    
    elif x < -9.5 or x > 9.5 or y < -9.5 or y > 9.5:
        goal_ok = False
        
    return goal_ok

def check_pos_DWA(x, y):   # 230126
    # 221219 buffer 추가
    #buffer_length = 0.25
    buffer_length = 0.5
    goal_ok = True
    
    # 밖으로 나가는 것도 고려해야 함
    if x <= -5.5+buffer_length or x>= 5.5-buffer_length or y <= -5.5+buffer_length or y> 5.5-buffer_length:
        goal_ok = False

    if -3.5-buffer_length <= x <= -1.5+buffer_length and -1-buffer_length <= y <= 2.5+buffer_length:
        goal_ok = False

    if 0-buffer_length <= x <= 2+buffer_length and -2.5-buffer_length <= y <= 0+buffer_length:
        goal_ok = False

    if 2-buffer_length <= x <= 5.5 and 2.5-buffer_length <= y <= 2.5 + buffer_length:
        goal_ok = False

    if 4-buffer_length <= x <= 5.5+buffer_length and -5.5-buffer_length <= y <= 0+buffer_length:
        goal_ok = False
        
        
    #### 221222 Evaluate 용 #####
    #### eleverter scene에서 시점, 종점이 사람 지역에 안생기도록
    if -1.5 <= x <= 2 and -1 <= y <= 2.5:
        goal_ok = False
        
    return goal_ok


def check_pos_DWA_evaluate(x, y):   # 230126
    # 221219 buffer 추가
    #buffer_length = 0.25
    buffer_length = 0.5
    goal_ok = True
    #print('야')
    
    # 밖으로 나가는 것도 고려해야 함
    if x <= -5.5+buffer_length or x>= 5.5-buffer_length or y <= -5.5+buffer_length or y> 5.5-buffer_length:
        goal_ok = False

    if -3.5-buffer_length <= x <= -1.5+buffer_length and -1-buffer_length <= y <= 2.5+buffer_length:
        goal_ok = False

    if 0-buffer_length <= x <= 2+buffer_length and -2.5-buffer_length <= y <= 0+buffer_length:
        goal_ok = False

    if 2-buffer_length <= x <= 5.5 and 2.5-buffer_length <= y <= 2.5 + buffer_length:
        goal_ok = False

    if 4-buffer_length <= x <= 5.5+buffer_length and -5.5-buffer_length <= y <= 0+buffer_length:
        goal_ok = False
        
        
    #### 230222 Evaluate 용 #####
    #### eleverter scene에서 시점, 종점이 사람 지역에 안생기도록
    if -3 <= x <= 3 and -3 <= y <= 3:
        goal_ok = False
        
    return goal_ok


class GazeboEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, environment_dim, path_as_input):   # launchfile = "multi_robot_scenario.launch", env_dim = 20
        self.sac_path = path_as_input
        
        self.environment_dim = environment_dim   # 20
        self.odom_x = 0
        self.odom_y = 0

        self.goal_x = 1
        self.goal_y = 0.0
        
        self.optimal_gx = self.goal_x
        self.optimal_gy = self.goal_y
        
        self.euler = 0
        self.current_path_id = 0
        self.final_path_id = -1

        self.upper = 5.0
        self.lower = -5.0
        self.velodyne_data = np.ones(self.environment_dim) * 10
        self.last_odom = None
        self.pedsim_agents_list = None
        self.pedsim_agents_list_oracle = None
        self.pedsim_agents_list_oracle_id = None
        self.human_num = 12

        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0
        
        self.distOld = math.sqrt(math.pow(self.odom_x - self.goal_x, 2)+ math.pow(self.odom_y - self.goal_y, 2))
        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
            )
        self.gaps[-1][-1] += 0.03
        
        self.dwa_x = np.array([self.set_self_state.pose.position.x, self.set_self_state.pose.position.y, self.set_self_state.pose.orientation.w, 0.0, 0.0])
        
        self.a_star = None

        port = "11311"
        subprocess.Popen(["roscore", "-p", port])

        print("Roscore launched!")

        # Launch the simulation with the given launchfile name
        rospy.init_node("gym", anonymous=True)
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo launched!")

        # Set up the ROS publishers and subscribers
        self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        self.publisher4 = rospy.Publisher("waypoints", MarkerArray, queue_size=1)
        self.publisher5 = rospy.Publisher("optimal_goal", MarkerArray, queue_size=1)
        self.publisher6 = rospy.Publisher("path_as_init", MarkerArray, queue_size=1)   # 221020
        self.publisher7 = rospy.Publisher("flow_map", MarkerArray, queue_size=1)   # 230221
        self.publisher8 = rospy.Publisher("static_map", MarkerArray, queue_size=10)   # 231107
        self.velodyne = rospy.Subscriber(
            "/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1
        )
        self.odom = rospy.Subscriber(
            "/r1/odom", Odometry, self.odom_callback, queue_size=1
        )
        
        self.final_goal_pub = rospy.Publisher("/final_goal", Point, queue_size=1)   # 240214   
        
        ### PED_MAP 생성
        self.ped = rospy.Subscriber('/pedsim_visualizer/tracked_persons', TrackedPersons, self.ped_callback)
        self.get_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.track_ped_pub = rospy.Publisher('/track_ped', TrackedPersons, queue_size=10)
        
        ### NAV_MAP 생성
        self.ped_pos_map = []
        self.scan = [] #np.zeros(180)
        self.scan_all = np.zeros(1080)
        self.goal_cart = np.zeros(2)
        self.goal_final_cart = np.zeros(2)
        self.vel = np.zeros(2)
        # temporal data:
        self.ped_pos_map_tmp = np.zeros((2,80,80))  # cartesian velocity map
        self.scan_tmp = np.zeros(180)
        self.scan_all_tmp = np.zeros(1080)
        # initialize ROS objects
        self.ped_sub = rospy.Subscriber("/track_ped", TrackedPersons, self.ped_callback_sub)
#        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        self.scan_sub = rospy.Subscriber("/r1/front_laser/scan", LaserScan, self.scan_callback)
#        self.goal_sub = rospy.Subscriber("/cnn_goal", Point, self.goal_callback)
        self.goal_sub = rospy.Subscriber("/final_goal", Point, self.goal_callback)
#        self.vel_sub = rospy.Subscriber("/mobile_base/commands/velocity", Twist, self.vel_callback)
        self.vel_sub = rospy.Subscriber("/r1/cmd_vel", Twist, self.vel_callback)
#        self.cnn_data_pub = rospy.Publisher('/cnn_data', CNN_data, queue_size=1, latch=False)
        # timer:


        
        #220927 pedsim agent subscriber
        self.pedsim_agents = rospy.Subscriber("/pedsim_simulator/simulated_agents", AgentStates, self.actor_poses_callback)
        
        self.path_i_prev = None    # global path 저장
        self.path_rviz = None      # rviz에 보여줄 global path 저장
        self.path_as_input = []    # 221010
        self.path_as_input_no = 5  # 221010 5개 샘플
        self.path_as_init = None
        self.flow_map = None
        
        

    # Read velodyne pointcloud and turn it into distance data, then select the minimum value for each angle
    # range as state representation
    def velodyne_callback(self, v):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(self.environment_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / ((mag1 * mag2)+0.000000001)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        break

    def odom_callback(self, od_data):
        self.last_odom = od_data

        
    def get_robot_states(self):
        """
        Obtaining robot current position (x, y, theta) 
        :param x x-position of the robot
        :param y y-position of the robot
        :param theta theta-position of the robot
        """
        robot = None
        # get robot and pedestrian states:
        rospy.wait_for_service("/gazebo/get_model_state")
        try:
            # get robot state:
#            model_name = 'mobile_base'
             model_name = 'r1' 
#            relative_entity_name = 'world'
             relative_entity_name = ''
   # https://answers.gazebosim.org//question/18372/getting-model-state-via-rospy/            
             robot = self.get_state_service(model_name, relative_entity_name)
        except (rospy.ServiceException):
            rospy.logwarn("/gazebo/get_model_state service call failed") 
        
        return robot
    
    # Callback function for the path subscriber
    def ped_callback(self, peds_msg):
        peds = peds_msg
        # get robot states:
        robot = self.get_robot_states()

        if(robot is not None):
            # get robot poses and velocities: 
            robot_pos = np.zeros(3)
            robot_pos[:2] = np.array([robot.pose.position.x, robot.pose.position.y])
            robot_quaternion = (
                                robot.pose.orientation.x,
                                robot.pose.orientation.y,
                                robot.pose.orientation.z,
                                robot.pose.orientation.w)
            (_,_, robot_pos[2]) = tf.transformations.euler_from_quaternion(robot_quaternion)
            robot_vel = np.array([robot.twist.linear.x, robot.twist.linear.y])
            #print(robot_vel)
            # homogeneous transformation matrix: map_T_robot
            map_R_robot = np.array([[np.cos(robot_pos[2]), -np.sin(robot_pos[2])],
                                    [np.sin(robot_pos[2]),  np.cos(robot_pos[2])],
                                ])
            map_T_robot = np.array([[np.cos(robot_pos[2]), -np.sin(robot_pos[2]), robot_pos[0]],
                                    [np.sin(robot_pos[2]),  np.cos(robot_pos[2]), robot_pos[1]],
                                    [0, 0, 1]])
            # robot_T_map = (map_T_robot)^(-1)
            robot_R_map = np.linalg.inv(map_R_robot)
            robot_T_map = np.linalg.inv(map_T_robot)

            # get pedestrian poses and velocities:
            tracked_peds = TrackedPersons()
            #tracked_peds.header = peds.header
            tracked_peds.header.frame_id = 'base_footprint'
            tracked_peds.header.stamp = rospy.Time.now()
            for ped in peds.tracks:
                tracked_ped = TrackedPerson()
                # relative positions and velocities:
                ped_pos = np.array([ped.pose.pose.position.x, ped.pose.pose.position.y, 1])
                ped_vel = np.array([ped.twist.twist.linear.x, ped.twist.twist.linear.y])
                ped_pos_in_robot = np.matmul(robot_T_map, ped_pos.T)
                ped_vel_in_robot = np.matmul(robot_R_map, ped_vel.T) 
                # pedestrain message: 
                tracked_ped = ped
                tracked_ped.pose.pose.position.x = ped_pos_in_robot[0]
                tracked_ped.pose.pose.position.y = ped_pos_in_robot[1]
                tracked_ped.twist.twist.linear.x = ped_vel_in_robot[0]
                tracked_ped.twist.twist.linear.y = ped_vel_in_robot[1]
                tracked_peds.tracks.append(tracked_ped)
            # publish the pedestrains
            self.track_ped_pub.publish(tracked_peds)

    def ped_callback_sub(self, trackPed_msg):
        # get the pedstrain's position:
        self.ped_pos_map_tmp = np.zeros((2,80,80))  # cartesian velocity map
        if(len(trackPed_msg.tracks) != 0):  # tracker results
            for ped in trackPed_msg.tracks:
                #ped_id = ped.track_id 
                # create pedestrian's postion costmap: 10*10 m
                x = ped.pose.pose.position.x
                y = ped.pose.pose.position.y
                vx = ped.twist.twist.linear.x
                vy = ped.twist.twist.linear.y
                # 20m * 20m occupancy map:
                if(x >= 0 and x <= 20 and np.abs(y) <= 10):
                    # bin size: 0.25 m
                    c = int(np.floor(-(y-10)/0.25))
                    r = int(np.floor(x/0.25))

                    if(r == 80):
                        r = r - 1
                    if(c == 80):
                        c = c - 1
                    # cartesian velocity map
                    self.ped_pos_map_tmp[0,r,c] = vx
                    self.ped_pos_map_tmp[1,r,c] = vy

    # Callback function for the scan measurement subscriber
    def scan_callback(self, laserScan_msg):
        # get the laser scan data:
        self.scan_tmp = np.zeros(180)
        self.scan_all_tmp = np.zeros(1080)
        scan_data = np.array(laserScan_msg.ranges, dtype=np.float32)
        scan_data[np.isnan(scan_data)] = 0.
        scan_data[np.isinf(scan_data)] = 0.
        self.scan_tmp = scan_data[180:900]
        self.scan_all_tmp = scan_data

        
    # Callback function for the current goal subscriber
    def goal_callback(self, goal_msg):
        # Cartesian coordinate:
        self.goal_cart = np.zeros(2)
        self.goal_cart[0] = goal_msg.x
        self.goal_cart[1] = goal_msg.y


    # Callback function for the velocity subscriber
    def vel_callback(self, vel_msg):
        self.vel = np.zeros(2)
        self.vel[0] = vel_msg.linear.x
        self.vel[1] = vel_msg.angular.z



    # 220915
    # ref: https://github.com/xie9187/Monocular-Obstacle-Avoidance/blob/master/D3QN/GazeboWorld.py
    def GetRGBImageObservation(self):
        # convert ROS image message to to OpenCVcv2 image
        try:    
            # self.camera_image is an ndarray with shape (h, w, c) -> (228, 304, 3)
            cv_img = self.bridge.imgmsg_to_cv2(self.rgb_image, "bgr8")   # mono8, mono16, bgr8, rgb8
            #cv_img = self.bridge.imgmsg_to_cv2(self.rgb_image, desired_encoding='passthrough')   # http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
            # imgmsg data -> opencv
            # "bge8": CV_8UC3, color image with blue-green-red color order
        except Exception as e:
            raise e
        # resize
        dim = (self.rgb_image_size[0], self.rgb_image_size[1])
        cv_resized_img = cv2.resize(cv_img, dim, interpolation = cv2.INTER_AREA)
        # 밑에는 getDepthImage에서 가져온 파트
        # convert Opencv2 image to ROS image message and publish (반대과정. 퍼블리쉬 용)
        try:
            resized_img = self.bridge.cv2_to_imgmsg(cv_resized_img, "bgr8")
        except Exception as e:
            raise e
        #self.resized_rgb_img.publish(resized_img)   # 220915 주석 처리. resized된 이미지를 publish
        return(cv_resized_img)
    
    def actor_poses_callback(self, actors):
        self.pedsim_agents_list = []
        self.pedsim_agents_list_oracle = []
        self.pedsim_agents_list_oracle_id = []
        for actor in actors.agent_states:
            actor_id = str( actor.id )
            actor_pose = actor.pose
            #print("Spawning model: actor_id = %s", actor_id)
            x= actor_pose.position.x
            y= actor_pose.position.y
            #print(actor_id, x, y)
            actor_twist = actor.twist   # 230131
            vx = actor_twist.linear.x
            vy = actor_twist.linear.y
            #print(actor_id, vx, vy)
            
            self.pedsim_agents_list_oracle.append([x,y, vx, vy]) # 230828 for SD metric
            self.pedsim_agents_list_oracle_id.append([actor_id, x, y])
            
            
            # 221114 partial view 상황 가정
            if PARTIAL_VIEW != True:   # fully observable일때
                self.pedsim_agents_list.append([x,y, vx, vy])  # 230131
                #print('액터:',actor_id,'model_pose:',x, y)
                
            if PARTIAL_VIEW and SCENARIO=='TD3':   # partial view이고 TD3 환경일때
                if -5 < y < 0:    # 아래쪽 다 보이는 경우
                    self.pedsim_agents_list.append([x,y, vx, vy])  # 230131
                
            if PARTIAL_VIEW and SCENARIO=='warehouse':  # partial view이고 warehouse 환경일때
                if -10 < y < 10:   # 아래쪽 다 보이는 경우
                    self.pedsim_agents_list.append([x,y, vx, vy])  # 230131
                    
            if PARTIAL_VIEW and SCENARIO=='U':
                if -1 < x < 1 and 3 < y < 5:
                    self.pedsim_agents_list.append([x,y, vx, vy])  # 230131
            
            if PARTIAL_VIEW and SCENARIO=='DWA':   # partial view이고 dwa 환경일때
                if (-5.5 <= x <= 0.0 and -5.5 <= y <= -1) or (-1.5 <= x <= 0.0 and -5.0 <= y <= 2.5) or (2.0 <= x <= 4.0 and -5.5 <= y <= 2.5):      # CCTV 3개 alive(ORIGINAL)
                # Ablation study
                ####if (-5.5 <= x <= 0.0 and -5.5 <= y <= -1) or (-1.5 <= x <= 0.0 and -5.0 <= y <= 2.5):      # CCTV 2개 (1,2))
                ####if (-5.5 <= x <= 0.0 and -5.5 <= y <= -1) or (2.0 <= x <= 4.0 and -5.5 <= y <= 2.5):      # CCTV 2개 (1,3))
                ####if (-1.5 <= x <= 0.0 and -5.0 <= y <= 2.5) or (2.0 <= x <= 4.0 and -5.5 <= y <= 2.5):      # CCTV 2개 (2,3))
                ####if (-5.5 <= x <= 0.0 and -5.5 <= y <= -1):      # CCTV 1개 alive (1번)
                ####if (-1.5 <= x <= 0.0 and -5.0 <= y <= 2.5):     # CCTV 1개 alive (2번)
                ####if (2.0 <= x <= 4.0 and -5.5 <= y <= 2.5):      # CCTV 1개 alive (3번)
                    self.pedsim_agents_list.append([x,y, vx, vy])  # 230131
                # unlimited case는 if 삭제하고 아래줄 한칸 앞으로 땡기면 됨
            if PARTIAL_VIEW and SCENARIO=='warehouse_RAL':   # 240205 for RA-L Rebuttal stage
                if (-9 <= x <= 9.5 and -6.0 <= y <= -2.6) or (-9 <= x <= 0 and 2.8 <= y <= 5.5) or (-3.6 <= x <= 5 and -2.6 <= y <= 5.5):      # CCTV 3개 alive(ORIGINAL)
                    self.pedsim_agents_list.append([x,y, vx, vy])

        #print('페드심 리스트: ', self.pedsim_agents_list)
            
            
    # Perform an action and read a new state
    def step(self, action, episode_steps):        
        target = False
        # 221005 이동하기 전 거리
        distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])
        self.pre_distance = copy.deepcopy(distance)
        self.pre_odom_x = copy.deepcopy(self.odom_x)   # 221101
        self.pre_odom_y = copy.deepcopy(self.odom_y)   # 221101
        
        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)   # the robot movement is executed by a ROS publisher      
        
        final_goal_step = Point()
        final_goal_step.x = self.goal_x
        final_goal_step.y = self.goal_y
        self.final_goal_pub.publish(final_goal_step)    # 240214  

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
            
        # 220915 CCTV1 정보 cv로 받아옴
        '''
        self.rgb_cv_cctv1 = self.GetRGBImageObservation()   # shape: (512, 512, 3)
        cctv = cv2.resize(self.rgb_cv_cctv1, dsize=(512,512))
        cctv = cv2.cvtColor(cctv, cv2.COLOR_BGR2RGB) 
        '''
        ## (Optional) cctv visualize
        #cv2.imshow('cctv1', cctv)
        #cv2.waitKey(1)
        
        ### 220920 edge detector (Harris corner detector)
        #self.harris_corder_detector(self.rgb_cv_cctv1)
        
        
        # read velodyne laser state
        done, collision, min_laser = self.observe_collision(self.velodyne_data)   # 0.35보다 min_laser 작으면 충돌, 아니면 return
        v_state = []
        v_state[:] = self.velodyne_data[:]   
        laser_state = [v_state]

        # Calculate robot heading from odometry data
        self.odom_x = self.last_odom.pose.pose.position.x   
        self.odom_y = self.last_odom.pose.pose.position.y 
        self.odom_vx = self.last_odom.twist.twist.linear.x
        self.odom_vw = self.last_odom.twist.twist.angular.z
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)
        (_, _, self.euler) = euler_from_quaternion([self.last_odom.pose.pose.orientation.x, self.last_odom.pose.pose.orientation.y, self.last_odom.pose.pose.orientation.z, self.last_odom.pose.pose.orientation.w])
        
        # Calculate distance to the goal from the robot
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )
        self.distance = distance   # 221005

        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / ((mag1 * mag2)+0.000000001))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            target = True
            done = True
        
        # 221019
        local_goal = self.get_local_goal(self.odom_x, self.odom_y, self.goal_x, self.goal_y, self.euler)
        #robot_state = [local_goal[0], local_goal[1], action[0], action[1]]    # local_gx, local_gy
        robot_state = [distance, theta, action[0], action[1]]    # 상대거리, 헤딩, v, w
        
        state = np.append(laser_state, robot_state)              # 20 + 4

        self.path_as_input = copy.deepcopy(self.path_as_init)
        #print('[step]self.path_as_init;',self.path_as_init)   # global paoint에서 path as init 보여줌
        
        
        # 221114 Reliability checker   TODO
        # input: path_as_input
        # method: partial observation area에 또는 robot visible area에 위치하면 reliability = 1.0, 아니면 0.2
        # output: realibility
        reliability_score = self.get_reliablity(self.path_as_input, PARTIAL_VIEW)
        #print('웨이포인트:',self.path_as_input)
        #print('리얼리비티:',reliability_score)

        '''
        #if PATH_AS_INPUT:
        if self.sac_path:
            #reward = self.get_reward_path(target, collision, action, min_laser, self.odom_x, self.odom_y, self.path_as_input, self.goal_x, self.goal_y)
            reward = self.get_reward_path_230214_realibility(target, collision, action, min_laser, self.odom_x, self.odom_y, self.path_as_input, self.goal_x, self.goal_y, self.pre_distance, self.distance, self.pre_odom_x, self.pre_odom_y, reliability_score)
            #reward = self.get_reward_path_230206_noreliability(target, collision, action, min_laser, self.odom_x, self.odom_y, self.path_as_input, self.goal_x, self.goal_y, self.pre_distance, self.distance, self.pre_odom_x, self.pre_odom_y)  # 230206
            #reward = self.get_reward(target, collision, action, min_laser) # 230205 for test
            #reward = self.get_reward_path_230203_TD3(target, collision, action, min_laser, self.odom_x, self.odom_y, self.path_as_input, self.goal_x, self.goal_y, self.pre_distance, self.distance, self.pre_odom_x, self.pre_odom_y)
            #원복해야됨!
            #reward = self.get_reward_path_230206_pureDRL(target, collision, action, min_laser, self.odom_x, self.odom_y, self.path_as_input, self.goal_x, self.goal_y, self.pre_distance, self.distance, self.pre_odom_x, self.pre_odom_y)
        else:
            #reward = self.get_reward(target, collision, action, min_laser)
            # 230227 pureDRL
            reward = self.get_reward_path_230206_pureDRL(target, collision, action, min_laser, self.odom_x, self.odom_y, self.path_as_input, self.goal_x, self.goal_y, self.pre_distance, self.distance, self.pre_odom_x, self.pre_odom_y)
        '''
        
        #220928 path 생성
        # 정적 생성(option 1)
        if DYNAMIC_GLOBAL is not True:
            path = self.path_i_prev   # reset단에 최초로 생성된 global path
            self.path_i_rviz = path
        
        # 동적 생성(option 2)
        # try, excep: https://dojang.io/mod/page/view.php?id=2398
        #if DYNAMIC_GLOBAL:
        
        
        # Waypoint replanning 조건 230130
        RECAL_WPT = False
        for i, waypoint in enumerate(self.path_as_init):
            wpt_distance = np.linalg.norm(waypoint - [self.odom_x, self.odom_y])
            #print(i, waypoint, wpt_distance)
            if wpt_distance < 1.5:    # 1.5 as ahead distance
                RECAL_WPT = True
        
        #if DYNAMIC_GLOBAL and episode_steps%1 ==0:   # 선택 1(fixed rewrind)
        #####if DYNAMIC_GLOBAL and RECAL_WPT:             # 선택 2 아무 웨이포인트나 1.5안에 들어오면 replanning
        #####if DYNAMIC_GLOBAL and self.pedsim_agents_list != None:   # 선택 3. CCTV안에 pedsim list 들어오면   # 230206
        #if DYNAMIC_GLOBAL and self.pedsim_agents_list != None and episode_steps%time_interval == 0:   # 선택 4. CCTV안에 pedsim list 들어오면 + 너무 자주 리플래닝 되지는 않게  # 230209
        if DYNAMIC_GLOBAL and episode_steps%time_interval == 0:   # 선택 5. pedsim 무관 replanning  240206
    
            while True:
                try:
                    if SCENARIO=='warehouse':
                        path = planner_warehouse.main(self.odom_x, self.odom_y, self.goal_x, self.goal_y, self.pedsim_agents_list)
                    elif SCENARIO=='TD3':
                        path = planner.main(self.odom_x, self.odom_y, self.goal_x, self.goal_y, self.pedsim_agents_list)  
                    elif SCENARIO=='U':
                        path = planner_U.main(self.odom_x, self.odom_y, self.goal_x, self.goal_y, self.pedsim_agents_list)  
                    elif SCENARIO=='DWA':
                        #path = planner_DWA.main(self.odom_x, self.odom_y, self.goal_x, self.goal_y, self.pedsim_agents_list)  
                        #path = planner_astar.main(self.odom_x, self.odom_y, self.goal_x, self.goal_y, self.pedsim_agents_list) 
                        
                        #############################
                        #print('로봇 오돔:',self.odom_vx, self.odom_vw)
                        map_bias = 5.5
                        resolution = 0.1
                        grid_size = 1.0  # [m]
                        robot_radius = 5.0  # [m]
                        sx = int((self.odom_x+map_bias)/resolution)
                        sy = int((self.odom_y+map_bias)/resolution)
                        gx = int((self.goal_x+map_bias)/resolution)
                        gy = int((self.goal_y+map_bias)/resolution)
                        
                        #self.pause()   # 240206 한번 해제해 볼까?
                        if PURE_GP:
                            self.pedsim_agents_list = []    # Pure Global planner 하고 싶으면 주석해제
                        #rx, ry, flow_map = self.a_star.planning(sx, sy, gx, gy, self.odom_vx, angle, skew_x, skew_y, self.pedsim_agents_list)
                        rx, ry, flow_map = self.a_star.planning(sx, sy, gx, gy, vel_cmd.linear.x, angle, skew_x, skew_y, self.pedsim_agents_list) #
                        self.flow_map = flow_map   # 230221
                        #self.unpause()
                        
                        final_path = []
                        for path in zip (rx, ry):
                            final_path.append([path[0], path[1]])

                        final_path = np.array(final_path)
                        final_path = final_path / 10
                        path = final_path 
                        #path = astar.pure_astar.main(self.odom_x, self.odom_y, self.goal_x, self.goal_y, self.pedsim_agents_list) 
                    elif SCENARIO=='warehouse_RAL':
                        ### TODO ###                        
                        map_bias = 10.0
                        resolution = 0.1
                        grid_size = 1.0  # [m]
                        robot_radius = 10.0  # 5.0[m]   
                        sx = int((self.odom_x+map_bias)/resolution)
                        sy = int((self.odom_y+map_bias)/resolution)
                        gx = int((self.goal_x+map_bias)/resolution)
                        gy = int((self.goal_y+map_bias)/resolution)
                        self.pause()
                        if PURE_GP:
                            self.pedsim_agents_list = []   
                        #rx, ry, flow_map = self.a_star.planning(sx, sy, gx, gy, self.odom_vx, angle, skew_x, skew_y, self.pedsim_agents_list)  # ??
                        rx, ry, flow_map = self.a_star.planning(sx, sy, gx, gy, vel_cmd.linear.x, angle, skew_x, skew_y, self.pedsim_agents_list)
                        self.flow_map = flow_map   # RViz에 보여주기 위해 저장
                        self.unpause()
                        
                        final_path = []
                        for path in zip (rx, ry):
                            final_path.append([path[0], path[1]])

                        #print(final_path)
                        final_path = np.array(final_path)
                        final_path = final_path / map_bias
                        path = final_path 
                        
                    self.path_i_rviz = path   # 각 시나리오별로 얻은 path를 self.path_i_rviz에 저장
                    break
                except:
                    path = [[self.goal_x+5.5, self.goal_y+5.5]]   # 230209 5.5 더해줌
                    path = np.asarray(path)   # 221103
                    self.path_i_rviz = path
                    print('예외발생[step]. path를 global goal로 지정: ', path)
                    break
            #print('스탭때 패스:',path)
            #print('path as rviz:',self.path_i_rviz)
            
            #rx, ry = self.a_star.planning(sx, sy, gx, gy, self.odom_vx, angle, skew_x, skew_y, self.pedsim_agents_list)   
            # TODO sampling 방법에 대해 고려
            #############################    
            ############# 221010 고정된 5 사이즈의 path output    self.path_as_input
            self.path_as_input = []
            for i in range(self.path_as_input_no):
                self.path_as_input.append([self.goal_x, self.goal_y])
                
            self.path_as_input = np.asarray(self.path_as_input, dtype=float)
            if SCENARIO=='warehouse':
                # 만약 path가 더 작다면: # 앞단의 패스 길이만큼으로 대치 (남는 뒷부분들은 init goals)
                if len(path) < self.path_as_input_no:
                    #print(path.shape, self.path_as_input.shape)   # 8, 2  5, 2
                    self.path_as_input[:len(path), :] = path-10.0
                
                # 221114
                # 만약 path가 더 길다면: # 패스중 5를 랜덤하게 샘플링 (https://jimmy-ai.tistory.com/287)      
                elif len(path) > self.path_as_input_no:   # 8>5
                     # 패스중 5를 랜덤하게 샘플링 (https://jimmy-ai.tistory.com/287)      
                    numbers = np.random.choice(range(0, len(path)), 5, replace = False)
                    for i, number in enumerate(numbers): # e.g. [0, 4, 2, 3, 8]
                        self.path_as_input[i, :] = path[number, :] - 10.0
                                            
                # 만약 크기 같다면: 
                elif len(path) == self.path_as_input_no:
                    self.path_as_input = path - 10.0
            elif SCENARIO=='TD3':
                # 만약 path가 더 작다면: # 앞단의 패스 길이만큼으로 대치 (남는 뒷부분들은 init goals)
                if len(path) < self.path_as_input_no:
                    #print(path.shape, self.path_as_input.shape)   # 8, 2  5, 2
                    self.path_as_input[:len(path), :] = path-5.5
                
                # 만약 path가 더 길다면: # 패스중 5를 랜덤하게 샘플링 (https://jimmy-ai.tistory.com/287)      
                elif len(path) > self.path_as_input_no:   # 8>5
                    numbers = np.random.choice(range(0, len(path)), 5, replace = False)
                    for i, number in enumerate(numbers): # e.g. [0, 4, 2, 3, 8]
                        self.path_as_input[i, :] = path[number, :] - 5.5
                                    
                # 만약 크기 같다면: 
                elif len(path) == self.path_as_input_no:
                    self.path_as_input = path - 5.5
            elif SCENARIO=='U':
                # 만약 path가 더 작다면: # 앞단의 패스 길이만큼으로 대치 (남는 뒷부분들은 init goals)
                if len(path) < self.path_as_input_no:
                    #print(path.shape, self.path_as_input.shape)   # 8, 2  5, 2
                    self.path_as_input[:len(path), :] = path-5.5
                
                # 만약 path가 더 길다면: # 패스중 5를 랜덤하게 샘플링 (https://jimmy-ai.tistory.com/287)      
                elif len(path) > self.path_as_input_no:   # 8>5
                    numbers = np.random.choice(range(0, len(path)), 5, replace = False)
                    for i, number in enumerate(numbers): # e.g. [0, 4, 2, 3, 8]
                        self.path_as_input[i, :] = path[number, :] - 5.5
                                    
                # 만약 크기 같다면: 
                elif len(path) == self.path_as_input_no:
                    self.path_as_input = path - 5.5
                    
            elif SCENARIO=='DWA':
                # 만약 path가 더 작다면: # 앞단의 패스 길이만큼으로 대치 (남는 뒷부분들은 init goals)
                if len(path) < self.path_as_input_no:
                    #print(path.shape, self.path_as_input.shape)   # 8, 2  5, 2
                    self.path_as_input[:len(path), :] = path-5.5
                
                # 만약 path가 더 길다면:
                elif len(path) > self.path_as_input_no:   # 8>5
                    # 아래 샘플링 방법 중 3개 선택
                    
                    '''
                    # Sampling 1. 패스중 5를 랜덤하게 샘플링 (https://jimmy-ai.tistory.com/287)
                    numbers = np.random.choice(range(0, len(path)), 5, replace = False)
                    for i, number in enumerate(numbers): # e.g. [0, 4, 2, 3, 8]
                        #print('전',i, self.path_as_input[i, :])
                        self.path_as_input[i, :] = path[number, :] - 5.5
                        #print(i, self.path_as_input[i, :])
                        
                    '''

                    # 230127 Sampling 2. 패스중 5개를 uniform하게 샘플링
                    #print('오리지널 패스:',path)
                    divdiv = int(len(path) / self.path_as_input_no)   # e.g. 13/5 = 2.6 --int--> 2
                    self.path_as_input.astype(float)
                    for i in range(self.path_as_input_no):
                        #print(i, divdiv, len(path))
                        #print((i+1)*divdiv)
                        self.path_as_input[i,:] = (path[(i+1)*divdiv-1, :]-5.5)
                        #print('기본패스:',path[(i+1)*divdiv-1, :])
                        #print('bias패스:',path[(i+1)*divdiv-1, :]-5.5)
                        #print('입력휴패스:',self.path_as_input[i,:])   # 231107 integer로 바뀌어 들어가는 문제 식별
                        
                    
                    '''
                    # 230130 Sampling 3. 패스중 landmark selection
                    # landmark: 로봇이 이동간 가장 큰 변화를 해야 하는 곳(굴곡이 큰곳)
                    # visibility: 현재 cctv가 보고 있는곳
                    # near_human: 패스 주변 반경에 사람 수
                    #print('랜피스:',len(path))
                    top_k_list = []
                    for i, node in enumerate(path):
                        
                        def getAngle(a, b, c):
                            ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
                            return ang + 360 if ang < 0 else ang
                        if i!=0 and i<len(path)-1:
                            #print(i, getAngle(path[i-1],path[i],path[i+1]))
                            top_k_list.append(getAngle(path[i-1],path[i],path[i+1]))
                    #print(top_k_list)
                    if len(top_k_list)<=4:
                        top_k_list.append(top_k_list[-1])
                    #print('넨피:',np.argpartition(top_k_list, -5)[-5:])
                    order_by_desc = np.argpartition(top_k_list, -5)[-5:]   # https://stackoverflow.com/questions/65038206/how-to-get-indices-of-top-k-values-from-a-numpy-array
                    
                    for i, pros in enumerate(order_by_desc):
                        self.path_as_input[i,:] = path[pros,:]-5.5
                    '''       
 
                # 만약 크기 같다면: 
                elif len(path) == self.path_as_input_no:
                    self.path_as_input = path - 5.5
            if SCENARIO=='warehouse_RAL':   # 240205 for RA-L Rebuttal stage
                # 만약 path가 더 작다면: # 앞단의 패스 길이만큼으로 대치 (남는 뒷부분들은 init goals)
                if len(path) < self.path_as_input_no:
                    self.path_as_input[:len(path), :] = path-10.0

                # 만약 path가 더 길다면: 
                elif len(path) > self.path_as_input_no:   # 8>5
                    # 230127 패스중 5개를 uniform하게 샘플링
                    divdiv = int(len(path) / self.path_as_input_no)   # e.g. 13/5 = 2.6 --int--> 2
                    for i in range(self.path_as_input_no):
                        self.path_as_input[i,:] = path[(i+1)*divdiv-1, :]-10.0
                
                # 만약 크기 같다면: 
                elif len(path) == self.path_as_input_no:
                    self.path_as_input = path - 10.0                
           

            self.path_as_init = self.path_as_input
        ### TODO adaptiveavgpooling2D
        ### m = nn.AdpativeAvgPool1d(5)
        ### input1 = torch.rand(1, 64, 8)
        ### input2 = torch.rand(1, 64, 13)
        ### m(input1), m(input2) -> [1, 64, 5]
        
        #print('패스 애즈 이닛:',self.path_as_init)        
        #print('골:',self.goal_x, self.goal_y)
        #print('패스:',self.path_as_input)
        
        self.publish_markers(action)   # RVIZ 상 marker publish
        
        self.temp_path_as_input = copy.deepcopy(self.path_as_input)
        # 221019 self.path_as_input을 robot centric으로 변환     
        for i, p in enumerate(self.path_as_input):
            xx = p[0]
            yy = p[1]
            
            optimal_g_x = xx
            optimal_g_y = yy
            
            skew_xx = optimal_g_x - self.odom_x
            skew_yy = optimal_g_y - self.odom_y
            dot = skew_xx * 1 + skew_yy * 0
            mag1 = math.sqrt(math.pow(skew_xx, 2) + math.pow(skew_yy, 2))
            mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
            beta = math.acos(dot / ((mag1 * mag2)+0.000000001))
            if skew_yy < 0:
                if skew_xx < 0:
                    beta = -beta
                else:
                    beta = 0 - beta
            theta = beta - angle
            if theta > np.pi:
                theta = np.pi - theta
                theta = -np.pi - theta
            if theta < -np.pi:
                theta = -np.pi - theta
                theta = np.pi - theta
                
            distance = np.linalg.norm([self.odom_x - optimal_g_x, self.odom_y - optimal_g_y])
            
            self.temp_path_as_input[i][0] = distance
            self.temp_path_as_input[i][1] = theta
        #print(self.temp_path_as_input)
        self.temp_path_as_input = self.temp_path_as_input.reshape(-1,)
        
        ### 221014
        if self.sac_path:
        #if PATH_AS_INPUT:
            state = np.append(state, self.temp_path_as_input)
            #print('[step]self.temp_path_as_input:',self.temp_path_as_input)
        
        
        # 221110 integrity checker
        ## 잘못된 locomotion으로 robot이 하늘 위를 날아다니는 거를 체크해서 
        if self.last_odom.pose.pose.position.z > 0.05:   # 에러날 대 보면 0.12, 0.22, .24 막 이럼
            print('Error: Locomotion fail. 강제로 done = True')
            done = True
            
        if evaluate:   # 231025   매번 할때마다 안에 텍스트 지워야함
            #print('로봇 위치:',self.odom_x, self.odom_y)
            #print('사람 위치:',self.pedsim_agents_list_oracle_id)
        
            with open("robot_traj.txt", "a") as f:
                
                x, y = self.odom_x, self.odom_y
                f.write(f"{x}, {y}\n")
                
            with open("ped_traj.txt", "a") as f:
                
                x = self.pedsim_agents_list_oracle_id
                f.write(f"{x}\n")
        
        # DRL-VO Implication, 
        # 1. State 받아오기
        state_DRLVO = self._get_observation()
        #reward_DRLVO = self.get_reward_path_230206_VO(target, collision, action, self.goal_x, self.goal_y, self.pedsim_agents_list)
        reward_DRLVO = self.get_reward_path_230206_VO(target, collision, action, self.goal_x, self.goal_y, skew_x, skew_y, self.pre_distance, self.distance, laser_state, self.pedsim_agents_list)
        # 2. Reward 받아고기
        # 3. Done 받아오기

        #return state, reward, done, target
        return state_DRLVO, reward_DRLVO, done, target
    
    def _get_observation(self):
        #self.ped_pos = self.cnn_data.ped_pos_map
        self.ped_pos = self.ped_pos_map_tmp
        #self.scan = self.cnn_data.scan
        #self.scan = self.scan_sub
        #self.scan.append(self.scan_tmp.tolist())  
        self.scan = self.scan_tmp      
        self.goal = self.final_goal_pub
        self.vel = [self.last_odom.twist.twist.linear.x, self.last_odom.twist.twist.linear.z]
        #self.cnn_data.vel
        
        # ped map:
        # MaxAbsScaler:
        v_min = -2
        v_max = 2
        self.ped_pos = np.array(self.ped_pos, dtype=np.float32)
        self.ped_pos = 2 * (self.ped_pos - v_min) / (v_max - v_min) + (-1)

        # scan map:
        # MaxAbsScaler:
        temp = np.array(self.scan, dtype=np.float32)  # 전체 스캔 배열을 사용
        scan_avg = np.zeros((20, 80))
        for n in range(10):
            start_idx = n*180
            end_idx = (n+1)*180
            scan_tmp = temp[start_idx:end_idx] if end_idx <= len(temp) else temp[start_idx:]
            for i in range(80):
                # 슬라이스 길이 확인
                if len(scan_tmp[i*9:(i+1)*9]) > 0:
                    scan_avg[2*n, i] = np.min(scan_tmp[i*9:(i+1)*9])
                    scan_avg[2*n+1, i] = np.mean(scan_tmp[i*9:(i+1)*9])
                else:
                    # 길이가 0일 경우의 처리 (예: 0 할당)
                    scan_avg[2*n, i] = 0
                    scan_avg[2*n+1, i] = 0
        
        scan_avg = scan_avg.reshape(1600)
        scan_avg_map = np.matlib.repmat(scan_avg,1,4)
        self.scan = scan_avg_map.reshape(6400)
        s_min = 0
        s_max = 30
        self.scan = 2 * (self.scan - s_min) / (s_max - s_min) + (-1)        

        # goal:
        g_min = -2
        g_max = 2
        #self.goal = np.array(self.goal, dtype=np.float32)
        self.goal = np.array([self.goal_x, self.goal_y], dtype=np.float32)
        self.goal = 2 * (self.goal - g_min) / (g_max - g_min) + (-1)
        
        #print(self.ped_pos.shape, self.scan.shape, self.goal.shape)
    
        # observation:
        self.observation = np.concatenate((self.ped_pos, self.scan, self.goal), axis=None) #list(itertools.chain(self.ped_pos, self.scan, self.goal))
#        print('페드포스:',self.ped_pos, '스캔:',self.scan, '골:',self.goal)
        return self.observation    

    def IDR_checker(self):
        
        # for 사람 in 사람s
        # cal. L2 dist btwn robot and human
        IDR = 0.
        SD = 999.
        robot_p_x = self.last_odom.pose.pose.position.x
        robot_p_y = self.last_odom.pose.pose.position.y
        
        if self.pedsim_agents_list_oracle is not None:
            for actor in self.pedsim_agents_list_oracle:
                dist_robot_h = np.linalg.norm([actor[0] - robot_p_x, actor[1] - robot_p_y])
                #1. 매 스탭마다, 현재 로봇이 IS 안에 있는지 체크. 일단은 fixed된 1m
                if dist_robot_h <= 1.0:
                    IDR = 1.
                #2. cal. L2 dist btwn robot and human. 젤 작은거 return
                if dist_robot_h < SD:
                    SD = dist_robot_h
        return IDR, SD
        
        '''
        for actor in actors.agent_states:
            actor_id = str( actor.id )
            actor_pose = actor.pose
            #print("Spawning model: actor_id = %s", actor_id)
            x= actor_pose.position.x
            y= actor_pose.position.y
            #print(actor_id, x, y)
            actor_twist = actor.twist   # 230131
            vx = actor_twist.linear.x
            vy = actor_twist.linear.y
            #print(actor_id, vx, vy)
        '''
        
        return robot_p_x, robot_p_y
        
        #2. 매 스탭마다, 현재 로봇과 사람의 가장 가까운 거리 체크
  

    def reset(self):
        time.sleep(TIME_DELTA)  # 필요?
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        x = 0
        y = 0
        position_ok = False
        while not position_ok:
            if SCENARIO=='warehouse':
                x = np.random.uniform(-9.5, 9.5)
                y = np.random.uniform(-9.5, 9.5)
                position_ok = check_pos_warehouse(x, y)
            elif SCENARIO=='TD3':
                x = np.random.uniform(-4.5, 4.5)
                y = np.random.uniform(-4.5, 4.5)
                position_ok = check_pos(x, y)       
            elif SCENARIO=='U':
                x = np.random.uniform(-4.5, 4.5)
                y = np.random.uniform(-4.5, 4.5)
                position_ok = check_pos_U(x, y)    
            elif SCENARIO=='DWA':
                x = np.random.uniform(-4.5, 4.5)
                y = np.random.uniform(-4.5, 4.5)
                position_ok = check_pos_DWA(x, y)
            elif SCENARIO=='warehouse_RAL':  # 240205
                x = np.random.uniform(-9.5, 9.5)
                y = np.random.uniform(-9.5, 9.5)
                position_ok = check_pos_warehouse_RAL(x, y)
                
        
        if debug:
            position_ok = False
            
            while not position_ok:
                x = np.random.uniform(-5.5, 5.5)
                y = np.random.uniform(-5.5, 5.5)
                position_ok = check_pos_DWA_evaluate(x, y)
                
            ### DEBUG 용
            ###x = -4.5
            ###y = 4.5
        
        if evaluate:
            if SCENARIO=='warehouse_RAL':
                x = 0
                y = -9
            else:
                #x = 2.5         #Holding
                #y = -4.5
                #x = -4.5         #Circling
                #y = 4.5
                x = 4.5         #Holding RA-L Rebuttal
                y = 4.5
                
        
                
                
        # set a random goal in empty space in environment
        #self.change_goal()    # short-term 
        self.change_goal(x, y)    # 240212
        # randomly scatter boxes in the environment
        #self.random_box()   # 220919 dynamic obstacle 추가로 일단 해제
        
        if debug:
            goal_ok = False
            
            while not goal_ok:
                self.goal_x = -x + random.uniform(-1.5, 1.5)  
                self.goal_y = -y + random.uniform(-1.5, 1.5) # [-5, 5] -> [-10, 10]
                goal_ok = check_pos_DWA_evaluate(self.goal_x, self.goal_y)
                #print(self.goal_x, self.goal_y, goal_ok)
                
            ### DEBUG 용
            ###self.goal_x = 3.0
            ###self.goal_y = -4.0
                
                
        if evaluate:
            if SCENARIO=='warehouse_RAL':
                self.goal_x = 0
                self.goal_y = 9
            else:
                self.goal_x = 0    # holding
                self.goal_y = 4.5
                #self.goal_x = 3   # Circling
                #self.goal_y = -4
                self.goal_x = -1    # holding_RAL
                self.goal_y = -4.5
                
                
        #angle = np.random.uniform(-np.pi, np.pi)
        angle = np.arctan2(self.goal_y - y, self.goal_x - x)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state
            
            
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        #object_state.pose.position.z = 0.
        #object_state.pose.position.z = 0.2   ### 220923 cafe.world일때
        
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y
        (_, _, self.euler) = euler_from_quaternion([object_state.pose.orientation.x, object_state.pose.orientation.y, object_state.pose.orientation.z, object_state.pose.orientation.w])


            

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)
        
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        
        self.distOld = math.sqrt(math.pow(self.odom_x - self.goal_x, 2) + math.pow(self.odom_y - self.goal_y, 2))
        
        # 220915 cctv1 cv정보 받아옴
        #self.rgb_cv_cctv1 = self.GetRGBImageObservation()
        
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]
        
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )
        self.distance = distance

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / ((mag1 * mag2)+0.000000001))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta
            
        # 221019
        local_goal = self.get_local_goal(self.odom_x, self.odom_y, self.goal_x, self.goal_y, self.euler)
        #robot_state = [local_goal[0], local_goal[1], 0.0, 0.0]    # local_gx, local_gy
        robot_state = [distance, theta, 0.0, 0.0]   # 골까지의 거리, 로봇 현재 heading, init_v=0, init_w=0 (4개)
        
        state = np.append(laser_state, robot_state)  # laser 정보(20) + 로봇 state(4)

        #220928 최초 initial path 생성
        while True:
            try:
                if SCENARIO=='warehouse':
                    path = planner_warehouse.main(self.odom_x, self.odom_y, self.goal_x, self.goal_y, self.pedsim_agents_list)
                elif SCENARIO=='TD3':
                    path = planner.main(self.odom_x, self.odom_y, self.goal_x, self.goal_y, self.pedsim_agents_list)
                elif SCENARIO=='U':
                    path = planner_U.main(self.odom_x, self.odom_y, self.goal_x, self.goal_y, self.pedsim_agents_list)
                elif SCENARIO=='DWA':
                    #path = planner_DWA.main(self.odom_x, self.odom_y, self.goal_x, self.goal_y, self.pedsim_agents_list)
                    #path = planner_astar.main(self.odom_x, self.odom_y, self.goal_x, self.goal_y, self.pedsim_agents_list) 
                    
                    ######
                    map_bias = 5.5
                    resolution = 0.1
                    grid_size = 1.0  # [m]
                    robot_radius = 5.0  # [m]
                    sx = int((self.odom_x+map_bias)/resolution)
                    sy = int((self.odom_y+map_bias)/resolution)
                    gx = int((self.goal_x+map_bias)/resolution)
                    gy = int((self.goal_y+map_bias)/resolution)
                    
                    ox, oy = [], []
                    for i in range(0, 110):
                        ox.append(i)
                        oy.append(0)
                    for i in range(0, 110):
                        ox.append(110)
                        oy.append(i)
                    for i in range(0, 110):
                        ox.append(i)
                        oy.append(110)
                    for i in range(0, 110):
                        ox.append(0)
                        oy.append(i)
                        
                    for i in range(20, 40):
                        for j in range(45, 80):
                            ox.append(i)
                            oy.append(j)
                    for i in range(55, 75):
                        for j in range(30, 55):
                            ox.append(i)
                            oy.append(j)
                    for i in range(95, 110):
                        for j in range(0, 55):
                            ox.append(i)
                            oy.append(j)
                            
                    for i in range(75, 110):
                        ox.append(i)
                        oy.append(80)
                    
                    self.a_star = astar.pure_astar.AStarPlanner(ox, oy, grid_size, robot_radius)
                    if PURE_GP:
                        self.pedsim_agents_list = []   # Pure ASTAR global pallner하고 싶으면 주석 해제
                    rx, ry, flow_map = self.a_star.planning(sx, sy, gx, gy, self.odom_vx, angle, skew_x, skew_y, self.pedsim_agents_list)
                    self.flow_map = flow_map  # 230221
                    
                    final_path = []
                    for path in zip (rx, ry):
                        final_path.append([path[0], path[1]])
                    #final_path.reverse()
                    #print(final_path)
                    final_path = np.array(final_path)
                    final_path = final_path / 10
                    path = final_path
                    
                    #path = astar.pure_astar.main(self.odom_x, self.odom_y, self.goal_x, self.goal_y, self.pedsim_agents_list) 
                elif SCENARIO=='warehouse_RAL':   # 240205
                    ######
                    map_bias = 10.0
                    resolution = 0.1
                    grid_size = 1.0  # [m]
                    robot_radius = 10.0  # 5.0[m]   
                    sx = int((self.odom_x+map_bias)/resolution)
                    sy = int((self.odom_y+map_bias)/resolution)
                    gx = int((self.goal_x+map_bias)/resolution)
                    gy = int((self.goal_y+map_bias)/resolution)
                    

                    
                    def make_intgrid(data):
                        data = (data + map_bias) / resolution
                        return int(data)
                    
                    #### 장애물 define ###
                    ox, oy = [], []
                    ##1. 테두리 장애물 define ##
                    for i in range(make_intgrid(-10), make_intgrid(10)):  # 북쪽   # [0~200]
                        ox.append(i)
                        oy.append(make_intgrid(10))
                    for i in range(make_intgrid(-10), make_intgrid(10)):  # 동쪽
                        ox.append(make_intgrid(10))
                        oy.append(i)
                    for i in range(make_intgrid(-10), make_intgrid(10)):  # 남쪽
                        ox.append(i)
                        oy.append(make_intgrid(-10))
                    for i in range(make_intgrid(-10), make_intgrid(10)):  # 서쪽
                        ox.append(0)
                        oy.append(make_intgrid(-10))
                        
                    # 왼쪽사이드 장애물 (1)                       
                    for i in range(make_intgrid(-10), make_intgrid(-9)):
                        for j in range(make_intgrid(-5), make_intgrid(8)):
                            ox.append(i)
                            oy.append(j)
                    # 왼쪽하단 w계단 장애물 (2)
                    for i in range(make_intgrid(-10), make_intgrid(-1)):
                        for j in range(make_intgrid(-10), make_intgrid(-5)):
                            ox.append(i)
                            oy.append(j)
                    # 하단 길쭉이 장애물 (2-1)        
                    for i in range(make_intgrid(-1), make_intgrid(10)):
                        for j in range(make_intgrid(-10), make_intgrid(-9.7)):
                            ox.append(i)
                            oy.append(j)
                    # 우측하단 장애물 랙3개 (3)
                    for i in range(make_intgrid(2.2), make_intgrid(7.3)):
                        for j in range(make_intgrid(-9.7), make_intgrid(-5.9)):
                            ox.append(i)
                            oy.append(j)
                    # 왼쪽상단 팔레트 (4)
                    for i in range(make_intgrid(-7.5), make_intgrid(0)):
                        for j in range(make_intgrid(5.5), make_intgrid(7.5)):
                            ox.append(i)
                            oy.append(j)
                    # 왼쪽 팔레트 (5)
                    for i in range(make_intgrid(-9), make_intgrid(-4.6)):
                        for j in range(make_intgrid(1.0), make_intgrid(2.8)):
                            ox.append(i)
                            oy.append(j)                            
                    # 우측 팔레트 (6)
                    for i in range(make_intgrid(5.2), make_intgrid(7.7)):
                        for j in range(make_intgrid(0.7), make_intgrid(2.8)):
                            ox.append(i)
                            oy.append(j)                    
                    # 우측상단 박스 5개 (7)
                    for i in range(make_intgrid(2.5), make_intgrid(8.5)):
                        for j in range(make_intgrid(5.5), make_intgrid(6.0)):
                            ox.append(i)
                            oy.append(j)
                    # 컨트롤판넬 (8)
                    for i in range(make_intgrid(1.7), make_intgrid(3.1)):
                        for j in range(make_intgrid(0.8), make_intgrid(1.5)):
                            ox.append(i)
                            oy.append(j)
                    # 필라스 (9,10,11)
                    for i in range(make_intgrid(-4.2), make_intgrid(-3.6)):
                        for j in range(make_intgrid(-2.6), make_intgrid(-2.1)):
                            ox.append(i)
                            oy.append(j)                            
                    for i in range(make_intgrid(0.1), make_intgrid(0.6)):
                        for j in range(make_intgrid(-2.6), make_intgrid(-2.1)):
                            ox.append(i)
                            oy.append(j)                            
                    for i in range(make_intgrid(4.7), make_intgrid(5.3)):
                        for j in range(make_intgrid(-2.6), make_intgrid(-2.1)):
                            ox.append(i)
                            oy.append(j)                            
                    

                    self.a_star = astar.pure_astar_RAL.AStarPlanner(ox, oy, grid_size, robot_radius)
                    if PURE_GP:
                        self.pedsim_agents_list = []  
                    
                    #print(sx, sy, gx, gy, self.odom_vx, angle, skew_x, skew_y, self.pedsim_agents_list)
                    rx, ry, flow_map = self.a_star.planning(sx, sy, gx, gy, self.odom_vx, angle, skew_x, skew_y, self.pedsim_agents_list)
                    self.flow_map = flow_map 
                    
                    final_path = []
                    for path in zip (rx, ry):
                        final_path.append([path[0], path[1]])
                    
                    final_path = np.array(final_path)
                    final_path = final_path / map_bias
                    path = final_path

                break
            except:
                path = [[self.goal_x, self.goal_y]]
                path = np.asarray(path)   # 221103
                print('SCENARIO:',SCENARIO,'의 [reset]예외발생. path <<-- global_goal로 지정', path)
                break
        
        self.path_i_prev = path
        self.path_i_rviz = path
        
        
        #################################
        #### Waypoint Sampling Module ###
        #################################    
        ############# 221010 고정된 5 사이즈의 path output    self.path_as_input
        self.path_as_input = []
        for i in range(self.path_as_input_no):
            self.path_as_input.append([self.goal_x, self.goal_y])
        self.path_as_input = np.asarray(self.path_as_input, dtype=float)
        if SCENARIO=='warehouse':
            # 만약 path가 더 작다면: # 앞단의 패스 길이만큼으로 대치 (남는 뒷부분들은 init goals)
            if len(path) < self.path_as_input_no:
                #print(path.shape, self.path_as_input.shape)   # 8, 2  5, 2
                self.path_as_input[:len(path), :] = path-10.0
            
            #### 만약 path가 더 길다면: # 패스의 뒤에 5개 부분으로 대치  (8, 2)   
            ###elif len(path) > self.path_as_input_no:   # 8>5
            ###    self.path_as_input = path[-5:, :]-10.0
                
            # 221114
            # 만약 path가 더 길다면: # 패스중 5를 랜덤하게 샘플링 (https://jimmy-ai.tistory.com/287)      
            elif len(path) > self.path_as_input_no:   # 8>5
                numbers = np.random.choice(range(0, len(path)), 5, replace = False)
                for i, number in enumerate(numbers): # e.g. [0, 4, 2, 3, 8]
                    self.path_as_input[i, :] = path[number, :] - 10.0
            
            # 만약 크기 같다면: 
            elif len(path) == self.path_as_input_no:
                self.path_as_input = path - 10.0
        
        elif SCENARIO=='TD3':
            # 만약 path가 더 작다면: # 앞단의 패스 길이만큼으로 대치 (남는 뒷부분들은 init goals)
            if len(path) < self.path_as_input_no:
                #print(path.shape, self.path_as_input.shape)   # 8, 2  5, 2
                self.path_as_input[:len(path), :] = path-5.5
            
            # 221114
            # 만약 path가 더 길다면: # 패스중 5를 랜덤하게 샘플링 (https://jimmy-ai.tistory.com/287)      
            elif len(path) > self.path_as_input_no:   # 8>5
                numbers = np.random.choice(range(0, len(path)), 5, replace = False)
                for i, number in enumerate(numbers): # e.g. [0, 4, 2, 3, 8]
                    self.path_as_input[i, :] = path[number, :] - 5.5              
            
            # 만약 크기 같다면: 
            elif len(path) == self.path_as_input_no:
                self.path_as_input = path - 5.5
                
        elif SCENARIO=='U':
            # 만약 path가 더 작다면: # 앞단의 패스 길이만큼으로 대치 (남는 뒷부분들은 init goals)
            if len(path) < self.path_as_input_no:
                #print(path.shape, self.path_as_input.shape)   # 8, 2  5, 2
                self.path_as_input[:len(path), :] = path-5.5
            
            # 221114
            # 만약 path가 더 길다면: # 패스중 5를 랜덤하게 샘플링 (https://jimmy-ai.tistory.com/287)      
            elif len(path) > self.path_as_input_no:   # 8>5
                numbers = np.random.choice(range(0, len(path)), 5, replace = False)
                for i, number in enumerate(numbers): # e.g. [0, 4, 2, 3, 8]
                    self.path_as_input[i, :] = path[number, :] - 5.5              
            
            # 만약 크기 같다면: 
            elif len(path) == self.path_as_input_no:
                self.path_as_input = path - 5.5
                
        elif SCENARIO=='DWA':
            # 만약 path가 더 작다면: # 앞단의 패스 길이만큼으로 대치 (남는 뒷부분들은 init goals)
            if len(path) < self.path_as_input_no:
                #print(path.shape, self.path_as_input.shape)   # 8, 2  5, 2
                self.path_as_input[:len(path), :] = path-5.5
            
            # 만약 path가 더 길다면:
            elif len(path) > self.path_as_input_no:   # 8>5
                '''
                # 패스중 5를 랜덤하게 샘플링 (https://jimmy-ai.tistory.com/287)      
                numbers = np.random.choice(range(0, len(path)), 5, replace = False)
                for i, number in enumerate(numbers): # e.g. [0, 4, 2, 3, 8]
                    self.path_as_input[i, :] = path[number, :] - 5.5
                '''
                
                # 230127 패스중 5개를 uniform하게 샘플링
                divdiv = int(len(path) / self.path_as_input_no)   # e.g. 13/5 = 2.6 --int--> 2
                for i in range(self.path_as_input_no):
                    #print(i, divdiv, len(path))
                    #print((i+1)*divdiv)
                    self.path_as_input[i,:] = path[(i+1)*divdiv-1, :]-5.5       
                
                '''
                # 230130 Sampling 3. 패스중 landmark selection
                # landmark: 로봇이 이동간 가장 큰 변화를 해야 하는 곳(굴곡이 큰곳)
                # visibility: 현재 cctv가 보고 있는곳
                # near_human: 패스 주변 반경에 사람 수
                #print('랜피스:',len(path))
                top_k_list = []
                for i, node in enumerate(path):
                    
                    def getAngle(a, b, c):
                        ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
                        return ang + 360 if ang < 0 else ang
                    if i!=0 and i<len(path)-1:
                        #print(i, getAngle(path[i-1],path[i],path[i+1]))
                        top_k_list.append(getAngle(path[i-1],path[i],path[i+1]))
                #print(top_k_list)
                if len(top_k_list)<=4:
                        top_k_list.append(top_k_list[-1])
                #print('넨피:',np.argpartition(top_k_list, -5)[-5:])
                order_by_desc = np.argpartition(top_k_list, -5)[-5:]
                
                for i, pros in enumerate(order_by_desc):
                    self.path_as_input[i,:] = path[pros,:]-5.5                    
                '''  
            
            # 만약 크기 같다면: 
            elif len(path) == self.path_as_input_no:
                self.path_as_input = path - 5.5
        if SCENARIO=='warehouse_RAL':   # 240205 for RA-L Rebuttal stage
            # 만약 path가 더 작다면: # 앞단의 패스 길이만큼으로 대치 (남는 뒷부분들은 init goals)
            if len(path) < self.path_as_input_no:
                self.path_as_input[:len(path), :] = path-10.0

            # 만약 path가 더 길다면: 
            elif len(path) > self.path_as_input_no:   # 8>5
                # 230127 패스중 5개를 uniform하게 샘플링
                divdiv = int(len(path) / self.path_as_input_no)   # e.g. 13/5 = 2.6 --int--> 2
                for i in range(self.path_as_input_no):
                    self.path_as_input[i,:] = path[(i+1)*divdiv-1, :]-10.0
            
            # 만약 크기 같다면: 
            elif len(path) == self.path_as_input_no:
                self.path_as_input = path - 10.0                



        self.path_as_init = copy.deepcopy(self.path_as_input)     # raw global path
        #print('[reset]path_as_init:',self.path_as_init)
        
        self.publish_markers([0.0, 0.0])

        self.temp_path_as_input = self.path_as_input
        # 221019 self.path_as_input을 robot centric으로 변환     
        for i, p in enumerate(self.path_as_input):
            xx = p[0]
            yy = p[1]
            
            optimal_g_x = xx
            optimal_g_y = yy
            
            skew_xx = optimal_g_x - self.odom_x
            skew_yy = optimal_g_y - self.odom_y
            dot = skew_xx * 1 + skew_yy * 0
            mag1 = math.sqrt(math.pow(skew_xx, 2) + math.pow(skew_yy, 2))
            mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
            beta = math.acos(dot / ((mag1 * mag2)+0.000000001))
            if skew_yy < 0:
                if skew_xx < 0:
                    beta = -beta
                else:
                    beta = 0 - beta
            theta = beta - angle
            if theta > np.pi:
                theta = np.pi - theta
                theta = -np.pi - theta
            if theta < -np.pi:
                theta = -np.pi - theta
                theta = np.pi - theta
            
            distance = np.linalg.norm([self.odom_x - optimal_g_x, self.odom_y - optimal_g_y])
            
            self.temp_path_as_input[i][0] = distance
            self.temp_path_as_input[i][1] = theta
                
        self.temp_path_as_input = self.temp_path_as_input.reshape(-1,)

        
        # 221014
        if self.sac_path:
        #if PATH_AS_INPUT:
            state = np.append(state, self.temp_path_as_input)
            #print('[reset]self.temp_path_as_input:',self.temp_path_as_input)
            
        state_DRLVO = self._get_observation()
            
        #return state
        return state_DRLVO

    #def change_goal(self):   # adaptive goal positioning
    def change_goal(self, x, y):   # 240212
        # Place a new goal and check if its location is not on one of the obstacles
        #if self.upper < 10:    # 5
        if self.upper < 5:  # 221108
            self.upper += 0.004
        #if self.lower > -10:   # -5
        if self.lower > -5: # 221108
            self.lower -= 0.004

        goal_ok = False

        while not goal_ok:
            #self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)  
            #self.goal_y = self.odom_y + random.uniform(self.upper, self.lower) # [-5, 5] -> [-10, 10]
            self.goal_x = x + random.uniform(self.upper, self.lower)  
            self.goal_y = y + random.uniform(self.upper, self.lower) # [-5, 5] -> [-10, 10]
            if SCENARIO=='warehouse':
                goal_ok = check_pos_warehouse(self.goal_x, self.goal_y)
            elif SCENARIO=='TD3': 
                goal_ok = check_pos(self.goal_x, self.goal_y)
            elif SCENARIO=='U':
                goal_ok = check_pos_U(self.goal_x, self.goal_y)
            elif SCENARIO=='DWA':
                goal_ok = check_pos_DWA(self.goal_x, self.goal_y)
            elif SCENARIO=='warehouse_RAL':   # 240205
                goal_ok = check_pos_warehouse_RAL(self.goal_x, self.goal_y)
                

    def random_box(self):
        # Randomly change the location of the boxes in the environment on each reset to randomize the training
        # environment
        for i in range(4):
            name = "cardboard_box_" + str(i)
            
            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y)
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                    box_ok = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)
        
    def publish_markers(self, action):
        # Publish visual data in Rviz
        # marker = init goal pose
        marker_lifetime = rospy.Duration(1.0)

        
        # flow_map from pure_astar visualize
        if viz_flow_map:
            markerArray7 = MarkerArray()
            if self.flow_map is None:
                pass
            else:
                #print('self.flow_map:',self.flow_map)
                #print(self.flow_map.shape)
                marker_id = 0
                for i in range(self.flow_map.shape[0]):
                    for j in range(self.flow_map.shape[1]):
                        marker7 = Marker()
                        marker7.action = Marker.DELETE
                        #marker7.lifetime = marker_lifetime
                        marker7.id = marker_id #(i*10)+j
                        #print(marker7.id)
                        marker7.header.frame_id = "odom"
                        marker7.type = marker7.CYLINDER
                        marker7.action = marker7.ADD
                        marker7.scale.x = 0.1
                        marker7.scale.y = 0.1
                        marker7.scale.z = 0.01
                        marker7.color.a = 0.51
                        marker7.color.g = 0.5
                        marker7.color.b = 0.5
                        #print(self.flow_map[i][j], self.flow_map[i][j] + 99, (self.flow_map[i][j] + 99)/198)
                        if self.flow_map[i][j] == 0.0:
                            marker7.color.r = 0.5
                        else:
                            ratio = self.flow_map[i][j] / 99
                            #print('색깔:',self.flow_map[i][j])
                            #marker7.color.r = (self.flow_map[i][j]+98)/198
                            #print(i,'라티오:',ratio)
                            marker7.color.r = ratio
                            marker7.color.g = 1-ratio
                            marker7.color.b = 0.0
                            
                        #marker7.color.g = 0.8
                        #marker7.color.b = 0.7
                        
                        marker7.pose.orientation.w = 1.0
                        if SCENARIO=='warehouse':
                            marker7.pose.position.x = i - 10
                            marker7.pose.position.y = j - 10 
                        elif SCENARIO=='TD3':
                            marker7.pose.position.x = i - 5.5
                            marker7.pose.position.y = j - 5.5
                        elif SCENARIO=='U':
                            marker7.pose.position.x = i - 5.5
                            marker7.pose.position.y = j - 5.5
                        elif SCENARIO=='DWA':
                            marker7.pose.position.x = i/10 - 5.5
                            marker7.pose.position.y = j/10 - 5.5
                        elif SCENARIO=='warehouse_RAL':   # 240205
                            marker7.pose.position.x = i/10 - 10
                            marker7.pose.position.y = j/10 - 10                     
                        #print([i,j],'의:',[marker7.pose.position.x,marker7.pose.position.y],marker7.pose.position.z)
                        marker7.pose.position.z = (self.flow_map[i][j] + 99)/198

                        markerArray7.markers.append(marker7)
                        marker_id += 1

            self.publisher7.publish(markerArray7)
        
        
        
        
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(action[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)
        #self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0

        markerArray3.markers.append(marker3)
        #self.publisher3.publish(markerArray3)
        # global trajectory
        markerArray4 = MarkerArray()
        #print('self.pre_i:',self.path_i_prev)
        for i, pose in enumerate(self.path_i_rviz):
    
            marker4 = Marker()
            marker4.id = i
            marker4.header.frame_id = "odom"
            marker4.action = Marker.DELETE
            marker4.lifetime = marker_lifetime 
            marker4.type = marker4.CYLINDER
            marker4.action = marker4.ADD
            marker4.scale.x = 0.1
            marker4.scale.y = 0.1
            marker4.scale.z = 0.01
            marker4.color.a = 0.51
            marker4.color.r = 0.87
            marker4.color.g = 1.0
            marker4.color.b = 0.0
            marker4.pose.orientation.w = 1.0
            if SCENARIO=='warehouse':
                marker4.pose.position.x = pose[0] - 10
                marker4.pose.position.y = pose[1] - 10 
            elif SCENARIO=='TD3':
                marker4.pose.position.x = pose[0] - 5.5
                marker4.pose.position.y = pose[1] - 5.5
            elif SCENARIO=='U':
                marker4.pose.position.x = pose[0] - 5.5
                marker4.pose.position.y = pose[1] - 5.5
            elif SCENARIO=='DWA':
                marker4.pose.position.x = pose[0] - 5.5
                marker4.pose.position.y = pose[1] - 5.5
            elif SCENARIO=='warehouse_RAL':   # 240205
                marker4.pose.position.x = pose[0] - 10
                marker4.pose.position.y = pose[1] - 10
            marker4.pose.position.z = 0

            markerArray4.markers.append(marker4)

        self.publisher4.publish(markerArray4)
        
        
        # optimal goal
        markerArray5 = MarkerArray()
        marker5 = Marker()
        marker5.header.frame_id = "odom"
        marker5.type = marker.CYLINDER
        marker5.action = marker.ADD
        marker5.scale.x = 0.15
        marker5.scale.y = 0.15
        marker5.scale.z = 0.01
        marker5.color.a = 1.0
        marker5.color.r = 0.1
        marker5.color.g = 0.1
        marker5.color.b = 1.0
        marker5.pose.orientation.w = 1.0
        marker5.pose.position.x = self.optimal_gx
        marker5.pose.position.y = self.optimal_gy
        marker5.pose.position.z = 0

        markerArray5.markers.append(marker5)

        self.publisher5.publish(markerArray5)
        
        # 221020
        # 5 optimal path (waypoints)
        markerArray6 = MarkerArray()
        for i, pose in enumerate(self.path_as_input):
    
            marker6 = Marker()
            marker6.id = i
            marker6.header.frame_id = "odom"
            marker6.type = marker.CYLINDER
            marker6.action = marker.ADD
            marker6.scale.x = 0.1
            marker6.scale.y = 0.1
            marker6.scale.z = 0.03
            marker6.color.a = 1.0
            marker6.color.r = 1.0
            marker6.color.g = 0.1
            marker6.color.b = 0.1
            marker6.pose.orientation.w = 1.0
            #print(self.get_reliablity(self.path_as_input, PARTIAL_VIEW)[i])
            #if self.get_reliablity(self.path_as_input, PARTIAL_VIEW)[i] == [1.0]:
            #    marker6.color.r = 0.0
            #    marker6.color.g = 0.5
            #    marker6.color.b = 1.0
                
            marker6.pose.position.x = pose[0] 
            marker6.pose.position.y = pose[1] 
            marker6.pose.position.z = 0

            markerArray6.markers.append(marker6)

        self.publisher6.publish(markerArray6)
        
        # static map # 231107
        markerArray8 = MarkerArray()
        for i in range(4):
            marker8 = Marker()
            marker8.header.frame_id = "odom"
            marker8.ns = "square"
            marker8.id = i
            marker8.type = marker.CUBE
            marker5.action = marker.ADD
            marker8.lifetime = rospy.Duration()
            marker8.color.a = 1.0
            marker8.color.r = 0.1
            marker8.color.g = 0.1
            marker8.color.b = 0.1
            marker8.pose.orientation.w = 1.0
            if i==0:
                marker8.scale.x = 2.0
                marker8.scale.y = 3.5
                marker8.scale.z = 0.1
                marker8.pose.position.x = -2.5
                marker8.pose.position.y = 0.75
                markerArray8.markers.append(marker8)
            elif i==1:
                marker8.scale.x = 2.0
                marker8.scale.y = 2.5
                marker8.scale.z = 0.1
                marker8.pose.position.x = 1.0
                marker8.pose.position.y = -1.25
                markerArray8.markers.append(marker8)
            elif i==2:
                marker8.scale.x = 1.5
                marker8.scale.y = 5.5
                marker8.scale.z = 0.1
                marker8.pose.position.x = 4.75
                marker8.pose.position.y = -2.75
                markerArray8.markers.append(marker8)
            elif i==3:
                marker8.scale.x = 3.5
                marker8.scale.y = 0.2
                marker8.scale.z = 0.1
                marker8.pose.position.x = 3.75
                marker8.pose.position.y = 2.5           
                marker8.pose.position.z = 0
                markerArray8.markers.append(marker8)

            

        self.publisher8.publish(markerArray8)
        
    
    # 220920    
    def harris_corder_detector(self, rgb_cv_cctv1):
        # ref: https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=samsjang&logNo=220637582963
        rooming = cv2.cvtColor(rgb_cv_cctv1, cv2.COLOR_BGR2GRAY)   
        rooming2 = rgb_cv_cctv1.copy()
        cv2.imshow('raw', rooming)
        
        rooming = np.float32(rooming)  
        dst = cv2.cornerHarris(rooming, 2, 3, 0, 0.04) 
        dst = cv2.dilate(dst, None)    
        rooming2[dst>0.01*dst.max()]=[0,0,255]   
        
        cv2.imshow('Harris', rooming2)
        cv2.waitKey(1)
    
    # 221019 로봇 egocentric local goal
    def get_local_goal(self, odom_x, odom_y, global_x, global_y, theta):
        local_x = (global_x - odom_x) * np.cos(theta) + (global_y - odom_y) * np.sin(theta)
        local_y = -(global_x - odom_x) * np.sin(theta) + (global_y - odom_y) * np.cos(theta)   # relative robot aspect to goal(local goal)
        
        return [local_x, local_y]
        

    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:   # 0.35
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser):
        if target:
            return 100.0
        elif collision:
            return -100.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2
    
    @staticmethod     # 221014  waypoint에 가까워 지면 sparse reward
    def get_reward_path(target, collision, action, min_laser, odom_x, odom_y, path_as_input, goal_x, goal_y):
        reward_w_sum = 0.0    # 각 웨이포인트별 먼 거리 penalty
        realibility = 1.0

        for i, path in enumerate(path_as_input):
            #print(i, path)
            dist_waypoint_goal = np.sqrt((goal_x - path[0])**2+(goal_y-path[1])**2)
            dist_robot_goal = np.sqrt((goal_x - odom_x)**2+(goal_y - odom_y)**2)
            dist_waypoint_robot = np.sqrt((path[0] - odom_x)**2 + (path[1] - odom_y)**2)
            if (dist_robot_goal > dist_waypoint_goal or dist_waypoint_goal < 0.11) and dist_waypoint_robot < 0.35:   # 로봇보다 골에 가까운 웨이포인트가 남아있을 경우
                reward_w_sum += 0.5 *realibility               # 완화값 * 신뢰도 * 로봇-웨이포인트 거리(로봇이 웨이포인트와 멀리 있음 페널티)
        if target:
            return 100.0
        elif collision:
            return -100.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2 - reward_w_sum  
    
    @staticmethod     
    def get_reward_path_230203_TD3(target, collision, action, min_laser, odom_x, odom_y, path_as_input, goal_x, goal_y, pre_dist, dist, pre_odom_x, pre_odom_y):
        # 1. Navigation reward
        R_g = 0.0
        R_c = 0.0
        R_move = 0.0
        if target:
            R_g = 100.0
        elif collision:
            R_c = -100.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            R_move = action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2
        R_nav = R_g + R_c + R_move
        # 2. Waypoint reward (dense)  
        R_w = 0.0
        num_valid_wpt = 0
        for i, path in enumerate(path_as_input):
            dist_waypoint_goal = np.sqrt((goal_x - path[0])**2+(goal_y-path[1])**2)
            dist_robot_goal = np.sqrt((goal_x - odom_x)**2+(goal_y - odom_y)**2)
            dist_waypoint_robot = np.sqrt((path[0] - odom_x)**2 + (path[1] - odom_y)**2)
            if (dist_robot_goal > dist_waypoint_goal or i==4):   # candidate waypoint 선정
                num_valid_wpt += 1
                pre_dist_wpt = np.sqrt((path[0] - pre_odom_x)**2 + (path[1] - pre_odom_y)**2)
                cur_dist_wpt = np.sqrt((path[0] - odom_x)**2 + (path[1] - odom_y)**2)
                diff_dist_wpt = pre_dist_wpt - cur_dist_wpt
                if diff_dist_wpt >= 0:
                    reward = 2 * diff_dist_wpt
                else:
                    reward = diff_dist_wpt
                                
                R_w += reward
        R_w = R_w / (num_valid_wpt + 0.000000000001)
        
        # total
        R_t = R_nav + R_w
        #print('tot:',R_t, 'R_nav:',R_nav, 'R_w:',R_w)
        
        return R_t
    
    @staticmethod     
    def get_reward_path_230214_realibility(target, collision, action, min_laser, odom_x, odom_y, path_as_input, goal_x, goal_y, pre_dist, dist, pre_odom_x, pre_odom_y, relliability_score):
        ## 221101 reward design main idea: R_guldering 참조
        # R = R_g + R_wpt + R_o + R_v
        R_g = 0.0
        R_c = 0.0
        R_p = 0.0
        R_w = 0.0
        R_laser = 0.0
        R_t = 0.0  # total
        # 1. Success
        if target:
            R_g = 100.0
            
        # 2. Collision
        if collision:
            R_c = -100
            
        # 3. Progress
        R_p = pre_dist - dist
        
        # 4. Waypoint
        '''
        ##### sparse waypoint reward ######
        ##### waypoint에 0.35 영역 내부에 있으면(존재하는 중이면) reward
        for i, path in enumerate(path_as_input):
            #print(i, path)
            dist_waypoint_goal = np.sqrt((goal_x - path[0])**2+(goal_y-path[1])**2)
            dist_robot_goal = np.sqrt((goal_x - odom_x)**2+(goal_y - odom_y)**2)
            dist_waypoint_robot = np.sqrt((path[0] - odom_x)**2 + (path[1] - odom_y)**2)
            #print(i, dist_robot_goal, dist_waypoint_goal, dist_robot_goal>dist_waypoint_goal, dist_waypoint_robot, dist_waypoint_robot<0.35)
            if (dist_robot_goal > dist_waypoint_goal or i==4) and dist_waypoint_robot < 0.35:   # 로봇보다 골에 가까운 웨이포인트가 남아있을 경우
                #print(i,'번째 대상 웨이포인트',path, dist_waypoint_robot)
                #reward_w_sum += 0.5 *realibility               # 완화값 * 신뢰도 * 로봇-웨이포인트 거리(로봇이 웨이포인트와 멀리 있음 페널티)
                reward = 0.35 - dist_waypoint_robot   # 221101
                R_w += reward * realibility
        '''
                
        ##### 230214 dense waypoint reward with reliability #####
        num_valid_wpt = 0
        for i, path in enumerate(path_as_input):
            dist_waypoint_goal = np.sqrt((goal_x - path[0])**2+(goal_y-path[1])**2)
            dist_robot_goal = np.sqrt((goal_x - odom_x)**2+(goal_y - odom_y)**2)
            if (dist_robot_goal > dist_waypoint_goal or i==4):   # candidate waypoint 선정
                num_valid_wpt += 1
                pre_dist_wpt = np.sqrt((path[0] - pre_odom_x)**2 + (path[1] - pre_odom_y)**2)
                cur_dist_wpt = np.sqrt((path[0] - odom_x)**2 + (path[1] - odom_y)**2)
                diff_dist_wpt = pre_dist_wpt - cur_dist_wpt
                if diff_dist_wpt >= 0:
                    reward = 2 * diff_dist_wpt
                else:
                    reward = diff_dist_wpt
                
                reward = reward * relliability_score[i][0]   # 221101
                #print(i, reward, relliability_score)
                
                R_w += reward
                #print(i,'번째 웨이포인트의 신뢰도:',relliability_score[i][0],'리워드:',reward)
        R_w = R_w / (num_valid_wpt + 0.000000000001)
        
        # total
        R_t = R_g + R_c + R_p + R_w
        #print('tot:',R_t, '골:',R_g, '충돌:',R_c, '전진:',R_p, '웨이포인트:',R_w,'(',num_valid_wpt,')', '레이저:',R_laser)
        #print('R_nav:',R_g+R_c+R_p, '  R_wpt:',R_w)
        return R_t    
    
    @staticmethod     
    def get_reward_path_230206_noreliability(target, collision, action, min_laser, odom_x, odom_y, path_as_input, goal_x, goal_y, pre_dist, dist, pre_odom_x, pre_odom_y):
        ## 221101 reward design main idea: R_guldering 참조
        # R = R_g + R_wpt + R_o + R_v
        R_g = 0.0
        R_c = 0.0
        R_p = 0.0
        R_w = 0.0
        R_laser = 0.0
        R_t = 0.0  # total
        # 1. Success
        if target:
            R_g = 100.0
            
        # 2. Collision
        if collision:
            R_c = -100
            
        # 3. Progress
        R_p = pre_dist - dist
        
        # 4. Waypoint
        '''
        ##### sparse waypoint reward ######
        ##### waypoint에 0.35 영역 내부에 있으면(존재하는 중이면) reward
        for i, path in enumerate(path_as_input):
            #print(i, path)
            dist_waypoint_goal = np.sqrt((goal_x - path[0])**2+(goal_y-path[1])**2)
            dist_robot_goal = np.sqrt((goal_x - odom_x)**2+(goal_y - odom_y)**2)
            dist_waypoint_robot = np.sqrt((path[0] - odom_x)**2 + (path[1] - odom_y)**2)
            #print(i, dist_robot_goal, dist_waypoint_goal, dist_robot_goal>dist_waypoint_goal, dist_waypoint_robot, dist_waypoint_robot<0.35)
            if (dist_robot_goal > dist_waypoint_goal or i==4) and dist_waypoint_robot < 0.35:   # 로봇보다 골에 가까운 웨이포인트가 남아있을 경우
                #print(i,'번째 대상 웨이포인트',path, dist_waypoint_robot)
                #reward_w_sum += 0.5 *realibility               # 완화값 * 신뢰도 * 로봇-웨이포인트 거리(로봇이 웨이포인트와 멀리 있음 페널티)
                reward = 0.35 - dist_waypoint_robot   # 221101
                R_w += reward * realibility
        '''
                
        ##### 221114 dense waypoint reward without reliability #####
        num_valid_wpt = 0
        for i, path in enumerate(path_as_input):
            dist_waypoint_goal = np.sqrt((goal_x - path[0])**2+(goal_y-path[1])**2)
            dist_robot_goal = np.sqrt((goal_x - odom_x)**2+(goal_y - odom_y)**2)
            if (dist_robot_goal > dist_waypoint_goal or i==4):   # candidate waypoint 선정
                num_valid_wpt += 1
                pre_dist_wpt = np.sqrt((path[0] - pre_odom_x)**2 + (path[1] - pre_odom_y)**2)
                cur_dist_wpt = np.sqrt((path[0] - odom_x)**2 + (path[1] - odom_y)**2)
                diff_dist_wpt = pre_dist_wpt - cur_dist_wpt
                if diff_dist_wpt >= 0:
                    reward = 2 * diff_dist_wpt
                else:
                    reward = diff_dist_wpt
                                
                R_w += reward
        R_w = R_w / (num_valid_wpt + 0.000000000001)
        
        # total
        R_t = R_g + R_c + R_p + R_w
        #print('tot:',R_t, '골:',R_g, '충돌:',R_c, '전진:',R_p, '웨이포인트:',R_w,'(',num_valid_wpt,')', '레이저:',R_laser)
        
        return R_t    

    # 230227 get_reward_path_230206_noreliability와 비교군. pure DRL, R_w 없음    
    @staticmethod     
    def get_reward_path_230206_pureDRL(target, collision, action, min_laser, odom_x, odom_y, path_as_input, goal_x, goal_y, pre_dist, dist, pre_odom_x, pre_odom_y):
        ## 221101 reward design main idea: R_guldering 참조
        R_g = 0.0
        R_c = 0.0
        R_p = 0.0
        R_t = 0.0  # total
        # 1. Success
        if target:
            R_g = 100.0
        # 2. Collision
        if collision:
            R_c = -100
        # 3. Progress
        R_p = pre_dist - dist
        # total
        R_t = R_g + R_c + R_p
        return R_t    
    
    @staticmethod     
    #def get_reward_path_230206_VO(target, collision, action, goal_x, goal_y, mht_peds):
    def get_reward_path_230206_VO(target, collision, action, goal_x, goal_y, skew_x, skew_y, pre_dist, dist, laser_state, mht_peds):
        ## 221101 reward design main idea: R_guldering 참조
        R_g = 0.0
        R_c = 0.0
        R_t = 0.0  # total
        R_th = 0.0
        R_p = 0.0
        w_thresh = 1
        r_rotation = -0.1
        r_scan = -0.2
        r_angle = 0.6
        angle_thresh = np.pi/6
        
        ls = np.array(laser_state[0])
        min_scan_dist = np.amin(ls[ls!=0])
        
        # 1. Success
        if target:
            R_g = 100.0
        # 2. Collision
        if collision:
            R_c = -100
        elif(min_scan_dist < 3*COLLISION_DIST):
            R_c = r_scan * (3*COLLISION_DIST - min_scan_dist)
        else:
            R_c = 0.0
        R_p = pre_dist - dist
        #. THETA REWARD
        # prefer goal theta:
        #theta_pre = np.arctan2(goal_y, goal_x)
        theta_pre = np.arctan2(skew_y, skew_x)   # 240220
        d_theta = theta_pre
        #print('레이저 스테이트:',laser_state)

        #print('최소 스캔 거리:',min_scan_dist)
        
        # get the pedstrain's position:
        if(mht_peds != None):  # tracker results
            d_theta = np.pi/2 #theta_pre
            N = 60
            theta_min = 1000
            for i in range(N):
                theta = random.uniform(-np.pi, np.pi)
                free = True
                for ped in mht_peds:
                    #ped_id = ped.track_id 
                    # create pedestrian's postion costmap: 10*10 m
                    p_x = ped[0]
                    p_y = ped[1]
                    p_vx = ped[2]
                    p_vy = ped[3]
                    
                    ped_dis = np.linalg.norm([p_x, p_y])
                    if(ped_dis <= 7):
                        ped_theta = np.arctan2(p_y, p_x)
                        vo_theta = np.arctan2(3*0.35, np.sqrt(ped_dis**2 - (3*0.35)**2))
                        # collision cone:
                        theta_rp = np.arctan2(action[0]*np.sin(theta)-p_vy, action[1]*np.cos(theta) - p_vx)
                        if(theta_rp >= (ped_theta - vo_theta) and theta_rp <= (ped_theta + vo_theta)):
                            free = False
                            break

                # reachable available theta:
                if(free):
                    theta_diff = (theta - theta_pre)**2
                    if(theta_diff < theta_min):
                        theta_min = theta_diff
                        d_theta = theta
                
        else: # no obstacles:
            d_theta = theta_pre

        R_th = r_angle*(angle_thresh - abs(d_theta))

        #print(R_th, mht_peds)
        # total
        R_t = R_g + R_p + R_c + R_th
        return R_t    


    @staticmethod     
    def get_reliablity(path_as_input, PARTIAL_VIEW):
        reliablity_scores = []
        reliability_score = 0.0
        
        for i, path in enumerate(path_as_input):
            #print(i, path)
            if PARTIAL_VIEW != True:    # GT
                reliability_score = 1.0
            else:                       # Partial view일경우 default로 invisible로 세팅
                reliability_score = 0.2

            #1. CCTV 영역에 노드가 위치할 경우 (Partial observation)
            if PARTIAL_VIEW and SCENARIO=='TD3' != True:  # TD3 환경일 경우
                if -5< path[1] <0:
                    reliability_score = 1.0
                    
            if PARTIAL_VIEW and SCENARIO=='warehouse':  # warehouse 환경일 경우
                if -10 < path[1] < 0:
                    reliability_score = 1.0
            if PARTIAL_VIEW and SCENARIO=='U':  # 
                if -1 < path[0] < 1 and 3 < path[1] < 5:
                    reliability_score = 1.0
            if PARTIAL_VIEW and SCENARIO=='DWA':
                #if (-5.5 <= path[0] <= -3.5 and -5.5 <= path[1] <= -1) or (-1.5 <= path[0] <= 0.0 and -1.0 <= path[1] <= 2.5) or (2.0 <= path[0] <= 4.0 and -5.5 <= path[1] <= 1.0):
                if (-5.5 <= path[0] <= 0.0 and -5.5 <= path[1] <= -1) or (-1.5 <= path[0] <= 0.0 and -5.5 <= path[1] <= 2.5) or (2.0 <= path[0] <= 4.0 and -5.5 <= path[1] <= 2.5):
                    reliability_score = 1.0   # GT
                else:
                    reliability_score = 0.2   # Partial view일경우 default로 invisible로 세팅
            if PARTIAL_VIEW and SCENARIO=='warehouse_RAL':   # 240205
                if -10 < path[1] < 0:
                    reliability_score = 1.0
                else:
                    reliability_score = 0.2
   
            #2. 로봇 영역에 노드가 위치할 경우 (TODO)   
                     
            reliablity_scores.append([reliability_score])

        return reliablity_scores

    def get_action_dwa(self, rx = -3.0, ry= -3.0, gx=4.0, gy=4.0):
        self.odom_x = self.last_odom.pose.pose.position.x   
        self.odom_y = self.last_odom.pose.pose.position.y 
        rx = self.odom_x
        ry = self.odom_y
        gx = self.goal_x
        gy = self.goal_y
        skew_x = gx - rx
        skew_y = gy - ry
        
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)
        (_, _, self.euler) = euler_from_quaternion([self.last_odom.pose.pose.orientation.x, self.last_odom.pose.pose.orientation.y, self.last_odom.pose.pose.orientation.z, self.last_odom.pose.pose.orientation.w])
      
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / ((mag1 * mag2)+0.000000001))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        temp_gx = gx
        temp_gy = gy
        threshold_reach = 1.0
        
        #print('sequenti waypoint:',self.seq_graph_path)
        ## seq_graph_path에서 intermediat goal 설정
        for i, path in enumerate(self.path_as_init):
            robot_to_goal = np.sqrt((gx-rx)**2+(gy-ry)**2)
            wpt_to_goal = np.sqrt((gx-path[0])**2+(gy-path[1])**2)
            robot_to_wpt = np.sqrt((path[0]-rx)**2+(path[1]-ry)**2)
            #print('로봇-골 거리:',robot_to_goal, '웨이포인트-골 거리:',wpt_to_goal)
            #print('현재 :',i,'번째의 패스 :',path)
            if wpt_to_goal < robot_to_goal and robot_to_wpt>threshold_reach:
                temp_gx = path[0]
                temp_gy = path[1]
                break
            
        #print('imediate goal:',temp_gx, temp_gy)
        
        ped_list = []
        
        if self.pedsim_agents_list is not None:
            for ped in self.pedsim_agents_list:
                ped_list.append([ped[0],ped[1]]) 
            
        #ped_list = self.pedsim_agents_list
        
        self.dwa_x[0] = rx
        self.dwa_x[1] = ry
        self.dwa_x[2] = angle
        self.dwa_x[3] = self.last_odom.twist.twist.linear.x
        self.dwa_x[4] = self.last_odom.twist.twist.angular.z

        ### dwa_pythonrobotics
        ## 1. global path의 waypoints들을 고려해서 이동 (dwa+GP)
        #uu, xx = dwa_pythonrobotics.main(rx, ry, temp_gx, temp_gy, angle, self.dwa_x, ped_list)   
        ## 2. fixed global goal만 고려 (naive dwa)
        #print('페드 리스트:',ped_list[:,:2])
        uu, xx = dwa_pythonrobotics.main(rx, ry, gx, gy, angle, self.dwa_x, ped_list)     
        
        diff_x = gx-rx
        diff_y = gy-ry
        to_go_rot = np.arctan2(diff_y, diff_x)
        return [uu[0], -uu[1]]    