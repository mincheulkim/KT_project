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
from os import path

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from pedsim_msgs.msg import AgentStates

from tf.transformations import euler_from_quaternion

# 220915 image 처리
from sensor_msgs.msg import Image, CameraInfo
import cv2
from cv_bridge import CvBridge, CvBridgeError    # CvBridge: connection btwn ROS and OpenCV interface

GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.35
TIME_DELTA = 0.1

DYNAMIC_GLOBAL = True  # 221003

#PATH_AS_INPUT = False # 221014
PATH_AS_INPUT = True # 221019

PLANNER_WAREHOUSE = True # 221102

consider_ped = False


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
        
    return goal_ok


class GazeboEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, environment_dim):   # launchfile = "multi_robot_scenario.launch", env_dim = 20
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
        self.human_num = 12
        self.pedsim_agents_distance = np.ones(self.human_num) * 99.0

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
        self.publisher5 = rospy.Publisher("optimal_goal", MarkerArray, queue_size=3)
        self.publisher6 = rospy.Publisher("path_as_init", MarkerArray, queue_size=10)   # 221020
        self.velodyne = rospy.Subscriber(
            "/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1
        )
        self.odom = rospy.Subscriber(
            "/r1/odom", Odometry, self.odom_callback, queue_size=1
        )
        
        self.bridge = CvBridge()
        self.rgb_image_size = [512, 512]
        ## TODO raw_image from CCTV1, CCTV2, or robot  
        # ref: https://github.com/xie9187/Monocular-Obstacle-Avoidance/blob/master/D3QN/GazeboWorld.py
        self.rgb_cctv1 = rospy.Subscriber("/cctv1/image_raw", Image, self.RGBImageCallBack)
        
        #220927 pedsim agent subscriber
        self.pedsim_agents = rospy.Subscriber("/pedsim_simulator/simulated_agents", AgentStates, self.actor_poses_callback)
        
        self.path_i_prev = None    # global path 저장
        self.path_rviz = None      # rviz에 보여줄 global path 저장
        self.path_as_input = []    # 221010
        self.path_as_input_no = 5  # 221010 5개 샘플
        self.path_as_init = None
        

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
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        break

    def odom_callback(self, od_data):
        self.last_odom = od_data
        
    # 220915 
    # ref: https://github.com/xie9187/Monocular-Obstacle-Avoidance/blob/master/D3QN/GazeboWorld.py
    def RGBImageCallBack(self, img):
        self.rgb_image = img
        
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
        for actor in actors.agent_states:
            actor_id = str( actor.id )
            actor_pose = actor.pose
            #print("Spawning model: actor_id = %s", actor_id)
            x= actor_pose.position.x
            y= actor_pose.position.y
            self.pedsim_agents_list.append([x,y])
            #print('액터:',actor_id,'model_pose:',x, y)
            
    def GetRGBImageObservation(self):
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
        
        
        # convert Opencv2 image to ROS image message and publish
        try:
            resized_img = self.bridge.cv2_to_imgmsg(cv_resized_img, "bgr8")
        except Exception as e:
            raise e
        #self.resized_rgb_img.publish(resized_img)   # 220915 주석 처리. resized된 이미지를 publish
        return(cv_resized_img)


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
        
        # 220927 사람 global 정보 받아옴
        if self.pedsim_agents_list != None:
            self.pedsim = self.pedsim_agents_list
            # 사람 rel_dist 구하기
            for i, ped in enumerate(self.pedsim):
                x = ped[0] - self.odom_x
                y = ped[1] - self.odom_y
                rel_dist =  math.sqrt(math.pow(x, 2) + math.pow(y, 2))
                self.pedsim_agents_distance[i] = rel_dist
        #print('pedsim_list:',self.pedsim_agents_distance)
        
        # read velodyne laser state
        done, collision, min_laser = self.observe_collision(self.velodyne_data)   # 0.35보다 min_laser 작으면 충돌, 아니면 return
        v_state = []
        v_state[:] = self.velodyne_data[:]   
        laser_state = [v_state]

        # Calculate robot heading from odometry data
        self.odom_x = self.last_odom.pose.pose.position.x   
        self.odom_y = self.last_odom.pose.pose.position.y 
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
        beta = math.acos(dot / (mag1 * mag2))
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
        # 220927
        '''
        if consider_ped:
            state = np.append(laser_state, robot_state)  # 20 + 4 + 12
            state = np.append(state, self.pedsim_agents_distance)  # 20 + 4 + 12
        '''
        self.path_as_input = copy.deepcopy(self.path_as_init)
        #print('[step]self.path_as_init;',self.path_as_init)   # global paoint에서 path as init 보여줌

        
        if PATH_AS_INPUT:
            #reward = self.get_reward_path(target, collision, action, min_laser, self.odom_x, self.odom_y, self.path_as_input, self.goal_x, self.goal_y)
            reward = self.get_reward_path_221101(target, collision, action, min_laser, self.odom_x, self.odom_y, self.path_as_input, self.goal_x, self.goal_y, self.pre_distance, self.distance, self.pre_odom_x, self.pre_odom_y)
        else:
            reward = self.get_reward(target, collision, action, min_laser)
        
        # 221005
        #reward = self.get_reward_new(target, collision, action, min_laser, self.pre_distance - self.distance)  # 221005
        
        #220928 path 생성
        # 정적 생성(option 1)
        if DYNAMIC_GLOBAL is not True:
            path = self.path_i_prev   # reset단에 최초로 생성된 global path
            self.path_i_rviz = path
        
        # 동적 생성(option 2)
        # try, excep: https://dojang.io/mod/page/view.php?id=2398
        #if DYNAMIC_GLOBAL:
        if DYNAMIC_GLOBAL and episode_steps%20 ==0:
            if PLANNER_WAREHOUSE:
                path = planner_warehouse.main(self.odom_x, self.odom_y, self.goal_x, self.goal_y, self.pedsim_agents_list)
            else:
                path = planner.main(self.odom_x, self.odom_y, self.goal_x, self.goal_y, self.pedsim_agents_list)  
            self.path_i_rviz = path
            '''
            while True:
                try:
                    if PLANNER_WAREHOUSE:
                        path = planner_warehouse.main(self.odom_x, self.odom_y, self.goal_x, self.goal_y, self.pedsim_agents_list)
                    else:
                        path = planner.main(self.odom_x, self.odom_y, self.goal_x, self.goal_y, self.pedsim_agents_list)  
                    self.path_i_rviz = path
                    break
                except:
                    print('예외발생[step]. path를 global goal로 지정')
                    path = [[self.goal_x, self.goal_y]]
                    path = np.asarray(path)   # 221103
                    self.path_i_rviz = path
                    break
            '''
            # TODO sampling 방법에 대해 고려
            #############################    
            ############# 221010 고정된 5 사이즈의 path output    self.path_as_input
            self.path_as_input = []
            for i in range(self.path_as_input_no):
                self.path_as_input.append([self.goal_x, self.goal_y])
                
            self.path_as_input = np.asarray(self.path_as_input)
            if PLANNER_WAREHOUSE:
                # 만약 path가 더 작다면: # 앞단의 패스 길이만큼으로 대치 (남는 뒷부분들은 init goals)
                if len(path) < self.path_as_input_no:
                    #print(path.shape, self.path_as_input.shape)   # 8, 2  5, 2
                    self.path_as_input[:len(path), :] = path-10.0
                
                # 만약 path가 더 길다면: # 패스의 뒤에 5개 부분으로 대치  (8, 2)   
                elif len(path) > self.path_as_input_no:   # 8>5
                    self.path_as_input = path[-5:, :]-10.0
                    
                
                # 만약 크기 같다면: 
                elif len(path) == self.path_as_input_no:
                    self.path_as_input = path - 10.0
            else:
                # 만약 path가 더 작다면: # 앞단의 패스 길이만큼으로 대치 (남는 뒷부분들은 init goals)
                if len(path) < self.path_as_input_no:
                    #print(path.shape, self.path_as_input.shape)   # 8, 2  5, 2
                    self.path_as_input[:len(path), :] = path-4.5
                
                # 만약 path가 더 길다면: # 패스의 뒤에 5개 부분으로 대치  (8, 2)   
                elif len(path) > self.path_as_input_no:   # 8>5
                    self.path_as_input = path[-5:, :]-4.5
                
                # 만약 크기 같다면: 
                elif len(path) == self.path_as_input_no:
                    self.path_as_input = path - 4.5
            self.path_as_init = self.path_as_input
        
        self.publish_markers(action)   # RVIZ 상 marker publish
        
        
        self.temp_path_as_input = copy.deepcopy(self.path_as_input)
        # 221019 self.path_as_input을 robot centric으로 변환     
        for i, p in enumerate(self.path_as_input):
            xx = p[0]
            yy = p[1]
            
            optimal_g_x = xx
            optimal_g_y = yy
            
            skew_x = optimal_g_x - self.odom_x
            skew_y = optimal_g_y - self.odom_y
            dot = skew_x * 1 + skew_y * 0
            mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
            mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
            beta = math.acos(dot / (mag1 * mag2))
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
                
            distance = np.linalg.norm([self.odom_x - optimal_g_x, self.odom_y - optimal_g_y])
            
            self.temp_path_as_input[i][0] = distance
            self.temp_path_as_input[i][1] = theta
        #print(self.temp_path_as_input)
        self.temp_path_as_input = self.temp_path_as_input.reshape(-1,)
        
        ### 221014
        if PATH_AS_INPUT:
            state = np.append(state, self.temp_path_as_input)
            #print('[step]self.temp_path_as_input:',self.temp_path_as_input)
        
        return state, reward, done, target

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state

        x = 0
        y = 0
        position_ok = False
        while not position_ok:
            if PLANNER_WAREHOUSE:
                x = np.random.uniform(-9.5, 9.5)
                y = np.random.uniform(-9.5, 9.5)
                position_ok = check_pos(x, y)
            else:
                x = np.random.uniform(-4.5, 4.5)
                y = np.random.uniform(-4.5, 4.5)
                position_ok = check_pos(x, y)
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

        # set a random goal in empty space in environment
        self.change_goal()
        # randomly scatter boxes in the environment
        #self.random_box()   # 220919 dynamic obstacle 추가로 일단 해제
        #self.publish_markers([0.0, 0.0])

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
        
        # 220927 pedsim 정보 받아옴
        if self.pedsim_agents_list != None:
            self.pedsim = self.pedsim_agents_list
            # 사람 rel_dist 구하기
            for i, ped in enumerate(self.pedsim):
                x = ped[0] - self.odom_x
                y = ped[1] - self.odom_y
                rel_dist =  math.sqrt(math.pow(x, 2) + math.pow(y, 2))
                self.pedsim_agents_distance[i] = rel_dist

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
        beta = math.acos(dot / (mag1 * mag2))

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
        # 220927
        if consider_ped:
            print('l:',laser_state)
            print('r:',robot_state)
            print('e:',self.pedsim_agents_distance)
            state = np.append(laser_state, robot_state)  # 20 + 4 + 12
            state = np.append(state, self.pedsim_agents_distance)  # 20 + 4 + 12
            
        #220928 최초 initial path 생성
        if PLANNER_WAREHOUSE:
            path = planner_warehouse.main(self.odom_x, self.odom_y, self.goal_x, self.goal_y, self.pedsim_agents_list)
        else:
            path = planner.main(self.odom_x, self.odom_y, self.goal_x, self.goal_y, self.pedsim_agents_list)
        '''
        try:
            if PLANNER_WAREHOUSE:
                path = planner_warehouse.main(self.odom_x, self.odom_y, self.goal_x, self.goal_y, self.pedsim_agents_list)
            else:
                path = planner.main(self.odom_x, self.odom_y, self.goal_x, self.goal_y, self.pedsim_agents_list)
        except:
            print('예외발생. path를 global_goal로 지정')
            path = [[self.goal_x, self.goal_y]]
            path = np.asarray(path)   # 221103
        '''
        
        self.path_i_prev = path
        self.path_i_rviz = path
        
        
        # TODO sampling 방법에 대해 고려
        #############################    
        ############# 221010 고정된 5 사이즈의 path output    self.path_as_input
        self.path_as_input = []
        for i in range(self.path_as_input_no):
            self.path_as_input.append([self.goal_x, self.goal_y])
        self.path_as_input = np.asarray(self.path_as_input)
        if PLANNER_WAREHOUSE:
            # 만약 path가 더 작다면: # 앞단의 패스 길이만큼으로 대치 (남는 뒷부분들은 init goals)
            if len(path) < self.path_as_input_no:
                #print(path.shape, self.path_as_input.shape)   # 8, 2  5, 2
                self.path_as_input[:len(path), :] = path-10.0
            
            # 만약 path가 더 길다면: # 패스의 뒤에 5개 부분으로 대치  (8, 2)   
            elif len(path) > self.path_as_input_no:   # 8>5
                self.path_as_input = path[-5:, :]-10.0
            
            # 만약 크기 같다면: 
            elif len(path) == self.path_as_input_no:
                self.path_as_input = path - 10.0
        
        else:
            # 만약 path가 더 작다면: # 앞단의 패스 길이만큼으로 대치 (남는 뒷부분들은 init goals)
            if len(path) < self.path_as_input_no:
                #print(path.shape, self.path_as_input.shape)   # 8, 2  5, 2
                self.path_as_input[:len(path), :] = path-4.5
            
            # 만약 path가 더 길다면: # 패스의 뒤에 5개 부분으로 대치  (8, 2)   
            elif len(path) > self.path_as_input_no:   # 8>5
                self.path_as_input = path[-5:, :]-4.5
            
            # 만약 크기 같다면: 
            elif len(path) == self.path_as_input_no:
                self.path_as_input = path - 4.5
        #self.path_as_init = []
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
            
            skew_x = optimal_g_x - self.odom_x
            skew_y = optimal_g_y - self.odom_y
            dot = skew_x * 1 + skew_y * 0
            mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
            mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
            beta = math.acos(dot / (mag1 * mag2))
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
                
            distance = np.linalg.norm([self.odom_x - optimal_g_x, self.odom_y - optimal_g_y])
            
            self.temp_path_as_input[i][0] = distance
            self.temp_path_as_input[i][1] = theta
                
        self.temp_path_as_input = self.temp_path_as_input.reshape(-1,)

        
        # 221014
        if PATH_AS_INPUT:
            state = np.append(state, self.temp_path_as_input)
            #print('[reset]self.temp_path_as_input:',self.temp_path_as_input)
        
        return state

    def change_goal(self):   # adaptive goal positioning
        # Place a new goal and check if its location is not on one of the obstacles
        if self.upper < 10:    # 5
            self.upper += 0.004
        if self.lower > -10:   # -5
            self.lower -= 0.004

        goal_ok = False

        while not goal_ok:
            self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)  
            self.goal_y = self.odom_y + random.uniform(self.upper, self.lower) # [-5, 5] -> [-10, 10]
            if PLANNER_WAREHOUSE:
                goal_ok = check_pos_warehouse(self.goal_x, self.goal_y)
            else: 
                goal_ok = check_pos(self.goal_x, self.goal_y)

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
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
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
        self.publisher2.publish(markerArray2)

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
        self.publisher3.publish(markerArray3)
        
        # global trajectory
        markerArray4 = MarkerArray()
        #print('self.pre_i:',self.path_i_prev)
        for i, pose in enumerate(self.path_i_rviz):
    
            marker = Marker()
            marker.id = i
            marker.header.frame_id = "odom"
            marker.type = marker.CYLINDER
            marker.action = marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.01
            marker.color.a = 0.5
            marker.color.r = 0.5
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.pose.orientation.w = 1.0
            if PLANNER_WAREHOUSE:
                marker.pose.position.x = pose[0] - 10
                marker.pose.position.y = pose[1] - 10 
            else:
                marker.pose.position.x = pose[0] - 4.5
                marker.pose.position.y = pose[1] - 4.5
            marker.pose.position.z = 0

            markerArray4.markers.append(marker)

        self.publisher4.publish(markerArray4)
        
        
        # optimal goal
        markerArray5 = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.1
        marker.color.g = 0.1
        marker.color.b = 1.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.optimal_gx
        marker.pose.position.y = self.optimal_gy
        marker.pose.position.z = 0
        #print(self.optimal_gx, self.optimal_gy)   # DEBUG

        markerArray5.markers.append(marker)

        self.publisher5.publish(markerArray5)
        
        # 221020
        # 5 optimal path
        markerArray6 = MarkerArray()
        #print('self.path_as_input:',self.path_as_init)
        for i, pose in enumerate(self.path_as_init):
    
            marker = Marker()
            marker.id = i
            marker.header.frame_id = "odom"
            marker.type = marker.CYLINDER
            marker.action = marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.03
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.1
            marker.color.b = 0.1
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = pose[0]
            marker.pose.position.y = pose[1]
            marker.pose.position.z = 0

            markerArray6.markers.append(marker)

        self.publisher6.publish(markerArray6)
    
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
        
    # 220928
    def get_global_action(self):

        self.odom_x = self.last_odom.pose.pose.position.x   
        self.odom_y = self.last_odom.pose.pose.position.y  
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)
        (_, _, self.euler) = euler_from_quaternion([self.last_odom.pose.pose.orientation.x, self.last_odom.pose.pose.orientation.y, self.last_odom.pose.pose.orientation.z, self.last_odom.pose.pose.orientation.w])  
        
        targetPose = np.array([self.goal_x-self.odom_x, self.goal_y-self.odom_y])
        
        inc_x = self.goal_x - self.odom_x
        inc_y = self.goal_y - self.odom_y
        
        angles = np.arctan2(inc_y,inc_x)
        diff = angles - self.euler
        if diff <= -np.pi:
            diff = np.pi + (angles - np.pi) - self.euler
        elif diff > np.pi:
            diff = -np.pi - (np.pi + angles) - self.euler
        
        diff -= np.pi
            
        length = np.sqrt(inc_y**2+ inc_x **2)
        linear_x = length
        angular_z = diff

        return linear_x, angular_z
    
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
        
    @staticmethod  # 221005
    def get_reward_new(target, collision, action, min_laser, delta_distance):
        '''
        if target:
            return 100.0
        elif collision:
            return -100.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2
        '''    
        ### 0. 리워드 initialize
        reward_g = delta_distance * 30
        reward_c = 0
        reward_w = 0
        
        ### 1. 도착했을 때
        if target:
            reward_g = 100
        ### 2. 충돌했을 때
        if collision:
            reward_c = -100
        ### 3. rotation penalty
        if np.abs(action[1]) > 1.05:
            reward_w = -0.1 * np.abs(action[1])
            print('rotation penalty!')
        ### 4. reminscure reward
        r3 = lambda x: 1 - x if x < 1 else 0.0
        reward_r = action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2
            
        reward = reward_g + reward_c + reward_w + reward_r

        return reward        
    
    
    @staticmethod     # 221014  waypoint에 가까워 지면 sparse reward
    def get_reward_path(target, collision, action, min_laser, odom_x, odom_y, path_as_input, goal_x, goal_y):
        reward_w_sum = 0.0    # 각 웨이포인트별 먼 거리 penalty
        reward_w = 0.0
        realibility = 1.0
        #path_as_input = path_as_input.reshape(-1,2)
        #print(path_as_input)
        #print('로봇 오돔:',odom_x, odom_y)
        #print('goal:',goal_x, goal_y)
        for i, path in enumerate(path_as_input):
            #print(i, path)
            dist_waypoint_goal = np.sqrt((goal_x - path[0])**2+(goal_y-path[1])**2)
            dist_robot_goal = np.sqrt((goal_x - odom_x)**2+(goal_y - odom_y)**2)
            dist_waypoint_robot = np.sqrt((path[0] - odom_x)**2 + (path[1] - odom_y)**2)
            #print(i, dist_robot_goal, dist_waypoint_goal, dist_robot_goal>dist_waypoint_goal, dist_waypoint_robot, dist_waypoint_robot<0.35)
            if (dist_robot_goal > dist_waypoint_goal or dist_waypoint_goal < 0.11) and dist_waypoint_robot < 0.35:   # 로봇보다 골에 가까운 웨이포인트가 남아있을 경우
                #print(i,'번째 대상 웨이포인트',path, dist_waypoint_robot)
                reward_w_sum += 0.5 *realibility               # 완화값 * 신뢰도 * 로봇-웨이포인트 거리(로봇이 웨이포인트와 멀리 있음 페널티)
        if target:
            return 100.0
        elif collision:
            return -100.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            #print('리워드:',action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2)
            #return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2
            #print(action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2, penalty_w_sum)
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2 - reward_w_sum
        
    @staticmethod     
    def get_reward_path_221101(target, collision, action, min_laser, odom_x, odom_y, path_as_input, goal_x, goal_y, pre_dist, dist, pre_odom_x, pre_odom_y):
        ## 221101 reward design main idea: R_guldering 참조
        # R = R_g + R_wpt + R_o + R_v
        R_g = 0.0
        R_c = 0.0
        R_p = 0.0
        R_w = 0.0
        R_laser = 0.0
        R_t = 0.0  # total
        num_valid_wpt = 0
        # 1. Success
        if target:
            R_g = 100.0
            
        # 2. Collision
        if collision:
            R_c = -100
            
        # 3. Progress
        R_p = pre_dist - dist
        
        # 4. Waypoint
        realibility = 1.0   # 보이면 1.0, 안보이면 0.2
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
                
        ##### dense waypoint reward #####
        for i, path in enumerate(path_as_input):
            #print(i, path)
            dist_waypoint_goal = np.sqrt((goal_x - path[0])**2+(goal_y-path[1])**2)
            dist_robot_goal = np.sqrt((goal_x - odom_x)**2+(goal_y - odom_y)**2)
            dist_waypoint_robot = np.sqrt((path[0] - odom_x)**2 + (path[1] - odom_y)**2)
            #print(i, dist_robot_goal, dist_waypoint_goal, dist_robot_goal>dist_waypoint_goal, dist_waypoint_robot, dist_waypoint_robot<0.35)
            if (dist_robot_goal > dist_waypoint_goal or i==4):   # candidate waypoint 선정
                num_valid_wpt += 1
                pre_dist_wpt = np.sqrt((path[0] - pre_odom_x)**2 + (path[1] - pre_odom_y)**2)
                cur_dist_wpt = np.sqrt((path[0] - odom_x)**2 + (path[1] - odom_y)**2)
                diff_dist_wpt = pre_dist_wpt - cur_dist_wpt
                #print(i, diff_dist_wpt)
                #print(i,'번째 대상 웨이포인트',path, dist_waypoint_robot)
                if diff_dist_wpt >= 0:
                    reward = 2 * diff_dist_wpt
                else:
                    reward = diff_dist_wpt
                
                reward = reward * realibility   # 221101
                R_w += reward * realibility
        R_w = R_w / num_valid_wpt
        
        # 5. Laser distance
        if min_laser < 0.5:
            R_laser = min_laser - 0.5
        
        # total
        R_t = R_g + R_c + R_p + R_p + R_w + R_laser
        #print('tot:',R_t, '골:',R_g, '충돌:',R_c, '전진:',R_p, '웨이포인트:',R_w,'(',num_valid_wpt,')', '레이저:',R_laser)
        dx = odom_x - pre_odom_x
        dy = odom_y - pre_odom_y
        

        return R_t
