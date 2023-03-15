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
import planner_DWA
import planner_astar # 230131
import astar.pure_astar  # 230213
from os import path

import dwa_pythonrobotics as dwa_pythonrobotics
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

DYNAMIC_GLOBAL = True  # 221003    # global path replanning과 관련

#PATH_AS_INPUT = False # 221014      # false
#PATH_AS_INPUT = True # 221019      # waypoint(5개)를 input으로 쓸것인지 결정

PARTIAL_VIEW = True ## 221114 TD3(아래쪽 절반), warehouse(아래쪽 절반) visible

SCENARIO = 'DWA'    # TD3, warehouse, U, DWA

debug = False    # evaluate단에서 활성화할 시 시점과 종점을 대칭으로 생성해줌


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

def check_pos_DWA(x, y):   # 230126
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
            #print(actor_id, x, y)
            actor_twist = actor.twist   # 230131
            vx = actor_twist.linear.x
            vy = actor_twist.linear.y
            #print(actor_id, vx, vy)
            
            
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

        #print('페드심 리스트: ', self.pedsim_agents_list)
            
            
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

        
        #if PATH_AS_INPUT:
        if self.sac_path:
            #reward = self.get_reward_path(target, collision, action, min_laser, self.odom_x, self.odom_y, self.path_as_input, self.goal_x, self.goal_y)
            reward = self.get_reward_path_230214_realibility(target, collision, action, min_laser, self.odom_x, self.odom_y, self.path_as_input, self.goal_x, self.goal_y, self.pre_distance, self.distance, self.pre_odom_x, self.pre_odom_y, reliability_score)
            #reward = self.get_reward_path_230206_noreliability(target, collision, action, min_laser, self.odom_x, self.odom_y, self.path_as_input, self.goal_x, self.goal_y, self.pre_distance, self.distance, self.pre_odom_x, self.pre_odom_y)  # 230206
            #reward = self.get_reward(target, collision, action, min_laser) # 230205 for test
            #reward = self.get_reward_path_230203_TD3(target, collision, action, min_laser, self.odom_x, self.odom_y, self.path_as_input, self.goal_x, self.goal_y, self.pre_distance, self.distance, self.pre_odom_x, self.pre_odom_y)
        else:
            #reward = self.get_reward(target, collision, action, min_laser)
            # 230227 pureDRL
            reward = self.get_reward_path_230206_pureDRL(target, collision, action, min_laser, self.odom_x, self.odom_y, self.path_as_input, self.goal_x, self.goal_y, self.pre_distance, self.distance, self.pre_odom_x, self.pre_odom_y)
        
        
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
        
        if DYNAMIC_GLOBAL and episode_steps%20 ==0:   # 선택 1(fixed rewrind)
        #if DYNAMIC_GLOBAL and RECAL_WPT:             # 선택 2 아무 웨이포인트나 1.5안에 들어오면 replanning
        #if DYNAMIC_GLOBAL and self.pedsim_agents_list != None:   # 선택 3. CCTV안에 pedsim list 들어오면   # 230206
        #if DYNAMIC_GLOBAL and self.pedsim_agents_list != None and episode_steps%20 == 0:   # 선택 4. CCTV안에 pedsim list 들어오면 + 너무 자주 리플래닝 되지는 않게  # 230209
        
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
                        
                        self.pause()
                        rx, ry, flow_map = self.a_star.planning(sx, sy, gx, gy, self.odom_vx, angle, skew_x, skew_y, self.pedsim_agents_list)
                        self.flow_map = flow_map   # 230221
                        self.unpause()
                        
                        final_path = []
                        for path in zip (rx, ry):
                            final_path.append([path[0], path[1]])
                        #final_path.reverse()
                        #print(final_path)
                        final_path = np.array(final_path)
                        final_path = final_path / 10
                        path = final_path
                        
                        
                        
                        
                        
                        #path = astar.pure_astar.main(self.odom_x, self.odom_y, self.goal_x, self.goal_y, self.pedsim_agents_list) 
                    self.path_i_rviz = path
                    break
                except:
                    path = [[self.goal_x+5.5, self.goal_y+5.5]]   # 230209 5.5 더해줌
                    path = np.asarray(path)   # 221103
                    self.path_i_rviz = path
                    #print('예외발생[step]. path를 global goal로 지정: ', path)
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
                
            self.path_as_input = np.asarray(self.path_as_input)
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
                    order_by_desc = np.argpartition(top_k_list, -5)[-5:]   # https://stackoverflow.com/questions/65038206/how-to-get-indices-of-top-k-values-from-a-numpy-array
                    
                    for i, pros in enumerate(order_by_desc):
                        self.path_as_input[i,:] = path[pros,:]-5.5
                    '''       
 
                # 만약 크기 같다면: 
                elif len(path) == self.path_as_input_no:
                    self.path_as_input = path - 5.5
                    
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
            
            skew_x = optimal_g_x - self.odom_x
            skew_y = optimal_g_y - self.odom_y
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

        return state, reward, done, target

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
                
        
        if debug:
            position_ok = False
            
            while not position_ok:
                x = np.random.uniform(-5.5, 5.5)
                y = np.random.uniform(-5.5, 5.5)
                position_ok = check_pos_DWA_evaluate(x, y)
                
            ### DEBUG 용
            ###x = -4.5
            ###y = 4.5
                
        
                
                
        # set a random goal in empty space in environment
        self.change_goal()    # short-term 
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

                break
            except:
                path = [[self.goal_x, self.goal_y]]
                path = np.asarray(path)   # 221103
                print('[reset]예외발생. path를 global_goal로 지정', path)
                break
        
        self.path_i_prev = path
        self.path_i_rviz = path
        
        
        # TODO sampling 방법에 대해 고려
        #############################    
        ############# 221010 고정된 5 사이즈의 path output    self.path_as_input
        self.path_as_input = []
        for i in range(self.path_as_input_no):
            self.path_as_input.append([self.goal_x, self.goal_y])
        self.path_as_input = np.asarray(self.path_as_input)
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
            
            distance = np.linalg.norm([self.odom_x - optimal_g_x, self.odom_y - optimal_g_y])
            
            self.temp_path_as_input[i][0] = distance
            self.temp_path_as_input[i][1] = theta
                
        self.temp_path_as_input = self.temp_path_as_input.reshape(-1,)

        
        # 221014
        if self.sac_path:
        #if PATH_AS_INPUT:
            state = np.append(state, self.temp_path_as_input)
            #print('[reset]self.temp_path_as_input:',self.temp_path_as_input)
            
        return state

    def change_goal(self):   # adaptive goal positioning
        # Place a new goal and check if its location is not on one of the obstacles
        #if self.upper < 10:    # 5
        if self.upper < 5:  # 221108
            self.upper += 0.004
        #if self.lower > -10:   # -5
        if self.lower > -5: # 221108
            self.lower -= 0.004

        goal_ok = False

        while not goal_ok:
            self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)  
            self.goal_y = self.odom_y + random.uniform(self.upper, self.lower) # [-5, 5] -> [-10, 10]
            if SCENARIO=='warehouse':
                goal_ok = check_pos_warehouse(self.goal_x, self.goal_y)
            elif SCENARIO=='TD3': 
                goal_ok = check_pos(self.goal_x, self.goal_y)
            elif SCENARIO=='U':
                goal_ok = check_pos_U(self.goal_x, self.goal_y)
            elif SCENARIO=='DWA':
                goal_ok = check_pos_DWA(self.goal_x, self.goal_y)
                

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
    
            marker4 = Marker()
            marker4.id = i
            marker4.header.frame_id = "odom"
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
        # 5 optimal path
        markerArray6 = MarkerArray()
        for i, pose in enumerate(self.path_as_init):
    
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
            marker6.pose.position.x = pose[0]
            marker6.pose.position.y = pose[1]
            marker6.pose.position.z = 0

            markerArray6.markers.append(marker6)

        self.publisher6.publish(markerArray6)
        
        
        '''
        # flow_map from pure_astar visualize
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
                    marker7.id = marker_id #(i*10)+j
                    #print(marker7.id)
                    marker7.header.frame_id = "odom"
                    marker7.type = marker7.CYLINDER
                    marker7.action = marker7.ADD
                    marker7.scale.x = 0.1
                    marker7.scale.y = 0.1
                    marker7.scale.z = 0.01
                    marker7.color.a = 0.51
                    marker7.color.r = (self.flow_map[i][j] + 99)/198
                    marker7.color.g = 0.8
                    marker7.color.b = 0.7
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
                    #print([i,j],'의:',[marker7.pose.position.x,marker7.pose.position.y],marker7.pose.position.z)
                    marker7.pose.position.z = (self.flow_map[i][j] + 99)/198

                    markerArray7.markers.append(marker7)
                    marker_id += 1

        self.publisher7.publish(markerArray7)
        '''
    
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
        ## 1. global path의 waypoints들을 고려해서 이동
        #uu, xx = dwa_pythonrobotics.main(rx, ry, temp_gx, temp_gy, angle, self.dwa_x, ped_list)   
        ## 2. fixed global goal만 고려 (naive)
        #print('페드 리스트:',ped_list[:,:2])
        uu, xx = dwa_pythonrobotics.main(rx, ry, gx, gy, angle, self.dwa_x, ped_list)     
        
        diff_x = gx-rx
        diff_y = gy-ry
        to_go_rot = np.arctan2(diff_y, diff_x)
        return [uu[0], -uu[1]]    