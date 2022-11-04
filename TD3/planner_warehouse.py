# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Pratik Acharya, Hrushikesh Budhale
# Created Date: Saturday 7 May 2022
# =============================================================================
### 221102 warehouse 환경에서 플래너

# =============================================================================
# Imports
# =============================================================================

import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import math
import time
import cv2
import sys

# =============================================================================
# RRT + APF class
# =============================================================================

class RrtApf:
    #def __init__(self, start, goal, obs_map, offset=3, maxIter=5000, goal_radius=15, animate=False):
    #def __init__(self, start, goal, obs_map, offset=6, maxIter=5000, goal_radius=15, animate=False):   # obs_map = conf
    def __init__(self, start, goal, obs_map, offset=6, maxIter=200000, goal_radius=15, animate=False):   # 221103
        self.start = start
        self.goal = goal
        self.animate = animate                      # boolean variable to show expanding rrt search tree
        self.goal_probability = 0.1                 # probability to select goal node while selecting random nodes
        self.maxIter = maxIter                      # maximum number of iterations before returning 'path not found'
        self.nodes = [start]                        # stores visited nodes while exploring the map
        self.goal_radius = goal_radius              # vicinity aroung goal to consider path to be complete
        self.offset = offset                        # distance between selected random nodes and it's 2 neighbots
        self.came_from = {tuple(self.start): None}  # dictionary for back tracking found path
        self.distance_mat = ndimage.distance_transform_edt(obs_map/255 == 0)    # stores distance of each node from obstacle
        self.distance_mat = self.distance_mat / 40 + 0.00000001   # 201, 201
        self.pedsim_list = None

    def get_nearest_node(self, node):    # node = picked random node
        min_dist = np.inf                # 최소 거리 일단 무한대
        nearest_node = None
        for node_i in self.nodes:        # 최초 단에는 start위치만 들어가 있음
            dist = np.linalg.norm(node - node_i)   # self.nodes에서 가장 거리 가까운 놈을 셀렉
            if dist < min_dist:
                min_dist = dist
                nearest_node = node_i
        return nearest_node

    def calculate_apf(self, valid_nodes, attraction=0.5, repulsion=0.5):
        least_potential = np.inf
        new_node = None
        #print('valid nodes:',valid_nodes)
        for node in valid_nodes:
            distance_to_goal = np.linalg.norm(node - self.goal)
            positive_potential = attraction * distance_to_goal
            negative_potential = repulsion / self.distance_mat[node[1]][node[0]]
            
            # 1. Realibility (as visibility)
            realibility = 1.0
            # 2. Density     node 주변에 사람들의 거리를 sum
            density_sum = 0.0
            #print(self.pedsim_list)
            if self.pedsim_list != None:
                for i, ped in enumerate(self.pedsim_list):
                    px = RES*(ped[0] + 10)
                    py = RES*(ped[1] + 10)
                    distance = np.sqrt((node[0]-px)**2+(node[1])**2)
                    distance = distance / 10
                    #print(i, node, distance, ped, [px,py])
                    density_sum += 1 / distance
                    
            #total_potential = positive_potential + negative_potential
            total_potential = positive_potential + negative_potential + density_sum ## 221103
            if total_potential < least_potential:
                least_potential = total_potential
                new_node = node
            
            #print('node:',node,'p_p:',positive_potential,'n_p:',negative_potential, 'p_p:',density_sum)
        #print('new nodes:',new_node)
        return new_node

    def get_random_node(self):
        if np.random.random() < self.goal_probability:
            return np.array(self.goal)
        x = np.random.randint(0, self.distance_mat.shape[1])
        y = np.random.randint(0, self.distance_mat.shape[0])
        return np.array([x, y])

    def get_new_node(self, node, parent_node):           # random node, nearest node

        if np.array_equal(node, parent_node): return node

        direction = node - parent_node             # random node -> nearest node
        direction_norm = (direction / np.linalg.norm(direction)) * self.offset
        new_node_center = (parent_node + direction_norm).round().astype(int)

        dx = dy = 0
        if (node[0] - parent_node[0]) == 0: dx = -self.offset
        elif (node[1] - parent_node[1]) == 0: dy = -self.offset
        else:
            slope = (node[1] - parent_node[1]) / (node[0] - parent_node[0])
            dy = math.sqrt(self.offset ** 2 / (slope ** 2 + 1))
            dx = -slope * dy
        #print('오리진:',new_node_center)
        new_node_left = new_node_center + np.array([dx, dy])
        new_node_right = new_node_center - np.array([dx, dy])
        selected_nodes = np.array([new_node_center, new_node_left, new_node_right]).round().astype(int)

        valid_nodes = selected_nodes[np.asarray([[not check_if_invalid(node_i[0], node_i[1])] for node_i in
                                                  selected_nodes]).flatten()]

        return self.calculate_apf(valid_nodes) if len(valid_nodes) != 0 else None

    def back_track(self, new_node):
        path = [self.goal]
        while new_node is not None:
            path.append(new_node)
            new_node = self.came_from[tuple(new_node)]
        path.reverse()
        return np.array(path).reshape(-1,2)

    def find_path(self, goal):
        for iteration in range(self.maxIter):   # 최대 15000번 반복
            random_node = self.get_random_node()    # 랜덤 노드 [201, 201] 중에서 셀렉
            nearest_node = self.get_nearest_node(random_node)    # self.nodes list에서 random 노드와 가장 가까운 놈을 셀렉
            # TODO 여기서 nearest node가 사람이면 distance 길게 페널티
            # TODO 여기서 nearest node가 invisible area 면 distance 길게 페널티

            new_node = self.get_new_node(random_node, nearest_node)
            if (new_node is None) or np.array_equal(new_node, nearest_node):   # apf 통과한 노드 없을 경우: 이번 iter skip (다시 랜덤 셀렉)
                continue
            
            # when path reaches goal location
            if np.linalg.norm(new_node - self.goal) < self.goal_radius:
                self.came_from[tuple(new_node)] = nearest_node
                #print(f"Path found in {iteration} iterations")
                return self.back_track(new_node), self.nodes
                
            match_found = np.any(np.all(new_node == self.nodes, axis=1))
            if not match_found:                                   # 새로운 노드면
                self.came_from[tuple(new_node)] = nearest_node
                self.nodes.append(new_node)                       # self.node list에 new node 추가
                if self.animate:
                    ax.plot(new_node[0], new_node[1], 'b.', markersize=2)
                    plt.pause(0.001)

        print("Path not found, replaced by the goal")
        goall = [RES*(goal[0]+10), RES*(goal[1]+10)]
        goall = np.asarray(goall)
        return goall
        #sys.exit()

# =============================================================================
# Helper functions
# =============================================================================

def check_if_invalid(x, y):
    #print('conf.shape[1]:',conf.shape[1], 'conf.shape[0]:',conf.shape[0], 'conf[y,x]:',conf[y, x])
    #print( x < 0)
    #print(x >= conf.shape[1])
    #print(y >= conf.shape[0] )
    #print(conf[y, x] != 0)
    return x < 0 or x >= conf.shape[1] or y < 0 or y >= conf.shape[0] or conf[y, x] != 0

def check_in_poly(pts, poly):
    count = np.zeros(pts.shape[0])
    for i, _ in enumerate(poly[:-1]):
        # pts.y should be within y limits of line and pts.x should be less than intersection.x
        intersection_x = (poly[i+1,0] - poly[i,0]) * (pts[:,1]-poly[i,1]) / (poly[i+1,1] - poly[i,1]) + poly[i,0]
        count += (((pts[:,1] > poly[i,1]) != (pts[:,1] > poly[i+1,1])) & (pts[:,0] < intersection_x))*1
    return count % 2 # point is outside if even number of intersections

def create_rect(cx, cy, ht, wd):
    return np.array([(cx-(wd/2), cy-(ht/2)), (cx-(wd/2), cy+(ht/2)),
                     (cx+(wd/2), cy+(ht/2)), (cx+(wd/2), cy-(ht/2)),
                     (cx-(wd/2), cy-(ht/2))])

def get_obstacle_map(pts, pedsim_agent_list, r_x, r_y):
    # wall2
    map_bias = 10
    obs = check_in_poly(pts, RES*create_rect(-5+map_bias, -8+map_bias, 4.0, 8.0).reshape(-1,2))
    # wall3
    obs2 = check_in_poly(pts, RES*create_rect(-10+map_bias, 0+map_bias, 20.0, 2.0).reshape(-1,2))
    obs = np.logical_or(obs, obs2)
    # NW three pallets
    obs3 = check_in_poly(pts, RES*create_rect(-3.5+map_bias, 5.5+0.9+map_bias, 1.8, 7.0).reshape(-1,2))
    obs = np.logical_or(obs, obs3)
    # NW two pallets
    obs4 = check_in_poly(pts, RES*create_rect((-4.2-8.8)/2+map_bias, 1+1+map_bias, 2.0, 4.6).reshape(-1,2))
    obs = np.logical_or(obs, obs4)
    # control panel
    obs5 = check_in_poly(pts, RES*create_rect(2.0701+map_bias, 1.1622+map_bias, 0.6001, 0.8673).reshape(-1,2))
    obs = np.logical_or(obs, obs5)
    # rack1
    obs6 = check_in_poly(pts, RES*create_rect(7.35056+map_bias, 1.7443+map_bias, 1.9558, 0.8444).reshape(-1,2))
    obs = np.logical_or(obs, obs6)
    # rack2
    obs7 = check_in_poly(pts, RES*create_rect(5.36795+map_bias, 1.7443+map_bias, 1.9558, 0.8444).reshape(-1,2))
    obs = np.logical_or(obs, obs7)
    # rack3
    obs8 = check_in_poly(pts, RES*create_rect(7.001420+map_bias, -7.646700+map_bias, 3.91162, 0.8444).reshape(-1,2))
    obs = np.logical_or(obs, obs8)
    # rack4
    obs9 = check_in_poly(pts, RES*create_rect(4.97338+map_bias, -7.646700+map_bias, 3.91162, 0.8444).reshape(-1,2))
    obs = np.logical_or(obs, obs9)
    # rack5
    obs10 = check_in_poly(pts, RES*create_rect(2.669490+map_bias, -7.646700+map_bias, 3.91162, 0.8444).reshape(-1,2))
    obs = np.logical_or(obs, obs10)
    # pole1
    obs11 = check_in_poly(pts, RES*create_rect(5.02174+map_bias, -2.39478+map_bias, 0.49161, 0.49161).reshape(-1,2))
    obs = np.logical_or(obs, obs11)
    # pole2
    obs12 = check_in_poly(pts, RES*create_rect(0.470710+map_bias, -2.39478+map_bias, 0.49161, 0.49161).reshape(-1,2))
    obs = np.logical_or(obs, obs12)
    # pole3
    obs13 = check_in_poly(pts, RES*create_rect(-3.93786+map_bias, -2.39478+map_bias, 0.49161, 0.49161).reshape(-1,2))
    obs = np.logical_or(obs, obs13)

    
    
       # 4.5
    # 좌상단 십자
    # wall_11
    
    # set of recycle bin
    obs15 = check_in_poly(pts, RES*create_rect(5.5+map_bias, 5.8+map_bias, 0.8, 6).reshape(-1,2))
    obs = np.logical_or(obs, obs15)

    # add rectangle 1
    #obs2 = check_in_poly(pts, RES*create_rect(5, 5, 1.5, 2.5).reshape(-1,2))
    #obs = np.logical_or(obs, obs2)
    
    # add rectangle 2
    #obs2 = check_in_poly(pts, RES*create_rect(8, 3, 2, 1.5).reshape(-1,2))
    #obs = np.logical_or(obs, obs2)
    

    # add circle 1
    #center, radius = np.array([2,8])*RES, 1*RES
    #obs2 = np.linalg.norm(pts-center, axis=1) < radius
    #obs = np.logical_or(obs, obs2)
    
    # add circle 2
    #center, radius = np.array([2,2])*RES, 1*RES
    #obs2 = np.linalg.norm(pts-center, axis=1) < radius
    #obs = np.logical_or(obs, obs2)
    
    
    
    #print('pedsim_list:',pedsim_agent_list)  # DEBUG
    # 220928 사람 추가
    '''
    if pedsim_agent_list != None:
        for i, ped in enumerate(pedsim_agent_list):
            x = ped[0] + map_bias
            y = ped[1] + map_bias
            #center, radius = np.array([int(x),int(y)])*RES, 1*RES
            human_radius = 1   # 221103
            center, radius = np.array([int(x*RES),int(y*RES)]), human_radius * RES * 0.35 # 1*RES
            obs2 = np.linalg.norm(pts-center, axis=1) < radius
            obs = np.logical_or(obs, obs2)
    '''

    # create border
    obs2 = np.zeros((height,width), dtype=bool)
    bw = int(0.1 *RES) # border width
    obs2[:,0:bw] = obs2[0:bw,:] = obs2[:,-bw:] = obs2[-bw:,:] = True
    obs = np.logical_or(obs, obs2.flatten())

    obs_int = obs.reshape(height, width).astype('uint8')
    
    # create clearance kernel
    clearance_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(CLEARANCE*2, CLEARANCE*2))
    # create obstacles with clearance
    obs_int = cv2.dilate(obs_int, clearance_kernel)*250 # 250 is color intensity
    
    # create robot footprint
    robot_shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ROBOT_RADIUS*2, ROBOT_RADIUS*2))
    center = (robot_shape.shape[0]//2, robot_shape.shape[1]//2)

    conf = obs_int.copy()
    conf = cv2.dilate(conf, robot_shape)
    return conf, obs.reshape(height, width)

def run_application(start, goal, pedsim_list):   # RrtApf 돌린 후 path list return
    # find path
    planner = RrtApf(start, goal, conf, animate=False)   # conf: obstacles?
    planner.pedsim_list = pedsim_list

    if PLOT:
        ax.set_title("Searching Path")
        #ticks = np.arange(0, 10*RES+1, RES)
        ticks = np.arange(0, 20*RES+1, RES)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        plt.pause(0.001)    # Necessary for updating title   # 안그려지면 여기 숫자 늘려보기?
    
    start_time = time.perf_counter()
    path, nodes = planner.find_path(goal)
    exec_time = time.perf_counter() - start_time
    
    if PLOT:
        message = f"Path Found! in {round(exec_time, 2)} sec."
        ax.set_title(message)
        ax.plot(path[:,0],path[:,1], c='r')
        ax.plot(np.array(nodes)[:, 0], np.array(nodes)[:, 1], 'b.', markersize=2)
        print("Press 'q' to start following path")
        plt.show()

    return path / RES # rescale it to original resolution

def parse_args(argv):
    if len(argv) == 1:  # running with no arguments
        return (1*RES, 1*RES), (9*RES, 9*RES), 5, 10, 0.1, True
    elif len(argv) > 9: # Input from launch file
        argv = [float(i) for i in argv[:8]]
        start = (argv[0]*RES, argv[1]*RES)
        goal = (argv[3]*RES, argv[4]*RES)
        return start, goal, argv[5], argv[6], argv[7], True
    elif len(argv) == 9:  # running stand alone
        argv = [float(i) for i in argv[1:9]]
        return (argv[0]*RES, argv[1]*RES), \
               (argv[3]*RES, argv[4]*RES), argv[5], argv[6], argv[7], True
    print("Wrong input arguments")
    sys.exit()

# =============================================================================    
# Main logic
# =============================================================================    

#def main(argv=[]):
def main(r_x, r_y, g_x, g_y, pedsim_agents_list):   # [-4.5 ~ 4.5],
    global width, height, conf, ax
    global RES, STEP_SIZE, ROBOT_RADIUS, CLEARANCE, GOAL_RADIUS, PLOT
    RES = 10  # parts per meter
    #print('오리지날 로봇위치:',r_x,r_y)
    #print('오리지날 골위치:',g_x,g_y)
    #start, goal, rpm1, rpm2, clearance, PLOT = parse_args(argv)
    # gazebo상 좌표를 [0~, 0~]로 전환
    r_x += 10.0
    r_y += 10.0
    g_x += 10.0
    g_y += 10.0
    
    start= (int(r_x*RES), int(r_y*RES))   # (1*RES, 1*RES)
    #print('start:',start)
    goal = (int(g_x*RES), int(g_y*RES))   # (9*RES, 9*RES)
    #print('goal:',goal)
    rpm1 = 5
    rpm2 = 10
    clearance = 0.1
    
    PLOT = False    # visualize 할건지
    GOAL_RADIUS = round(0.25 *RES)
    ROBOT_RADIUS = round(0.2 *RES)
    CLEARANCE = round(clearance *RES)
    STEP_SIZE = round(1 *RES)

    # create exploration space
    #width, height = 10*RES+1, 10*RES+1    # (21, 21)
    width, height = 20*RES+1, 20*RES+1    # (-5.5 ~ 5.5)
    Y, X = np.mgrid[0:height, 0:width]    # np.mgrid[0:201, 0:201] -> 
    
    pts = np.array([X.ravel(), Y.ravel()]).T   # ravel: 다차원을 1차원으로 푸는것, reshape(-1) or flatten()
    #print('pts:',pts)

    # create obstacle map
    conf, obs = get_obstacle_map(pts, pedsim_agents_list, r_x, r_y)

    '''
    if check_if_invalid(round(start[0]), round(start[1])):  # start, goal이 영역 밖을 나갔는지, 해당 위치가 장애물이 있는 곳인지 확인
        print("Start position is not in configuration space. Exiting.")
        sys.exit()
    if check_if_invalid(round(goal[0]), round(goal[1])):
        print("Goal position is not in configuration space. Exiting.")
        sys.exit()
    '''
    
    if PLOT:
        fig, ax = plt.subplots()
        ax.axis('equal')
        # plt.subplots_adjust(bottom=0.15)
        #ax.scatter(pts[obs.flatten(),0], pts[obs.flatten(),1], s=1, c='k')
        ax.scatter(pts[obs.flatten(),0], pts[obs.flatten(),1], s=1, c='#ff7f0e', marker = '*')   # c=색깔, marker=형태   # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html
    #print('스타트:',start, 'goal:',goal)
    
    return run_application(start, goal, pedsim_agents_list)

if __name__ == "__main__":
    main(sys.argv)
