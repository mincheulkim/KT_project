# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Hrushikesh Budhale, Pratik Acharya
# Created Date: Wednesday 20 April 2022
# =============================================================================


# =============================================================================
# Imports
# =============================================================================

from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import heapq as hq
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import math
import time
import cv2
import sys

# =============================================================================
# Motion Model class
# =============================================================================

class MotionModel:
    def __init__(self, wheel_radius, wheel_base, rpm1, rpm2, delta, res):
        self.RES = res  # resolution: number of parts in 1 meter
        self.TRAJ_LEN = int(1/delta)    # 10
        # trajectories is a 3x(TRAJ_LEN * 8) size matrix containing 8 
        # TRAJECTORIES from origin, with each column representing pose (x,y,theta)
        self.TRAJECTORIES = None
        # NEIGHBORS are the end points of all trajectories from origin
        self.NEIGHBORS = None   # get updated with constant matrix of shape 3x8
        # COSTS stores the distance travelled in each trajectory
        self.COSTS = None      # gets updated once in _compute_traj
        self.dt = delta
        self.delta = delta
        self.r = wheel_radius
        self.b = wheel_base
        # self.action_set = np.array([[0,rpm1],[rpm1,0],[rpm1,rpm1],[rpm1,rpm2],
        #                             [0,rpm2],[rpm2,0],[rpm2,rpm2],[rpm2,rpm1]])
        self.action_set = np.array([[0,rpm2], [rpm1,rpm2], [rpm2,rpm2], [rpm2,rpm1], [rpm2,0]])
        
        self.TRAJECTORIES, self.NEIGHBORS, self.COSTS = self._compute_traj()

    def get_neighbors(self, node):
        x, y, th = node
        # neighbors are the end points of all trajectories from given location
        neighbors = self._rotM(th) @ self.NEIGHBORS # rotate about origin
        neighbors[[0],:] += x                       # translate by x and y
        neighbors[[1],:] += y
        neighbors[[2],:] += th
        return neighbors.T, self.COSTS

    def get_trajectories(self, node, actions=None):
        x, y, th = node
        if actions == None: actions=range(len(self.action_set))   # consider all actions
        #     actions = np.arange(0, len(self.action_set))
        new_t = self._rotM(th) @ self.TRAJECTORIES
        new_t[[0],:] += x
        new_t[[1],:] += y
        new_t[[2],:] += th
        return [new_t[:,i*self.TRAJ_LEN:(i+1)*self.TRAJ_LEN].T for i in actions]

    def plot_trajectories(self, ax, trajectories):
        for traj in trajectories: ax.plot(traj[:,0], traj[:,1], c='b')

    def _rotM(self, theta):
        return np.array([[math.cos(theta), -math.sin(theta), 0],
                         [math.sin(theta),  math.cos(theta), 0],
                         [              0,                0, 1],])

    def _compute_traj(self):
        # This function gets used only once when the object is created
        trajectories = np.array([]).reshape(3,0)
        neighbors = np.array([]).reshape(3,0)
        costs = np.array([])
        for i, _ in enumerate(self.action_set):
            x = y = th = cost = 0
            trajectory = np.zeros((3,self.TRAJ_LEN))
            for j in range(self.TRAJ_LEN):
                dx = 0.5*self.r * (self.action_set[i][0] + self.action_set[i][1]) * math.cos(trajectory[2,j-1]) * self.dt
                dy = 0.5*self.r * (self.action_set[i][0] + self.action_set[i][1]) * math.sin(trajectory[2,j-1]) * self.dt
                dth = (self.r / self.b) * (self.action_set[i][1] - self.action_set[i][0]) * self.dt
                trajectory[0,j] = trajectory[0,j-1] + dx
                trajectory[1,j] = trajectory[1,j-1] + dy
                trajectory[2,j] = trajectory[2,j-1] + dth
                cost += math.sqrt(dx**2 + dy**2 + dth**2)
            trajectories = np.hstack((trajectories,trajectory))
            neighbors = np.hstack((neighbors,trajectory[:,[-1]]))
            costs = np.hstack((costs,cost))

        trajectories[:2,:] *= self.RES
        neighbors[:2,:] *= self.RES
        return trajectories, neighbors, costs*self.RES

# =============================================================================
# Planning functions
# =============================================================================

def visited(nxt, cost_so_far):
    # returns cost of visited or adjacent to visited 
    # returns False if not visited
    nxt = [round(nxt[0]), round(nxt[1])]
    min_cost = np.inf
    for i in range(-1,2):
        for j in range(-1,2):
            cost = cost_so_far.get((nxt[0]+i, nxt[1]+j), False)
            if cost != False: 
                if cost < min_cost: min_cost = cost

    if min_cost == np.inf: return False
    return min_cost

def check_if_invalid(x, y):
    return x < 0 or x >= conf.shape[2] or y < 0 or y >= conf.shape[1] or conf[0, y, x] != 0

def check_in_poly(pts, poly):
    count = np.zeros(pts.shape[0])
    for i, _ in enumerate(poly[:-1]):
        # pts.y should be within y limits of line and pts.x should be less than intersection.x
        intersection_x = (poly[i+1,0] - poly[i,0]) * (pts[:,1]-poly[i,1]) / (poly[i+1,1] - poly[i,1]) + poly[i,0]
        count += (((pts[:,1] > poly[i,1]) != (pts[:,1] > poly[i+1,1])) & (pts[:,0] < intersection_x))*1
    return count % 2 # point is outside if even number of intersections

def back_track(came_from, start, parent):
    path = []
    action_history = []
    while parent != start: 
        parent, action = came_from[parent[:2]]
        path.append(mm.get_trajectories(parent, actions=[action]))
        action_history.append(action)
    path.reverse()
    action_history.reverse()
    return np.array(path).reshape(-1,3), action_history

def create_rect(cx, cy, ht, wd):
    return np.array([(cx-(wd/2), cy-(ht/2)), (cx-(wd/2), cy+(ht/2)),
                     (cx+(wd/2), cy+(ht/2)), (cx+(wd/2), cy-(ht/2)),
                     (cx-(wd/2), cy-(ht/2))])
    
def boundary_dist(velocity, rel_ang, laser_flag, const=0.354163):
        # Parameters from Rachel Kirby's thesis
        front_coeff = 1.0
        #front_coeff = 2.0
        side_coeff = 2.0 / 3.0
        rear_coeff = 0.5
        safety_dist = 0.5
        velocity_x = velocity[0]
        velocity_y = velocity[1]

        velocity_magnitude = np.sqrt(velocity_x ** 2 + velocity_y ** 2)
        variance_front = max(0.5, front_coeff * velocity_magnitude)
        variance_side = side_coeff * variance_front
        variance_rear = rear_coeff * variance_front

        rel_ang = rel_ang % (2 * np.pi)
        flag = int(np.floor(rel_ang / (np.pi / 2)))
        if flag == 0:
            prev_variance = variance_front
            next_variance = variance_side
        elif flag == 1:
            prev_variance = variance_rear
            next_variance = variance_side
        elif flag == 2:
            prev_variance = variance_rear
            next_variance = variance_side
        else:
            prev_variance = variance_front
            next_variance = variance_side

        dist = np.sqrt(const / ((np.cos(rel_ang) ** 2 / (2 * prev_variance)) + (np.sin(rel_ang) ** 2 / (2 * next_variance))))
        dist = max(safety_dist, dist)

        # Offset pedestrian radius
        if laser_flag:
            dist = dist - 0.5 + 1e-9

        return dist

def draw_social_shapes(position, velocity, laser_flag, const=0.35):
        # This function draws social group shapes
        # given the positions and velocities of the pedestrians.

        #total_increments = 20 # controls the resolution of the blobs
        total_increments = 80 # controls the resolution of the blobs  #0228 리포트때 480으로 함
        quater_increments = total_increments / 4
        angle_increment = 2 * np.pi / total_increments

        # Draw a personal space for each pedestrian within the group
        contour_points = []
        center_x = position[0]
        center_y = position[1]
        velocity_x = velocity[0]
        velocity_y = velocity[1]
        velocity_angle = np.arctan2(velocity_y, velocity_x)

        # Draw four quater-ovals with the axis determined by front, side and rear "variances"
        # The overall shape contour does not have discontinuities.
        for j in range(total_increments):

            rel_ang = angle_increment * j
            value = boundary_dist(velocity, rel_ang, laser_flag, const)
            #value *= 1.2  # 0228 리포트때는 1.2배 함
            addition_angle = velocity_angle + rel_ang
            x = center_x + np.cos(addition_angle) * value
            y = center_y + np.sin(addition_angle) * value
            contour_points.append((x, y))
                #print('컨투어 포인트:',j,x,y)

        # Get the convex hull of all the personal spaces
        

        return contour_points

def get_obstacle_map(pts, pedsim_agents_list):
    '''
    # add square
    obs = check_in_poly(pts, RES*create_rect(1, 5, 1.5, 1.5).reshape(-1,2))

    # add rectangle 1
    obs2 = check_in_poly(pts, RES*create_rect(5, 5, 1.5, 2.5).reshape(-1,2))
    obs = np.logical_or(obs, obs2)
    
    # add rectangle 2
    obs2 = check_in_poly(pts, RES*create_rect(8, 3, 2, 1.5).reshape(-1,2))
    obs = np.logical_or(obs, obs2)

    # add circle 1
    center, radius = np.array([2,8])*RES, 1*RES
    obs2 = np.linalg.norm(pts-center, axis=1) < radius
    obs = np.logical_or(obs, obs2)
    
    # add circle 2
    center, radius = np.array([2,2])*RES, 1*RES
    obs2 = np.linalg.norm(pts-center, axis=1) < radius
    obs = np.logical_or(obs, obs2)

    '''
    # 좌상단
    map_bias = 5.5   # 4.5
    obs = check_in_poly(pts, RES*create_rect(-2.5+map_bias, (1.5/2)+map_bias, 3.5, 2).reshape(-1,2))   # h, w

    # 센터
    obs2 = check_in_poly(pts, RES*create_rect(1+map_bias, (-2.5/2)+map_bias, 2.5, 2).reshape(-1,2))
    obs = np.logical_or(obs, obs2)
    
    # 우하단
    obs3 = check_in_poly(pts, RES*create_rect((9.5/2)+map_bias, (-5.5/2)+map_bias, 5.5, 1.5).reshape(-1,2))
    obs = np.logical_or(obs, obs3)
    
    # 우상단 라인
    # wall_15
    obs4 = check_in_poly(pts, RES*create_rect((7.5/2)+map_bias, 2.5+map_bias, 0.2, 3.5).reshape(-1,2))  # height, width
    obs = np.logical_or(obs, obs4)
    
    #print('디버그:',pedsim_agents_list)
    # 220928 사람 추가
    if pedsim_agents_list != None:
        for i, ped in enumerate(pedsim_agents_list):
            x = ped[0] + map_bias
            y = ped[1] + map_bias
            
            '''
            # Method 1. Fixed human area
            center, radius = np.array([int(x*RES),int(y*RES)]),  1*RES   # RES = 10
            obs_ped = np.linalg.norm(pts-center, axis=1) < radius   # len(obs_ped) = 10210
            '''
            
            # Method 2. Social individual space
            ind_space = draw_social_shapes(ped[:2], ped[2:4], False)   # position, velocity
            #print(ind_space)
            obs_ped = check_in_poly(pts, RES*(np.array(ind_space)+map_bias).reshape(-1,2))
            
            obs = np.logical_or(obs, obs_ped)
    
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

    # create 3 dimensional configuration space
    conf = np.zeros([12, height, width], dtype=np.uint8)
    for depth in range(12): # for 12 different orientatiosn 
        # add obstacles with clearance
        conf[depth,:,:] = obs_int

        # add robot radius based on angle (Commenting out since robot is circular)
        # M = cv2.getRotationMatrix2D(center, 30*depth, 1)
        # rotated_robot_shape = cv2.warpAffine(robot_shape, M, center)
        conf[depth,:,:] = cv2.dilate(conf[depth,:,:], robot_shape)

    return conf, obs.reshape(height, width)

def get_neighbors(current, cost_so_far):
    neighbors, costs = mm.get_neighbors(current)
    valid_i = []
    for i, n in enumerate(neighbors):
        if check_if_invalid(round(n[0]), round(n[1])): continue
        if not visited(n, cost_so_far): 
            valid_i.append(i)   # add to valid neighbors
    return neighbors[valid_i], costs[valid_i], valid_i

def find_path(start, goal, animate=False):
    frontier, tmp = [], []
    came_from, cost_so_far = dict(), dict()
    cost_so_far[tuple(start[:2])] = 0
    came_from[tuple(start[:2])] = (0, -1)  # stores tuple of parent node and action index
    hq.heappush(frontier, (0, start))

    while len(frontier) > 0:
        cost, current = hq.heappop(frontier)
        considered_actions = []
        csf = cost_so_far[(round(current[0]), round(current[1]))]
        for nxt, cost, i in zip(*get_neighbors(current, cost_so_far)):
            new_cost = csf + cost 
            visited_cost = visited(nxt, cost_so_far)
            if (not visited_cost) or (new_cost < visited_cost): 
                cost_so_far[(round(nxt[0]), round(nxt[1]))] = new_cost
                priority = new_cost + math.dist(nxt[:2], goal[:2])
                hq.heappush(frontier, (priority, tuple(nxt)))
                came_from[tuple(nxt[:2])] = (current, i)
                considered_actions.append(i)

                if math.dist(nxt[:2], goal[:2]) < GOAL_RADIUS:   # nxt가 골 범위 안에 도착시:
                    return back_track(came_from, start, current)   # return path, action

        if animate: # visualization of map exploration
            mm.plot_trajectories(ax, mm.get_trajectories(current, actions=considered_actions))
            if len(came_from)%50 == 0:
                plt.pause(0.001)

def run_application(start, goal):
    # find path

    if PLOT:
        ax.set_title("Searching Path")
        ticks = np.arange(0, 10*RES+1, RES)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        plt.pause(0.001)    # Necessary for updating title
    
    start_time = time.perf_counter()
    path, actions = find_path(start, goal)
    exec_time = time.perf_counter() - start_time
    #print('whw')
    #print(path)

    
    if PLOT:
        message = f"Path Found! in {round(exec_time, 2)} sec."
        ax.set_title(message)
        ax.plot(path[:,0],path[:,1], c='r')
        # show animation
        find_path(start, goal, animate=False)
        ax.plot(path[:,0],path[:,1], c='r')
        ax.scatter(path[::10,0], path[::10,1], s=5, c='b')
        plt.show()
    #print('path:',path / RES - 5.5)
    #return mm.action_set[actions]   # roginal
    path_indexing = np.array(path)
    #print('힝',path_indexing[:,:2])
    return path_indexing[:,:2] / RES
    #return path[:,:] / RES # rescale it to original resolution

def parse_args(argv):
    if len(argv) == 1:  # running with no arguments
        return (1*RES, 1*RES, 0), (9*RES, 9*RES, 0), 5, 10, 0.1, True
    elif len(argv) > 9: # Input from launch file
        argv = [float(i) for i in argv[:8]]
        start = (argv[0]*RES, argv[1]*RES, argv[2])
        goal = (argv[3]*RES, argv[4]*RES, 0.0)
        return start, goal, argv[5], argv[6], argv[7], False
    elif len(argv) == 9:  # running stand alone
        argv = [float(i) for i in argv[1:9]]
        return (argv[0]*RES, argv[1]*RES, argv[2]), \
               (argv[3]*RES, argv[4]*RES, 0.0), argv[5], argv[6], argv[7], True
    print("Wrong input arguments")
    sys.exit()

# =============================================================================    
# Main logic
# =============================================================================    

#def main(argv=[]):
def main(r_x, r_y, g_x, g_y, pedsim_agents_list):   # [-4.5 ~ 4.5],
    
    global width, height, obs, conf, ax, mm
    global RES, STEP_SIZE, ROBOT_RADIUS, CLEARANCE, GOAL_RADIUS, PLOT
    RES = 10  # parts per meter
    
    #start, goal, rpm1, rpm2, clearance, PLOT = parse_args(argv)
    #(1*RES, 1*RES, 0), (9*RES, 9*RES, 0), 5, 10, 0.1, True
    
    # DEBUG
    '''
    r_x = -4.0
    r_y = -4.0
    g_x = 2.0
    g_y = 4.0
    pedsim_agents_list = []
    '''
    PLOT = False    # visualize 할건지
    
    
    r_x += 5.5
    r_y += 5.5
    g_x += 5.5
    g_y += 5.5
    start= (int(r_x*RES), int(r_y*RES), 0)   # (1*RES, 1*RES)
    goal = (int(g_x*RES), int(g_y*RES), 0)   # (9*RES, 9*RES)
    rpm1 = 5
    rpm2 = 10
    clearance = 0.1
    
    
    
    GOAL_RADIUS = round(0.25 *RES)
    ROBOT_RADIUS = round(0.2 *RES)
    CLEARANCE = round(clearance *RES)
    STEP_SIZE = round(1 *RES)

    mm = MotionModel(wheel_radius=0.033, wheel_base=0.16, rpm1=rpm1, rpm2=rpm2, delta=0.1, res=RES)
    
    # create exploration space
    width, height = 10*RES+1, 10*RES+1
    Y, X = np.mgrid[0:height, 0:width]
    pts = np.array([X.ravel(), Y.ravel()]).T

    # create obstacle map
    conf, obs = get_obstacle_map(pts, pedsim_agents_list)

    if check_if_invalid(round(start[0]), round(start[1])):
        print("Start position is not in configuration space. Exiting.")
        sys.exit()
    if check_if_invalid(round(goal[0]), round(goal[1])):
        print("Goal position is not in configuration space. Exiting.")
        sys.exit()
    
    if PLOT:
        fig, ax = plt.subplots()
        ax.axis('equal')
        plt.subplots_adjust(bottom=0.15)
        ax.scatter(pts[obs.flatten(),0], pts[obs.flatten(),1], s=1, c='k')
    
    return run_application(start, goal)

if __name__ == "__main__":
    main(sys.argv)
    