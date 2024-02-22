

"""
A* grid planning
author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)
See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)
"""

import math

import matplotlib.pyplot as plt

import numpy as np
from numpy import dot
from numpy.linalg import norm

from skimage.draw import polygon2mask, polygon, polygon_perimeter


show_animation = False


class AStarPlanner:

    def __init__(self, ox, oy, resolution, rr):
        """
        Initialize grid map for a star planning
        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)
        # 230213
        self.flow_map = None
        self.calc_flow_map(0.0, 0.0, 0.0, 0.0, [])

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy, r_linear, r_angular, skew_x, skew_y, pedsim_list = []):
        """
        A star path search
        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]
        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """
        #pedsim_list = []
        self.flow_map = None
        self.calc_flow_map(r_linear, r_angular, skew_x, skew_y, pedsim_list)
        #print(r_linear, r_angular)
        
        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)  # 15, 15, 0, -1  (x, y, cost, parent)
        
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)   # 95, 95, 0, -1

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node   # open_set[1665]

        while True:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                #key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o]))   
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o])+ self.motion_flow(goal_node, open_set[o], self.flow_map))   
            # keyfmf lambda o: open_set.cost + heuristic, then return min cost element in open_set   # https://jaeworld.github.io/python/python_lambda_usage/
            
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:   # found path
                #print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry, self.flow_map

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):   # goal node, node o from open_set
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d
    
    @staticmethod
    def motion_flow(n1, n2, flow_map):   # n1=goal node, n2 = openset node
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        #print('flow g.x,g.y:',n1.x, n1.y)
        #if flow_map[n2.x][n2.y] != 0.0:
        #    print('flow ob x,y:',n2.x,n2.y, d, '플로우코스트:',flow_map[n2.x][n2.y])
        #return d
        flow_weight = flow_map[n2.x][n2.y]
        
        return flow_weight

    def calc_grid_position(self, index, min_position):
        """
        calc grid position
        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        #print("min_x:", self.min_x)
        #print("min_y:", self.min_y)
        #print("max_x:", self.max_x)
        #print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        #print("x_width:", self.x_width)
        #print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break
                    
    def calc_flow_map(self, r_linear, r_angular, skew_x, skew_y, pedsim_list=None):
    
        #print("x_width:", self.x_width)
        #print("y_width:", self.y_width)
        self.min_x = 0
        self.min_y = 0
        self.max_x = 110
        self.max_y = 110
        RES = 10
        map_bias = 5.5
        
        Y, X = np.mgrid[0:self.max_x, 0:self.max_y]
                
        self.x = 0
        self.y = 0
        
        ind_space = None
        
        vertices = np.array([[65,65]])
        polygon_array = create_polygon([110,110], vertices)
        
        self.flow_map = np.array(self.flow_map)
                
        self.flow_map = polygon_array   # (110,110)
        
        robot_vx = r_linear * np.cos(r_angular)    # 로봇 글로벌 vx
        robot_vy = r_linear * np.sin(r_angular)    # 로봇 글로벌 vy(linear * heading)
        
        #print('로봇vx,vy:', robot_vx, robot_vy)
        #print('스큐 x,y:', skew_x, skew_y)
        skew_size = np.sqrt(skew_x**2 + skew_y**2)   # 230214
        goal_to_robot_vel = [skew_x/skew_size, skew_y/skew_size]
        #print('스큐 파이널:',goal_to_robot_vel)
        #print(pedsim_list)
        
        #if pedsim_list == []:
        #print('야')
        
        if pedsim_list != None:
            original_flow_map = self.flow_map
            for i, ped in enumerate(pedsim_list):
                #print(i, ped)
                x = int((ped[0] + map_bias)/0.1)
                y = int((ped[1] + map_bias)/0.1)
                vx = int((ped[2] + map_bias)/0.1)
                vy = int((ped[3] + map_bias)/0.1)
                ind_space = draw_social_shapes([x,y], [vx,vy], False)   # position, velocity
                #print('ind_space:',ind_space)
                
                shape = (110, 110)
                                
                imgp2 = polygon2mask(shape, ind_space).astype(int)  # astype() converts bools to strings
                
                ori = self.flow_map
                
                # TODO
                # 사람 빼기 로봇 -> 이 벡터를 로봇 - 골 vector와 similarity 비교. cost -1~1 -> 0, abs(2)
                # 로봇-to-goal nominal vector = goal_to_robot_vel
                # 사람 속도 = human_v
                # 로봇 속도 = [robot_vx, robot_vy]
                # 1. relative 사람 속도 by 로봇
                
                #rel_ped_robot_v = [ped[2] - robot_vx,    ped[3]-robot_vy]
                rel_ped_robot_v = [ped[2],    ped[3]]   # 231107
                #skew_size_r = np.sqrt(rel_ped_robot_v[0]**2 + rel_ped_robot_v[1]**2)
                #rel_ped_robot_v = [rel_ped_robot_v[0]/skew_size_r,rel_ped_robot_v[1]/skew_size_r]
                # 2. goal nomial과 cosine similary 계산
                def cos_sim(A, B):
                    return dot(A, B)/(norm(A)*norm(B))
                cos_sim_ped = cos_sim(rel_ped_robot_v, goal_to_robot_vel)
                
                #print('사람위치',ped[0],ped[1],cos_sim_ped, '로봇',robot_vx, robot_vy, '사람속도:',ped[2],ped[3])
                #print('로봇 인텐션:',goal_to_robot_vel,'rel.사람:',rel_ped_robot_v)

                #print(i, cos_sim_ped)
                
                
                '''
                cos_sim = dot(rel_ped_robot_v, goal_to_robot_vel) / (norm(rel_ped_robot_v)*norm(goal_to_robot_vel))
                # uncertation가 기본 flow cost로 들어가 있다고 생각 as 0
                print(i,'번째 ped의 cos_sim:',cos_sim)
                '''
                # 3. cosim 값을 곱하기
                #ori[imgp2 == 1] = 99   # 여기임!
                ori[imgp2 == 1] = (1-cos_sim_ped)/2 * 99   # 여기임!  (유사도 1.0(동일방향:) -10보너스, 유사도 0.0(90도): 0(변화x), 유사도 -1(역방향) = 10페널티)
                
                ori[imgp2 == 0] = 0   # 일반(uncertain)
                
                original_flow_map = original_flow_map + ori
            self.flow_map = original_flow_map    
            

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion


#def main():
def main(r_x=-4, r_y=-4, g_x=4, g_y=4, vx = 1, vw = 0, pedsim_agents_list=[]):
    print(__file__ + " start!!")

    # start and goal position
    '''
    sx = -40.0  # [m]
    sy = -40.0  # [m]
    gx = 40.0  # [m]
    gy = 40.0  # [m]
    grid_size = 1.0  # [m]
    robot_radius = 5.0  # [m]
    '''
    map_bias = 5.5
    resolution = 0.1
    grid_size = 1.0  # [m]
    robot_radius = 5.0  # [m]
    sx = int((r_x+map_bias)/resolution)
    sy = int((r_y+map_bias)/resolution)
    gx = int((g_x+map_bias)/resolution)
    gy = int((g_y+map_bias)/resolution)
    

    # set obstacle positions
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
        for j in range(45, 75):
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

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    pedsim_agents_list = [[-1, -1.5, 0, 1]]
    rx, ry = a_star.planning(sx, sy, gx, gy, vx, vw, pedsim_agents_list)

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()
        
    # 230119 최종 패스 출력
    final_path = []
    for path in zip (rx, ry):
        final_path.append([path[0], path[1]])
    final_path.reverse()
    #print(final_path)
    final_path = np.array(final_path)
    final_path = final_path / 10
    return final_path




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
            #contour_points.append((x, y))
            contour_points.append((int(x), int(y)))
                #print('컨투어 포인트:',j,x,y)

        # Get the convex hull of all the personal spaces
        

        return contour_points    
    
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
    
def check_in_poly(pts, poly):
    count = np.zeros(pts.shape[0])
    for i, _ in enumerate(poly[:-1]):
        # pts.y should be within y limits of line and pts.x should be less than intersection.x
        intersection_x = (poly[i+1,0] - poly[i,0]) * (pts[:,1]-poly[i,1]) / (poly[i+1,1] - poly[i,1]) + poly[i,0]
        count += (((pts[:,1] > poly[i,1]) != (pts[:,1] > poly[i+1,1])) & (pts[:,0] < intersection_x))*1
    return count % 2 # point is outside if even number of intersections


def check(p1, p2, base_array):
    """
    Uses the line defined by p1 and p2 to check array of 
    input indices against interpolated value

    Returns boolean array, with True inside and False outside of shape
    """
    idxs = np.indices(base_array.shape) # Create 3D array of indices

    p1 = p1.astype(float)
    p2 = p2.astype(float)

    # Calculate max column idx for each row idx based on interpolated line between two points
    max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) +  p1[1]    
    sign = np.sign(p2[0] - p1[0])
    return idxs[1] * sign <= max_col_idx * sign

def create_polygon(shape, vertices):
    """
    Creates np.array with dimensions defined by shape
    Fills polygon defined by vertices with ones, all other values zero"""
    base_array = np.zeros(shape, dtype=float)  # Initialize your array of zeros

    fill = np.ones(base_array.shape) * True  # Initialize boolean array defining shape fill

    # Create check array for each edge segment, combine into fill array
    for k in range(vertices.shape[0]):
        fill = np.all([fill, check(vertices[k-1], vertices[k], base_array)], axis=0)

    # Set all values inside polygon to one
    base_array[fill] = 99

    return base_array


if __name__ == '__main__':
    main()
    
