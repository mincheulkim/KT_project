import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import math
import time
import cv2
import sys

# ref. https://www.vtupulse.com/artificial-intelligence/implementation-of-a-star-search-algorithm-in-python/


def aStarAlgo(start_node, stop_node, pedsim_list):
    open_set = set(start_node)
    closed_set = set()
    g = {}               #store distance from starting node
    parents = {}         # parents contains an adjacency map of all nodes
    #distance of starting node from itself is zero
    g[start_node] = 0
    #start_node is root node i.e it has no parent nodes
    #so start_node is set to its own parent node
    parents[start_node] = start_node
    while len(open_set) > 0:
        n = None
        #node with lowest f() is found
        for v in open_set:
            if n == None or g[v] + heuristic(v, pedsim_list) < g[n] + heuristic(n, pedsim_list):
                n = v
        if n == stop_node or Graph_nodes[n] == None:
            pass
        else:
            for (m, weight) in get_neighbors(n):
                #nodes 'm' not in first and last set are added to first
                #n is set its parent
                if m not in open_set and m not in closed_set:
                    open_set.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight
                #for each node m,compare its distance from start i.e g(m) to the
                #from start through n node
                else:
                    if g[m] > g[n] + weight:
                        #update g(m)
                        g[m] = g[n] + weight
                        #change parent of m to n
                        parents[m] = n
                        #if m in closed set,remove and add to open
                        if m in closed_set:
                            closed_set.remove(m)
                            open_set.add(m)
        if n == None:
            print('Path does not exist!')
            return None
        
        # if the current node is the stop_node
        # then we begin reconstructin the path from it to the start_node
        if n == stop_node:
            path = []
            while parents[n] != n:
                path.append(n)
                n = parents[n]
            path.append(start_node)   # ~~~~~ + start_node
            path.reverse()
            #print('Path found: {}'.format(path))
            return path
        # remove n from the open_list, and add it to closed_list
        # because all of his neighbors were inspected
        open_set.remove(n)
        closed_set.add(n)
    print('Path does not exist!')
    return None

#define fuction to return neighbor and its distance
#from the passed node
def get_neighbors(v):
    if v in Graph_nodes:
        return Graph_nodes[v]
    else:
        return None
    
#for simplicity we ll consider heuristic distances given
#and this function returns heuristic distance for all nodes
def heuristic(n, pedsim_list):    # 각 노드별 휴리스틱 값
    H_dist = {
        'A': 1,
        'B': 1,
        'C': 1,
        'D': 1,
        'E': 1,
        'F': 1,
        'G': 1,
        'H': 1,
        'I': 1,
        'J': 1,
        'K': 1,
        'L': 1
    }
    
    # density 반영    
    if pedsim_list != None:
        #print('페드심 리스트:',pedsim_list)
        for i, ped in enumerate(pedsim_list):
            label = set_graph_node(ped[0], ped[1])
            if label =='A' or label =='C' or label == 'B' or label=='G':   # 221220 cctv visibile area에 따라 업데이트
                H_dist.update({label:H_dist[label]+10})  # 딕셔너리 값 업데이트 https://dojang.io/mod/page/view.php?id=2307
            
    #print('Heuristic table:',H_dist)
    return H_dist[n]

#Describe your graph here
Graph_nodes = {
    'A': [('D', 1.75),('F', 4.19076)],
    #'B': [('G', 1.767767),('K', 2.85044),('L', 3.71652),('H', 2.136),('C', 4.03113)],
    'B': [('G', 1.767767),('K', 2.85044),('H', 2.136),('C', 4.03113)],
    'C': [('B', 4.03113), ('H', 4.25), ('I', 3.91312),('E',2.657536)],
    'D': [('A', 1.75), ('G', 1.767767), ('E', 2.610077)],
    'E': [('D', 2.610077), ('C', 2.657536)],
    'F': [('A', 4.19076), ('J', 3.40037)],
    'G': [('K', 3.16228), ('B', 2.136), ('D', 4.31567)],
    'H': [('B', 2.136), ('C', 4.25), ('I', 3.640055)],
    'I': [('H', 3.640055), ('C', 3.91312)],
    'J': [('F', 3.40037), ('K', 3.75)],
    'K': [('J', 3.75), ('L', 3.75),('G',3.16228),('B',2.85044)],
    #'L': [('K', 3.75), ('B', 3.71652)],
    'L': [('K', 3.75)],
}

def set_graph_node(x, y):
    label = None
    if -5.5 <= x <= -3 and -5.5 <= y <= -1:
        label = 'A'
    elif 0<= x <= 2 and 0 <= y <= 2.5:
        label = 'B'
    elif 2<= x <= 4 and -5.5 <= y <= 1:
        label = 'C'
    elif -3 <= x <= 0 and -5.5 <= y <=-1:
        label = 'D'
    elif 0<= x <= 2 and -5.5 <= y <= -2.5:
        label = 'E'
    elif -5.5 <= x <= -3.5 and -1 <= y <= 2.5:
        label = 'F'
    elif -1.5 <= x <= 0 and -1 <= y <= 2.5:
        label = 'G'
    elif 2<= x <= 4 and 1 <= y <= 2.5:
        label = 'H'
    elif 4<= x <= 5.5 and 0<= y <= 2.5:
        label = 'I'
    elif -5.5 <= x <= -1.5 and 2.5 <= y <= 5.5:
        label = 'J'
    elif -1.5 <= x <= 2.5 and 2.5 <= y <= 5.5:
        label = 'K'
    elif 2<= x <= 5 and 2.5 <= y <= 5.5:
        label = 'L'
    else:
        label = 'I'
        
    return label

def trans_paths(paths, g_x, g_y):
    seq_path = []
    for path in paths:
        if path == 'A':
            seq_path.append([-3.25, -3.25])
        elif path =='B':
            seq_path.append([1, 1.25])
        elif path =='C':
            seq_path.append([3, -2.25])
        elif path =='D':
            seq_path.append([-1.5, -3.25])
        elif path =='E':
            seq_path.append([1, -4])
        elif path =='F':
            seq_path.append([-4.5, 0.75])
        elif path =='G':
            seq_path.append([-0.75, 1])
        elif path =='H':
            seq_path.append([3, 2])
        elif path =='I':
            seq_path.append([4.75, 1.25])
        elif path =='J':
            seq_path.append([-3.5, 4])
        elif path =='K':
            seq_path.append([0.25, 4])
        elif path =='L':
            seq_path.append([3.5, 4])
        
            
    seq_path.append([g_x, g_y])
    return seq_path
    

#def main(argv=[]):
def main(r_x, r_y, g_x, g_y, pedsim_agents_list):   # [-4.5 ~ 4.5],
        
    ## start 지점을 구역에 할당
    start_label = set_graph_node(r_x, r_y)
    
    ## goal 지점을 구역에 할당
    goal_label = set_graph_node(g_x, g_y)
    ## 경로계산
    
    path = aStarAlgo(start_label, goal_label, pedsim_agents_list)   # C, B, G, D, E 이런식
    
    path_coordinate = trans_paths(path, g_x, g_y)

    return path_coordinate

    

if __name__ == "__main__":
    main(sys.argv)
    


