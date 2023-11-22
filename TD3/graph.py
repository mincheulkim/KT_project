# ref. https://www.vtupulse.com/artificial-intelligence/implementation-of-a-star-search-algorithm-in-python/


def aStarAlgo(start_node, stop_node):
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
            if n == None or g[v] + heuristic(v) < g[n] + heuristic(n):
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
            ####print('Path found: {}'.format(path))
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
def heuristic(n):    # 각 노드별 휴리스틱 값
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

aStarAlgo('A', 'L')