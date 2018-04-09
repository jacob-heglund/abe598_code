# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 18:07:18 2018

@author: Jacob
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 17:03:17 2018

@author: Jacob
"""

# HW1 Problem 7, Breadth First Search
# import libraries
import networkx as nx
import matplotlib.pyplot as plt
import timeit

def bfs(g, start, end, plot):   
    # returns finalPath
    ##############################
    # initialize some data structures
    ##############################
    # the list of nodes we pass through to reach the final node
    path = [start]
    
    # a list of "next steps" to take as we traverse the graph
    # it takes the form queue = [(node, [path it took to get to the node])]
    # queue[n][0] = nth node in the queue
    # queue[n][1] = path to the nth node from the start
    #queue = [(start, path)]
    queue = [(start)]
        
    # a list of all visited nodes to avoid repeats
    visitedNodes = []
    
    ##############################
    # the main loop, as long as there are nodes in the queue
    # the algorithm will keep going
    ##############################     
    while len(queue) != 0:
        
        # take the first element of the queue as our current position
        # and remove it from the queue
        currPosition = queue.pop(0)
        
        currNode = currPosition
        
        #currPath = currPosition[1] 
        
        # look around at the neighbors of our current node
        for neighbor in g.neighbors(currNode):
            if neighbor not in visitedNodes:
                
                # make sure we don't visit a particular node twice
                visitedNodes.append(currNode)
                
                
                # find the paths  neighbor nodes
                #newNodePath = list(currPath)
                #newNodePath.append(neighbor)
            
                # keep the new node and its path in a tuple to be added to the queue
                #newNode = (neighbor, newNodePath)
                newNode = (neighbor)
                queue.append(newNode)
                
                # add an ending condion so that once we reach the end node, the 
                # loop stops
                #if currNode == end:
                    #finalPath = currPath
    
    ##############################
    # plot the graph, make it look nice
    ##############################  
    if plot:
        pos=nx.get_node_attributes(g,'pos')
        nx.draw(g,pos)
        labels = nx.get_edge_attributes(g,'weight')
        nx.draw_networkx_edge_labels(g,pos,edge_labels=labels)
        plt.show()
    
    #return finalPath

'''
##############################
# find the total cost of a path using BFS
##############################
def bfsPathCost(g, start, end):    
    path = bfs(g, start, end, plot=0)
    i = 0
    pathCost = 0
    while i < len(path)-1:
        # find the weight of a single edge in our path and add it to 
        # the total cost of traversing the path
        edgeWeight = g.get_edge_data(path[i], path[i+1])
        for edge in edgeWeight:
            pathCost += edgeWeight[edge]
        i+=1
    return pathCost
'''  
# create the graph and the nodes
g = nx.Graph()
n = 5 # square graph
# add the nodes
# (col, row)
i = 1
while i < n+1:
    j = 1
    while j < n+1:
        g.add_node((i,j), pos = (i,j))
        j+=1
    i+=1

# add the edges
i = 1
while i < n+1:
    if i < n:
        j = 1
        while j < n+1:
            g.add_edge((i,j), (i+1, j))
            j+=1
    i+=1 

i = 1
while i < n+1:
    if i < n:
        j = 1
        while j < n:
            g.add_edge((i,j), (i, j+1))
            j+=1
    i+=1 
g.add_edge((5,1), (5,2))
g.add_edge((5,2), (5,3))
g.add_edge((5,3), (5,4))
g.add_edge((5,4), (5,5))

'''
pos=nx.get_node_attributes(g,'pos')
nx.draw(g,pos)
labels = nx.get_edge_attributes(g,'weight')
nx.draw_networkx_edge_labels(g,pos,edge_labels=labels)
plt.show()
'''
if __name__ == "__main__":    
    path1 = bfs(g,(1,1), (5,5),0)
'''
if __name__ == "__main__":    
    startTime = timeit.default_timer()
    path1 = bfs(g,(1,1), (5,5),0)
    cost1 = bfsPathCost(g, 1, 7)
    
    endTime = timeit.default_timer()
     find the runtime of the program in seconds
    runTime = (endTime-startTime)*10**6
    
    print('The verticies found by BFS: ' + str(path1))
    print('Cost of BFS: ' + str(cost1))
    print('Runtime in microseconds: ' + str(runTime))
'''
























































