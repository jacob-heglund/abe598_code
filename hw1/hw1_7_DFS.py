# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 15:22:15 2018

@author: Jacob
"""
# HW1 Problem 7, Depth First Search
# import libraries
import networkx as nx
import matplotlib.pyplot as plt
import timeit

def dfs(g, start, end, plot):   
    # returns finalPath
    
    ##############################
    # initialize some data structures
    ##############################
    # the list of nodes we pass through to reach the final node
    path = [start]
    
    # a list of "next steps" to take as we traverse the graph
    # it takes the form stack = [(node, [path it took to get to the node])]
    # stack[n][0] = nth node in the stack
    # stack[n][1] = path to the nth node from the start
    stack = [(start, path)]
        
    # a list of all visited nodes to avoid repeats
    visitedNodes = []
    
    ##############################
    # the main loop, once we find a path from the starting node to the ending
    # node, the algorithm stops
    ##############################     
    calculations = 0
    done = 0
    while done == 0:
        # take the last element of the stack as our current position
        # and remove it from the stack
        currPosition = stack.pop()
        currNode = currPosition[0]
        currPath = currPosition[1] 
        
        # look around at the neighbors of our current node
        for neighbor in g.neighbors(currNode):
            if neighbor not in visitedNodes:
                # make sure we don't visit a particular node twice
                visitedNodes.append(currNode)
                
                # find the paths  neighbor nodes
                newNodePath = list(currPath)
                newNodePath.append(neighbor)
            
                # keep the new node and its path in a tuple to be added to the stack
                newNode = (neighbor, newNodePath)
                
                stack.append(newNode)
                # add an ending condion so that once we reach the end node, the 
                # loop stops
                if currNode == end:
                    finalPath = currPath
                    done = 1
                    break
            calculations += 1        
    ##############################
    # plot the graph, make it look nice
    ##############################  
    if plot:
        pos=nx.get_node_attributes(g,'pos')
        nx.draw(g,pos)
        labels = nx.get_edge_attributes(g,'weight')
        nx.draw_networkx_edge_labels(g,pos,edge_labels=labels)
        plt.show()
    
    return (finalPath, calculations)

##############################
# find the total cost of a path using BFS
##############################
def dfsPathCost(g, start, end):    
    path = dfs(g, start, end, plot=0)[0]
    i = 0
    pathCost = 0
    while i < len(path)-1:
        # find the weight of a single edge in our path and add it to 
        # the total cost of traversing the path
        edgeWeight = g.get_edge_data(path[i], path[i+1])
        for edge in edgeWeight:
            pathCost += edgeWeight[edge]
        i+=1
    return (pathCost, path)
  
# create the graph and the nodes
g = nx.Graph()

g.add_node(1, pos = (0,1))
g.add_node(2, pos = (1,1))
g.add_node(3, pos = (2,0))
g.add_node(4, pos = (2,1))
g.add_node(5, pos = (2,2))
g.add_node(6, pos = (3,1))
g.add_node(7, pos = (4,1))

# create the edges in the graph
g.add_edge(1,2, weight = 50)
g.add_edge(1,3, weight = 50)
g.add_edge(2,4, weight = 10)
g.add_edge(2,5, weight = 20)
g.add_edge(4,6, weight = 20)
g.add_edge(5,6, weight = 10)
g.add_edge(6,7, weight = 20)
g.add_edge(3,7, weight = 50)

if __name__ == "__main__":    
    startTime = timeit.default_timer()
    x = dfsPathCost(g, 1, 7)
    
    cost, path = x[0], x[1]
    
    endTime = timeit.default_timer()
    # find the runtime of the program in seconds
    runTime = (endTime-startTime)*10**6
    numcalc = dfs(g, 1, 7, 0)[1]
    print('The verticies found by DFS: ' + str(path))
    print('Cost of DFS: ' + str(cost))
    print('Runtime in microseconds: ' + str(runTime))





































