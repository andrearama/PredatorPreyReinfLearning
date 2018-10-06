#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 11:23:11 2018

@author: andrea
"""

def ExtractInfo(grid):
    import numpy as np
    import pandas as pd
     
    matrix = [];
     
    for agentInfo in grid.agentList:
        agentId = agentInfo[0]
        x = agentInfo[1]
        y = agentInfo[2]
        agentType = agentInfo[3]
        agent = None
        #Get current agent based on id
        for agents in grid.grid[x][y]:
            if agents.ID == agentId:
                agent = agents
                break           
             
        if agentType != 2:               #Predator
            v = agent.weights
            v = v.tolist() + [agentType]
            matrix.append(v)
    #Convert list of lists to np.matrix and then in pd.Dataframe   
       
    matrix = np.matrix(matrix)
    matrix = pd.DataFrame(matrix)         
    return matrix 