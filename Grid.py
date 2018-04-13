import random
from GrassAgent import *
from Classes_Agents import *
from matplotlib import pyplot as plt


class Grid:

    def __init__(self, xDim, yDim, nPredators, nPrey, nGrass, learningRate, discountFactor,
                 predRepAge, predDeathRate, predRepRate, preyRepAge, preyDeathRate, preyRepRate,
                 mPred, mPrey, grassRepRate, grassConsRate):
        self.xDim = xDim
        self.yDim = yDim
        self.nPredators = nPredators
        self.nPrey = nPrey
        self.grid = [[[] for x in range(xDim)] for y in range(yDim)]
        self.grassGrid = [[0 for x in range(xDim)] for y in range(yDim)]
        self.agentList = []
        self.ID = 1
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.predRepAge = predRepAge
        self.predDeathRate = predDeathRate
        self.predRepRate = predRepRate
        self.preyRepAge = preyRepAge
        self.grassRepRate = grassRepRate
        self.grassConsRate = grassConsRate
        self.mPred = mPred
        self.mPrey = mPrey
        self.preyDeaths = 0
        self.predDeaths = 0
        self.preyDeathAge = 0
        self.predDeathAge = 0
        self.numPred = nPredators
        self.numPrey = nPrey
        self.numGrass = nGrass
        for i in range(nPredators):
            initWeights = np.random.rand(12)*6 - 3
            x = random.randint(0, xDim - 1)
            y = random.randint(0, yDim - 1)
            pred = Predator(x, y, self.ID, 0, -1, predRepAge, predDeathRate,
                            predRepRate, initWeights, learningRate, discountFactor, mPred)
            self.grid[x][y].append(pred)
            self.agentList.append([self.ID, x, y, 0])
            self.ID += 1
        for i in range(nPrey):
            initWeights = np.random.rand(12)*6 - 3
            x = random.randint(0, xDim - 1)
            y = random.randint(0, yDim - 1)
            prey = Prey(x, y, self.ID, 0, -1, preyRepAge, preyDeathRate,
                        preyRepRate, initWeights, learningRate, discountFactor, mPrey)
            self.grid[x][y].append(prey)
            self.agentList.append([self.ID, x, y, 1])
            self.ID += 1
        for i in range(nGrass):
            x = random.randint(0, xDim - 1)
            y = random.randint(0, yDim - 1)
            while self.grassGrid[x][y] == 1:
                x = random.randint(0, xDim - 1)
                y = random.randint(0, yDim - 1)
            grass = Grass(x, y, self.grassRepRate, self.grassConsRate)
            grass.ID = self.ID
            self.grid[x][y].append(grass)
            self.grassGrid[x][y] = 1
            self.agentList.append([self.ID, x, y, 2])
            self.ID += 1

    def update(self, learning, i):
        random.shuffle(self.agentList)
        predLastAte=0
        preyLastAte=0
        deathsbyeat=0
        deathsbystv=0
        for agentInfo in self.agentList:
            agentId = agentInfo[0]
            x = agentInfo[1]
            y = agentInfo[2]
            agentType = agentInfo[3]
            agent = None
            #Get current agent based on id
            for agents in self.grid[x][y]:
                if agents.ID == agentId:
                    agent = agents
                    break
            #Move agents, add eating etc.
            if agentType == 0:               #Predator
                predLastAte =predLastAte + agent.lastAte
                agent.Aging(i)
                # Moving and learning
                [newCoordsX, newCoordsY] = agent.Change_Position(self)
                newCoordsX = int(newCoordsX)
                newCoordsY = int(newCoordsY)
                self.grid[x][y].remove(agent)
                agent.x_position = newCoordsX
                agent.y_position = newCoordsY
                self.grid[newCoordsX][newCoordsY].append(agent)
                agentInfo[1] = newCoordsX
                agentInfo[2] = newCoordsY
                x = newCoordsX
                y = newCoordsY
                if learning:
                    r=agent.Get_Reward(self)
                    agent.Update_Weight(r, self, agent.q)
                eatenID = agent.Eat(self.grid[x][y])
                if eatenID != -1:
                    for agents in self.grid[x][y]:
                        if agents.ID == eatenID:
                            eatenAgent = agents
                            break
                    self.preyDeathAge += eatenAgent.age
                    self.preyDeaths += 1
                    self.numPrey -= 1
                    self.grid[x][y].remove(eatenAgent)
                    deathsbyeat=deathsbyeat+1
                    for agentProperties in self.agentList:
                        if agentProperties[0] == eatenID:
                            eatenAgent = agentProperties
                            break
                    self.agentList.remove(eatenAgent)
                else:
                    if agent.Starve() != -1:
                        self.predDeathAge += agent.age
                        self.predDeaths += 1
                        self.numPred -= 1
                        self.grid[x][y].remove(agent)
                        self.agentList.remove(agentInfo)
                    else:
                        offspring = agent.Reproduce()
                        if offspring != 0:
                            offspring.ID = self.ID
                            offspring.epsilon = agent.epsilon
                            self.numPred += 1
                            self.grid[x][y].append(offspring)
                            self.agentList.append([self.ID, x, y, 0])
                            self.ID += 1
            elif agentType == 1:
                preyLastAte =preyLastAte + agent.lastAte
                agent.Aging(i)
                #Monving and learning
                [newCoordsX, newCoordsY] = agent.Change_Position(self)
                newCoordsX = int(newCoordsX)
                newCoordsY = int(newCoordsY)
                self.grid[x][y].remove(agent)
                agent.x_position = newCoordsX
                agent.y_position = newCoordsY
                self.grid[newCoordsX][newCoordsY].append(agent)
                agentInfo[1] = newCoordsX
                agentInfo[2] = newCoordsY
                x = newCoordsX
                y = newCoordsY
                if learning:
                    r=agent.Get_Reward(self)
                    agent.Update_Weight(r, self, agent.q)
                eatenID = agent.Eat(self.grid[x][y])
                if eatenID != -1:
                    for agents in self.grid[x][y]:
                        if agents.ID == eatenID:
                            eatenAgent = agents
                            break
                    self.grid[x][y].remove(eatenAgent)
                    self.grassGrid[x][y] = 0
                    self.numGrass -= 1
                    for agentProperties in self.agentList:
                        if agentProperties[0] == eatenID:
                            eatenAgent = agentProperties
                            break
                    self.agentList.remove(eatenAgent)
                else:
                    if agent.Starve() != -1:
                        self.preyDeathAge += agent.age
                        self.preyDeaths += 1
                        self.numPrey -= 1
                        self.grid[x][y].remove(agent)
                        self.agentList.remove(agentInfo)
                        deathsbystv=deathsbystv+1
                    else:
                        offspring = agent.Reproduce()
                        if offspring != 0:
                            offspring.ID = self.ID
                            offspring.epsilon = agent.epsilon
                            self.numPrey += 1
                            self.grid[x][y].append(offspring)
                            self.agentList.append([self.ID, x, y, 1])
                            self.ID += 1
            elif agentType == 2:
                offspring = agent.update()
                if offspring != 0:
                    offspring.ID = self.ID
                    coords = self.getGrassCoords(x, y)
                    if coords != 0:
                        offspring.x = coords[0]
                        offspring.y = coords[1]
                        self.grassGrid[coords[0]][coords[1]] = 1
                        self.numGrass += 1
                        self.grid[np.mod(offspring.x, self.xDim)][np.mod(offspring.y, self.yDim)].append(offspring)
                        self.agentList.append([self.ID, np.mod(offspring.x, self.xDim), np.mod(offspring.y, self.yDim), 2])
                        self.ID += 1
        if self.preyDeaths != 0:
            preyDeathAvg = self.preyDeathAge / self.preyDeaths
        else:
            preyDeathAvg = 0
        if self.predDeaths != 0:
            predDeathAvg = self.predDeathAge / self.predDeaths
        else:
            predDeathAvg = 0
        if self.numPrey != 0:     
            preyLastAteP = float(preyLastAte)/float(self.numPrey)   
        else:
            preyLastAteP = 0
        if self.numPred != 0:
            predLastAteP = float(predLastAte)/float(self.numPred)   
        else:
            predLastAteP = 0
        ratio = deathsbyeat/(deathsbystv+deathsbyeat+0.00000000001)    
        return [self.numPred, self.numPrey, self.numGrass, preyDeathAvg, predDeathAvg,preyLastAteP,predLastAteP,ratio]

    def draw(self):
        plt.clf()
        xs = [[], [], []]
        ys = [[], [], []]
        for agents in self.agentList:
            x = agents[1]
            y = agents[2]
            type = agents[3]
            if type == 0:
                xs[0].append(x)
                ys[0].append(y)
            elif type == 1:
                xs[1].append(x)
                ys[1].append(y)
            else:
                xs[2].append(x)
                ys[2].append(y)
        plt.scatter(xs[2], ys[2], color='g')
        plt.scatter(xs[1], ys[1], color='b')
        plt.scatter(xs[0], ys[0], color='r')
        plt.axis([-1, self.xDim, -1, self.yDim])
        plt.pause(0.01)
        plt.draw()

    def getGrassCoords(self, x, y):
        availablePositions = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                xi = (x + i) % self.xDim
                yi = (y + j) % self.yDim
                if self.grassGrid[xi][yi] == 0:
                    availablePositions.append([xi, yi])
        if len(availablePositions) > 0:
            index = random.randint(0, len(availablePositions) - 1)
            return availablePositions[index]
        else:
            return 0