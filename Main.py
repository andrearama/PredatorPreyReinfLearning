from Grid import *
import pandas as pd
from ExtractInfo import ExtractInfo

xDim = 50
yDim = 50
nPredators = 100
nPrey = 400
nGrass = 1200
learningRate = 0.05
discountFactor = 1
predRepAge = 10
predDeathRate = 0.019
predRepRate = 0.1
preyRepAge = 8
preyDeathRate = 0.05
preyRepRate = 0.1
mPred = 10
mPrey = 8
grassRepRate = 0.02
grassConsRate = 0.2

numLearningIterations = 500
totalNumIterations = 5000

preyV = []
predV = []
grassV = []
predLastAteV=[]
preyLastAteV=[]
ratioV=[]

WeightsInfo = []

grid = Grid(xDim, yDim, nPredators, nPrey, nGrass, learningRate, discountFactor,
            predRepAge, predDeathRate, predRepRate, preyRepAge, preyDeathRate, preyRepRate, mPred, mPrey,
            grassRepRate, grassConsRate)

numAgents = nPredators + nPrey + nGrass

for i in range(1, numLearningIterations):
    numAgents = grid.update(True, i)
    print("Iteration: %d. Pred: %d, prey: %d, grass: %d, avg. prey death age: %.2f, avg. pred death age: %.2f" % (i, numAgents[0], numAgents[1], numAgents[2], numAgents[3], numAgents[4]))
    preyV.append(numAgents[0])
    predV.append(numAgents[1])
    grassV.append(numAgents[2])
    [preyDeathAvg, predDeathAvg,preyLastAteP,predLastAteP,ratio] = numAgents[3:]
    predLastAteV.append(predLastAteP)
    preyLastAteV.append(preyLastAteP)
    ratioV.append(ratio)
    a=ExtractInfo(grid)
    WeightsInfo.append(a)
    
i = numLearningIterations

while numAgents[0] > 0 and i <= totalNumIterations:
    numAgents = grid.update(False, i)
    print("Iteration: %d. Pred: %d, prey: %d, grass: %d, avg. prey death age: %.2f, avg. pred death age: %.2f" % (i, numAgents[0], numAgents[1], numAgents[2], numAgents[3], numAgents[4]))
    i += 1
    grid.draw()
    preyV.append(numAgents[0])
    predV.append(numAgents[1])
    grassV.append(numAgents[2])
    [preyDeathAvg, predDeathAvg,preyLastAteP,predLastAteP,ratio] = numAgents[3:]
    predLastAteV.append(predLastAteP)
    preyLastAteV.append(preyLastAteP)    
    ratioV.append(ratio)    

predFile = open('PredatorList.txt', 'w')
for i in predV:
    predFile.write("%d\n" % i)
preyFile = open('PreyList.txt', 'w')
for i in preyV:
    preyFile.write("%d\n" % i)
grassFile = open('GrassList.txt', 'w')
for i in grassV:
    grassFile.write("%d\n" % i)

plt.clf()
plt.plot(predV, 'r')
plt.plot(preyV, 'b')
plt.plot(grassV, 'g')
plt.pause(0.01)
plt.draw()
plt.savefig('Distributions.pdf')
plt.figure(2)

input('Press any key to exit\n')
