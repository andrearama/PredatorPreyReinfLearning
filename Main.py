from Grid import Grid
from ExtractInfo import ExtractInfo
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gridDim', default=50,type = int, help='Size of the grid')
    parser.add_argument('--nPredators', default=100,type = int, help='Number of initial predators')
    parser.add_argument('--nPrey', default=400, type = int, help='Number of initial preys')
    parser.add_argument('--nGrass', default=1200, type = int, help='Number of initial grass')
    parser.add_argument('--learningRate', default=0.05, type = int, help='learning rate of RL')
    parser.add_argument('--discountFactor', default=1, type = int, help='Discount factor of RL')
    parser.add_argument('--predRepAge', default=10, type = int, help='Reproduction Age of predators')
    parser.add_argument('--predDeathRate', default=0.019, type = int, help='Probability of dying by hunger')
    parser.add_argument('--predRepRate', default=0.1, type = int, help='Probabiliy of giving birth')
    parser.add_argument('--preyRepAge', default=8, type = int, help='Reproduction Age of preys')
    parser.add_argument('--preyDeathRate', default=0.05, type = int, help='Probability of dying by hunger')
    parser.add_argument('--preyRepRate', default=0.1, type = int, help='Probabiliy of giving birth')
    parser.add_argument('--mPred', default=10, type = int)
    parser.add_argument('--mPrey', default=8, type = int)
    parser.add_argument('--grassRepRate', default=0.02, type = int, help='Probabiliy of giving birth')
    parser.add_argument('--grassConsRate', default=0.2, type = int, help='How much it gets consumed when eaten')
    parser.add_argument('--numLearningIterations', default=500, type = int, help='Time in which the agents can learn')
    parser.add_argument('--totalNumIterations', default=5000, type = int)
    
    args = parser.parse_args()


    xDim = args.gridDim
    yDim = args.gridDim
    nPredators = args.nPredators
    nPrey = args.nPrey
    nGrass = args.nGrass
    learningRate = args.learningRate
    discountFactor = args.discountFactor
    predRepAge = args.predRepAge
    predDeathRate = args.predDeathRate
    predRepRate = args.predRepRate
    preyRepAge = args.preyRepAge
    preyDeathRate = args.preyDeathRate
    preyRepRate = args.preyRepRate
    mPred = args.mPred
    mPrey = args.mPrey
    grassRepRate = args.grassRepRate
    grassConsRate = args.grassConsRate
    
    numLearningIterations = args.numLearningIterations
    totalNumIterations = args.totalNumIterations
    
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
