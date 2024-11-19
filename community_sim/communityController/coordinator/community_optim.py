import numpy as np
import cvxpy as cp
import json
import os
from communityController.coordinator.convex_problem import ConvexProblem

class Coordinator():
    '''
    Class for the community coordinator
    '''

    def __init__(self, numBuildings, nsteps):
        '''
        Constructor
        '''
        self.numBuildings = numBuildings
        self.nsteps = nsteps

        self.reductionFactor = np.zeros((self.numBuildings, self.nsteps))
        self.predictedLoad = np.zeros((self.numBuildings, self.nsteps))
        self.predictedFlexibility = np.zeros((self.numBuildings, self.nsteps))
        self.baseLoad = np.zeros((self.nsteps))
        self.countSinceChange = 0

        self.dirName = os.path.dirname(__file__)
        with open(os.path.join(self.dirName, 'transInfo.json')) as fp:
            self.transInfo = json.load(fp)
        self.overloadList = []

        self.assessProb = AssessOptimization()
        self.adjustProb = AdjustOptimization(self.numBuildings, self.nsteps)

    def AdjustInit(self, verbose=False):
        '''
        '''
        paramDict = {}
        paramDict['usagePenalty'] = np.zeros((self.numBuildings))
        paramDict['dr_reduce'] = np.zeros((self.nsteps))
        paramDict['flexMin'] = np.zeros((self.numBuildings, self.nsteps))
        paramDict['flexMax'] = np.ones((self.numBuildings, self.nsteps)) * 15
        self.adjustProb.SolveProblem(paramDict, verbose=verbose)
    
    def TransformerOverload(self):
        '''
        Checks if the community will overload the transformer in the near future
        '''
        self.overloadList = []
        overload = False
        for key, value in self.transInfo.items():
            temp = self.predictedLoad[int(value['Buildings'][0]):int(value['Buildings'][-1]), :] - value['rating']
            self.overloadList.append(temp)
            # Check if overload occurs at any point
            if np.any(temp >= 0):
                overload = True

        return overload

    def DemandResponse(self):
        '''
        Decides if the community should respond to a demand response signal
        '''
        return False

    def Assess(self):
        '''
        Assess phase: Determines if the community can/should adjust planned consumption
        '''

        # Can adjust?
        # Check flexibility

        # Should adjust?
        # Check transformers
        return self.TransformerOverload() or self.DemandResponse()


    def Adjust(self, verbose=False):
        '''
        Adjust phase: Determine how a desired change in consumption will occur
        '''
        paramDict = {}
        paramDict['usagePenalty'] = np.ones((self.numBuildings))
        paramDict['dr_reduce'] = -1 * (np.sum(self.overloadList[0], axis=0) + np.sum(self.overloadList[1], axis=0))
        paramDict['flexMin'] = np.zeros((self.numBuildings, self.nsteps))
        paramDict['flexMax'] = np.ones((self.numBuildings, self.nsteps)) * 100
        self.adjustProb.SolveProblem(paramDict, verbose=verbose)
        adjustValues = self.adjustProb.GetValues()
        return adjustValues

    def Dispatch(self, adjustValues):
        '''
        Dispatch phase: Format desired consumption changes as control signals to be provided to each house
        '''
        # self.countSinceChange += 1
        # if self.countSinceChange > 10:
        #     self.reductionFactor[:10,:] = np.ones((10, self.numBuildings)) * np.random.randint(0, 2)
        #     self.countSinceChange = 0
        self.reductionFactor = adjustValues['flexLoad'] / 100.0

    def Step(self):
        '''
        '''
        if self.Assess():
            adjustValues = self.Adjust()
            self.Dispatch(adjustValues)
        

class AssessOptimization(ConvexProblem):
    '''
    Optimization problem for the assess phase
    '''

    def __init__(self):
        '''
        Constructor
        '''
        super().__init__()

        self.DefineProblem()

    def DefineProblem(self):
        '''
        Define the optimization problem used in the assess phase to decide if responding to a demand response signal is worth it for the community
        '''
        pass

class AdjustOptimization(ConvexProblem):
    '''
    Optimization problem for the adjust phase
    '''

    def __init__(self, numBuildings, nsteps):
        '''
        Constructor
        '''
        super().__init__()

        self.numBuildings = numBuildings
        self.n = nsteps

        self.DefineProblem()

    def DefineProblem(self):
        '''
        Define the optimization problem used in the adjust phase to choose which buildings will utilize their flexibility
        '''
        flexLoad = cp.Variable((self.numBuildings, self.n), name='flexLoad')

        # flexUsage = cp.Parameter((numBuildings,n), name='flexUsage')
        # usageTime = cp.Parameter((numBuildings,n), name='usageTime')
        # price = cp.Parameter(n, name='price')
        usagePenalty = cp.Parameter(self.numBuildings, name='usagePenalty')
        dr_reduce = cp.Parameter(self.n, name='dr_reduce')
        W1 = 2
        W2 = 1

        flexMax = cp.Parameter((self.numBuildings, self.n), name='flexMax')
        flexMin = cp.Parameter((self.numBuildings, self.n), name='flexMin')

        objective = cp.Minimize(W1*usagePenalty@flexLoad@np.ones(self.n)+W2*cp.max(flexLoad, axis=0)@np.ones(self.n))
        constraints = []
        constraints.append(flexLoad <= flexMax)
        constraints.append(flexLoad >= flexMin)
        constraints.append(cp.sum(flexLoad, axis=0) >= dr_reduce)

        self.prob = cp.Problem(objective, constraints)

        print("Is DPP?", self.prob.is_dcp(dpp=True))
        print("Is DCP?", self.prob.is_dcp(dpp=False))

    def GetValues(self):
        '''
        '''
        values = {}
        values['flexLoad'] = self.FindVariableByName('flexLoad').value
        return values