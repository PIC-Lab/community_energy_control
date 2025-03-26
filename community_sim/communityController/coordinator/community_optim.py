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
        self.dirName = os.path.dirname(__file__)
        with open(os.path.join(self.dirName, 'transInfo.json')) as fp:
            self.transInfo = json.load(fp)

        self.numBuildings = numBuildings
        self.nsteps = nsteps

        self.reductionFactor = np.zeros((self.numBuildings, self.nsteps))
        self.predictedLoad = np.zeros((self.numBuildings, self.nsteps))
        self.predictedFlexibility = np.zeros((self.numBuildings, self.nsteps))
        self.baseLoad = np.zeros((len(self.transInfo.keys()), self.nsteps))
        self.countSinceChange = 0
        self.usagePenalty = np.zeros((self.numBuildings))
        self.adjustValues = {'flexLoad': np.zeros((self.numBuildings, self.nsteps))}

        self.overloadList = []
        self.predictedTransLoad = np.zeros((len(self.transInfo.keys()), self.nsteps))

        self.count = 0
        self.stepFrequency = 5      # Coordinator updates values every 5 minutes

        self.assessProb = AssessOptimization()
        self.adjustProb = AdjustOptimization(self.numBuildings, self.nsteps)

    def AdjustInit(self, verbose=False):
        '''
        First time run for adjust optimization with dummy parameter values. This ensures
        future, real-time runs will not have to go through the problem formulation process

        :param verbose: (bool) Should optimization solve be run with verbose output. For debugging purposes, defaults to False
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
        i = 0
        for key, value in self.transInfo.items():
            self.predictedTransLoad[i,:] = np.sum(self.predictedLoad[int(value['Buildings'][0]):int(value['Buildings'][-1]), :], axis=0)
            self.overloadList.append(self.predTransLoad[i,:] - value['rating'])
            # Check if overload occurs at any point
            if np.any(self.predTransLoad[i,:] >= 0):
                overload = True
            i += 1

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

        :param verbose: (bool) Should optimization solve be run with verbose output. For debugging purposes, defaults to False
        '''
        paramDict = {}
        paramDict['usagePenalty'] = self.usagePenalty
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
        if not(adjustValues['flexLoad'] is None):
            self.reductionFactor = adjustValues['flexLoad'] / 100.0
        else:
            print('Optimization infeasible')
            self.reductionFactor = np.ones((self.numBuildings, self.nsteps))
            self.adjustValues['flexLoad'] = np.zeros((self.numBuildings, self.nsteps))

    def Step(self):
        '''
        Calls methods that should run every step

        :return: (bool) whether or not the coordinator updated this step
        '''
        self.count += 1
        # Coordinator updates coordination signals at set timescale, not every minute
        if not(self.stepFrequency % self.count == 0):
            # Move reduction factor by one step, repeat last value
            updatedFactor = self.reductionFactor
            updatedFactor[:-1] = self.reductionFactor[1:]
            updatedFactor[-1] = self.reductionFactor[-1]
            return False
            
        self.count = 0

        self.usagePenalty -= self.stepFrequency * 0.01
        self.usagePenalty = np.clip(self.usagePenalty, a_min=0, a_max=100)

        if self.Assess():
            self.adjustValues = self.Adjust()
            self.Dispatch(self.adjustValues)

        return True
        

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