import numpy as np
import cvxpy as cp
import json
import os
from communityController.coordinator.convex_problem import ConvexProblem

class Coordinator():
    '''
    Class for the community coordinator
    '''

    def __init__(self, numBuildings, nsteps, stepFreq):
        '''
        Constructor
        '''
        self.dirName = os.path.dirname(__file__)
        with open(os.path.join(self.dirName, 'transInfo.json')) as fp:
            self.transInfo = json.load(fp)

        self.numBuildings = numBuildings
        self.nsteps = nsteps
        self.stepFrequency = stepFreq

        self.predictedLoad = np.zeros((self.numBuildings, self.nsteps))
        self.predictedFlexibility = np.zeros((self.numBuildings, self.nsteps))
        self.baseLoad = np.zeros((len(self.transInfo.keys()), self.nsteps))
        self.countSinceChange = 0
        self.usagePenalty = np.zeros((self.numBuildings))
        self.adjustValues = {'flexLoad': np.zeros((self.numBuildings, self.nsteps))}

        self.overloadList = []
        self.predictedTransLoad = np.zeros((len(self.transInfo.keys()), self.nsteps))

        self.assessProb = AssessOptimization()
        transLim = [v['rating'] for k,v in self.transInfo.items()]
        self.adjustProb = AdjustOptimization(self.numBuildings, self.nsteps, transLim)

    def AdjustInit(self, verbose=False):
        '''
        First time run for adjust optimization with dummy parameter values. This ensures
        future, real-time runs will not have to go through the problem formulation process

        :param verbose: (bool) Should optimization solve be run with verbose output. For debugging purposes, defaults to False
        '''
        paramDict = {}
        paramDict['usagePenalty'] = np.zeros((self.numBuildings))
        paramDict['dr_ref'] = np.zeros((self.nsteps))
        paramDict['flexMin'] = np.zeros((self.numBuildings, self.nsteps))
        paramDict['flexMax'] = np.ones((self.numBuildings, self.nsteps)) * 15
        paramDict['trans_ref'] = np.ones((len(self.transInfo.keys()), self.nsteps))
        paramDict['predLoad'] = np.ones_like(self.predictedLoad)
        paramDict['baseLoad'] = np.ones((len(self.transInfo.keys()), self.nsteps))
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
            self.overloadList.append(self.predictedTransLoad[i,:] - value['rating'])
            # Check if overload occurs at any point
            if np.any(self.predictedTransLoad[i,:] >= 0):
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
        # paramDict['dr_ref'] = -1 * (np.sum(self.overloadList[0], axis=0) + np.sum(self.overloadList[1], axis=0))
        paramDict['dr_ref'] = np.ones(self.nsteps) * 100
        temp = np.ones((len(self.transInfo.keys()), self.nsteps))
        i = 0
        for key, value in self.transInfo.items():
            temp[i,:] *= value['rating']
            i += 1
        paramDict['trans_ref'] = temp
        paramDict['flexMin'] = np.zeros((self.numBuildings, self.nsteps))
        paramDict['flexMax'] = np.ones((self.numBuildings, self.nsteps)) * 100
        paramDict['predLoad'] = self.predictedLoad
        paramDict['baseLoad'] = np.ones((len(self.transInfo.keys()), self.nsteps))
        self.adjustProb.SolveProblem(paramDict, verbose=verbose)
        adjustValues = self.adjustProb.GetValues()
        return adjustValues

    def Dispatch(self, adjustValues):
        '''
        Dispatch phase: Format desired consumption changes as control signals to be provided to each house
        '''
        if not(adjustValues['flexLoad'] is None):
            self.usagePenalty = np.abs(self.predictedLoad - adjustValues['flexLoad'])
            self.usagePenalty = self.usagePenalty / np.sum(self.usagePenalty)
        else:
            print('Optimization infeasible')
            self.adjustValues['flexLoad'] = np.zeros((self.numBuildings, self.nsteps))

    def Step(self):
        '''
        Calls methods that should run every step

        :return: (bool) whether or not the coordinator updated this step
        '''

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

    def __init__(self, numBuildings, nsteps, transLimits):
        '''
        Constructor
        '''
        super().__init__()

        self.numBuildings = numBuildings
        self.n = nsteps
        self.transLimits = transLimits

        self.DefineProblem()

    def DefineProblem(self):
        '''
        Define the optimization problem used in the adjust phase to choose which buildings will utilize their flexibility
        '''
        trans_map = np.ones((2,self.numBuildings))

        flexLoad = cp.Variable((self.numBuildings, self.n), name='flexLoad')

        usagePenalty = cp.Parameter(self.numBuildings, name='usagePenalty')
        dr_ref = cp.Parameter(self.n, name='dr_ref')
        trans_ref = cp.Parameter((len(self.transLimits), self.n), name='trans_ref')
        predLoad = cp.Parameter((self.numBuildings, self.n), name='predLoad')
        baseLoad = cp.Parameter((len(self.transLimits), self.n), name='baseLoad')
        W1 = 2
        W2 = 1

        flexMax = cp.Parameter((self.numBuildings, self.n), name='flexMax')
        flexMin = cp.Parameter((self.numBuildings, self.n), name='flexMin')

        objective = cp.Minimize(W1*usagePenalty@flexLoad@np.ones(self.n)+W2*cp.max(flexLoad, axis=0)@np.ones(self.n))
        constraints = []
        constraints.append(flexLoad + predLoad <= flexMax)
        constraints.append(flexLoad + predLoad >= flexMin)
        constraints.append(cp.sum(flexLoad+predLoad, axis=0)+cp.sum(baseLoad, axis=0) <= dr_ref)
        for i in range(0, len(self.transLimits)):
            constraints.append(trans_map[i,:]@flexLoad+baseLoad[i,:] <= trans_ref[i,:])

        self.prob = cp.Problem(objective, constraints)

        print("Is DPP?", self.prob.is_dcp(dpp=True))
        print("Is DCP?", self.prob.is_dcp(dpp=False))

    def GetValues(self):
        '''
        '''
        values = {}
        values['flexLoad'] = self.FindVariableByName('flexLoad').value
        return values