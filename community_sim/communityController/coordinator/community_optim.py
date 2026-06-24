import numpy as np
import cvxpy as cp
import json
import os
from communityController.coordinator.convex_problem import ConvexProblem
# from convex_problem import ConvexProblem

class Coordinator():
    '''
    Class for the community coordinator
    '''

    def __init__(self, buildingAliasList, nsteps, stepSize, logger):
        '''
        Constructor
        '''
        dirName = os.path.dirname(__file__)
        with open(os.path.join(dirName, 'transInfo.json')) as fp:
            self.transInfo = json.load(fp)

        self.numBuildings = len(buildingAliasList)
        self.nsteps = nsteps
        self.stepSize = stepSize
        self.logger = logger

        self.runFreq = 60
        self.count = self.runFreq

        self.predictedLoad = np.zeros((self.numBuildings, self.nsteps))
        self.predictedFlex = np.zeros((self.numBuildings, self.nsteps, 2))
        self.baseLoad = np.zeros((len(self.transInfo.keys()), self.nsteps))
        self.loadBounds = np.zeros((self.numBuildings, self.nsteps))

        self.countSinceChange = 0
        self.usagePenalty = np.zeros((self.numBuildings, self.nsteps))
    
        self.state = 0
        self.coord_queue = [[] for i in range(self.numBuildings)]
        self.queued = np.zeros(self.numBuildings)

        self.overloadList = []

        self.assessProb = AssessOptimization()
        self.transLim = [v['rating'] for k,v in self.transInfo.items()]
        self.transMap = np.zeros((len(self.transLim),self.numBuildings))
        for j, build in enumerate(buildingAliasList):
            i = 0
            while i < len(self.transInfo.keys()):
                if build in self.transInfo[f"Transformer {i+1}"]["Buildings"]:
                    self.transMap[i, j] = 1
                    break
                else:
                    i += 1
        self.logger.debug(f"transMap: {self.transMap}")
        self.adjustValues = self.ResetAdjust()
        self.boundValues = self.ResetBound()

        self.adjustBoundsProb = AdjustBoundsOptimization(self.numBuildings, self.nsteps, self.transLim, self.transMap)
        self.logger.debug("Community coordinator adjust bounds optimization")
        self.logger.debug(f"Is DPP? {self.adjustBoundsProb.prob.is_dcp(dpp=True)}")
        self.logger.debug(f"Is DCP? {self.adjustBoundsProb.prob.is_dcp(dpp=False)}")

        self.adjustProb = AdjustOptimization(self.numBuildings, self.nsteps, self.transLim, self.transMap)
        self.logger.debug("Community coordinator adjust optimization")
        self.logger.debug(f"Is DPP? {self.adjustProb.prob.is_dcp(dpp=True)}")
        self.logger.debug(f"Is DCP? {self.adjustProb.prob.is_dcp(dpp=False)}")

    def AdjustInit(self, verbose=False):
        '''
        First time run for adjust optimization with dummy parameter values. This ensures
        future, real-time runs will not have to go through the problem formulation process

        :param verbose: (bool) Should optimization solve be run with verbose output. For debugging purposes, defaults to False
        '''
        self.logger.debug("Running adjust optimization init")

        paramDict = {}
        paramDict['usagePenalty'] = np.ones((self.numBuildings, self.nsteps))
        paramDict['pow_ref'] = np.ones((self.nsteps)) * 4
        paramDict['flexMin'] = np.zeros((self.numBuildings, self.nsteps))
        paramDict['flexMax'] = np.ones((self.numBuildings, self.nsteps)) * 200
        paramDict['trans_ref'] = np.ones((len(self.transInfo.keys()), self.nsteps)) * 5
        paramDict['baseLoad'] = np.ones((len(self.transInfo.keys()), self.nsteps))

        self.adjustBoundsProb.SolveProblem(paramDict, verbose=verbose)

        paramDict['predLoad'] = np.ones_like(self.predictedLoad)
        paramDict['loadBounds'] = np.ones_like(self.predictedLoad) * 100

        self.adjustProb.SolveProblem(paramDict, verbose=verbose)
        self.logger.debug("Adjust optimization initialized")
    
    def TransformerOverload(self):
        '''
        Checks if the community will overload the transformer in the near future
        '''
        predTransLoad = np.zeros((len(self.transInfo.keys()), self.nsteps))
        self.overloadList = []
        overload = False
        i = 0
        for key, value in self.transInfo.items():
            predTransLoad[i,:] = self.baseLoad[i,:] + self.transMap[i,:]@self.predictedLoad
            # predTransLoad[i,:] = self.baseLoad[i,:] + self.transMap[i,:]@self.predictedFlex[:,:,1]
            self.overloadList.append(predTransLoad[i,:] - value['rating'])
            # Check if overload occurs at any point
            if np.any((predTransLoad[i,:] - value['rating']) >= 0):
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

    def Bounds(self, verbose=False):
        '''
        '''
        paramDict = {}
        paramDict['usagePenalty'] = self.usagePenalty
        paramDict['pow_ref'] = np.ones(self.nsteps) * 300
        temp = np.ones((len(self.transInfo.keys()), self.nsteps))
        i = 0
        for key, value in self.transInfo.items():
            temp[i,:] *= value['rating']
            i += 1
        paramDict['trans_ref'] = temp
        paramDict['flexMin'] = self.predictedFlex[:,:,0]
        paramDict['flexMax'] = self.predictedFlex[:,:,1]
        paramDict['baseLoad'] = self.baseLoad
        self.adjustBoundsProb.SolveProblem(paramDict, verbose=verbose)

        boundValues = {}
        for key, value in self.adjustBoundsProb.prob.var_dict.items():
            boundValues[key] = value.value

        if not(self.adjustBoundsProb.feasible):
            self.logger.info("Coordinator bounds optimization infeasible")
            boundValues = self.ResetBound()

        return boundValues

    def Adjust(self, verbose=False):
        '''
        Adjust phase: Determine how a desired change in consumption will occur

        :param verbose: (bool) Should optimization solve be run with verbose output. For debugging purposes, defaults to False
        '''
        self.logger.debug("Coordinator adjust")
        paramDict = {}
        paramDict['usagePenalty'] = self.usagePenalty
        paramDict['pow_ref'] = np.ones(self.nsteps) * 300
        temp = np.ones((len(self.transInfo.keys()), self.nsteps))
        i = 0
        for key, value in self.transInfo.items():
            temp[i,:] *= value['rating']
            i += 1
        paramDict['trans_ref'] = temp
        paramDict['flexMin'] = self.predictedFlex[:,:,0]
        paramDict['flexMax'] = self.predictedFlex[:,:,1]
        paramDict['predLoad'] = self.predictedLoad
        paramDict['baseLoad'] = self.baseLoad
        paramDict['loadBounds'] = self.boundValues['buildBoundLoad']
        self.adjustProb.SolveProblem(paramDict, verbose=verbose)

        adjustValues = {}
        for key, value in self.adjustProb.prob.var_dict.items():
            adjustValues[key] = value.value

        if not(self.adjustProb.feasible):
            self.logger.info("Coordinator optimization infeasible")
            adjustValues = self.ResetAdjust()
        # adjustValues['flexLoad'] = np.round(adjustValues['flexLoad'], 3)
        return adjustValues

    def Dispatch(self, boundValues, adjustValues):
        '''
        Dispatch phase: Format desired consumption changes as control signals to be provided to each house
        '''
        self.logger.debug("Coordinator dispatch")
        if boundValues is None:
            for key, value in self.boundValues.items():
                self.boundValues[key][:-1] = self.boundValues[key][1:]
        else:
            for key, value in boundValues.items():
                self.boundValues[key] = value

        if adjustValues is None:
            for key, value in self.adjustValues.items():
                self.adjustValues[key][:-1] = self.adjustValues[key][1:]
        else:
            for key, value in adjustValues.items():
                self.adjustValues[key] = value
        self.adjustValues['flexLoad'] = self.boundValues['buildBoundLoad'] + self.adjustValues['flexPredLoad']



        # self.usagePenalty[:,:-1] = self.usagePenalty[:,1:]
        # self.usagePenalty[:,-1] = np.sum(self.adjustValues['flexLoad'], axis=1) \
        #                           + (self.usagePenalty[:,-2] * pow(0.99,self.stepSize))
        # self.usagePenalty = np.clip(self.usagePenalty, a_min=0, a_max=100)

    def Step(self):
        '''
        Calls methods that should run every step

        :return: (bool) whether or not the coordinator updated this step
        '''
        self.logger.debug("Coordinator bounds adjust check")
        if self.count >= self.runFreq:
            # boundValues = self.Bounds()
            pass
        else:
            boundValues = None

        self.logger.debug("Coordinator assess check")
        if (self.Assess()) or (self.count >= self.runFreq):
            self.state = 1
            # adjustValues = self.Adjust()
            self.count = 0
        else:
            self.state = 0
            # adjustValues = self.ResetValues()
            adjustValues = None
        boundValues = self.Bounds()
        adjustValues = self.Adjust()

        # adjustValues['queue'] = adjustValues['flexPredLoad'].copy()
        for i in range(self.numBuildings):
            if self.queued[i]:
                # self.coord_queue[i] = np.maximum(np.abs(self.coord_queue[i], adjustValues['flexPredLoad'][i,:len(self.coord_queue[i])]))
                adjustValues['flexPredLoad'][i,:len(self.coord_queue[i])] = self.coord_queue[i]
                self.coord_queue[i] = self.coord_queue[i][1:]
                if len(self.coord_queue[i]) == 0:
                    self.queued[i] = 0
            else:
                if np.abs(adjustValues['flexPredLoad'][i,0]) >= 0.01:
                    self.queued[i] = 1
                    end_idx = np.where(np.abs(adjustValues['flexPredLoad'][i]) < 0.01)
                    if len(end_idx) > 0:
                        end_idx = end_idx[0][0]
                    else:
                        end_idx = self.nsteps-1
                    self.coord_queue[i] = adjustValues['flexPredLoad'][i,:end_idx]

        self.Dispatch(boundValues, adjustValues)

        self.count += self.stepSize

        return True
    
    def ResetAdjust(self):
        '''
        Method for resetting the values used by the other components of the coordination system (like flexLoad)
        '''
        adjustValues = {}
        # adjustValues['flexLoad'] = adjustValues['flexBoundLoad'] + adjustValues['flexPredLoad']
        adjustValues['flexPredLoad'] = np.zeros((self.numBuildings,self.nsteps))
        adjustValues['buildPredLoad'] = self.predictedLoad + adjustValues['flexPredLoad']
        adjustValues['transPredLoad'] = self.transMap@self.predictedLoad
        return adjustValues
    
    def ResetBound(self):
        adjustValues = {}
        adjustValues['flexBoundLoad'] = np.zeros((self.numBuildings,self.nsteps))
        adjustValues['buildBoundLoad'] = self.predictedFlex[:,:,1] - adjustValues['flexBoundLoad']
        adjustValues['transBoundLoad'] = self.transMap@adjustValues['buildBoundLoad']
        return adjustValues


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

class AdjustBoundsOptimization(ConvexProblem):
    '''
    '''

    def __init__(self, numBuildings, nsteps, transLimits, transMap):
        '''
        '''
        super().__init__()

        self.numBuildings = numBuildings
        self.n = nsteps
        self.transLimits = transLimits
        self.transMap = transMap

        # self.DefineProblem()
        self.DefineProblem()

    def DefineProblem(self):
        '''
        Define the optimization problem used in the adjust phase to choose which buildings will utilize their flexibility
        '''
        transNum = len(self.transLimits)

        flexLoad = cp.Variable((self.numBuildings, self.n), name='flexBoundLoad', nonneg=True)
        buildLoad = cp.Variable((self.numBuildings, self.n), name='buildBoundLoad')
        transLoad = cp.Variable((transNum, self.n), name='transBoundLoad')

        usagePenalty = cp.Parameter((self.numBuildings, self.n), name='usagePenalty')
        dr_ref = cp.Parameter(self.n, name='pow_ref')
        trans_ref = cp.Parameter((transNum, self.n), name='trans_ref', nonneg=True)
        baseLoad = cp.Parameter((transNum, self.n), name='baseLoad', nonneg=True)

        W1 = 2
        W2 = 1

        flexMax = cp.Parameter((self.numBuildings, self.n), name='flexMax')
        flexMin = cp.Parameter((self.numBuildings, self.n), name='flexMin')

        objective = cp.Minimize(W1*cp.sum_squares(usagePenalty.T@flexLoad)
                                +W2*cp.sum_squares(flexLoad))
        constraints = []
        constraints.append(buildLoad >= flexMin)
        constraints.append(buildLoad == flexMax - flexLoad)
        constraints.append(transLoad == self.transMap@buildLoad)
        constraints.append(transLoad + baseLoad <= trans_ref)
        constraints.append(cp.sum(transLoad, axis=0) <= dr_ref)

        self.prob = cp.Problem(objective, constraints)

class AdjustOptimization(ConvexProblem):
    '''
    Optimization problem for the adjust phase
    '''

    def __init__(self, numBuildings, nsteps, transLimits, transMap):
        '''
        Constructor
        '''
        super().__init__()

        self.numBuildings = numBuildings
        self.n = nsteps
        self.transLimits = transLimits
        self.transMap = transMap

        # self.DefineProblem()
        self.DefineProblem()

    def DefineProblem(self):
        '''
        Define the optimization problem used in the adjust phase to choose which buildings will utilize their flexibility
        '''
        transNum = len(self.transLimits)

        flexLoad = cp.Variable((self.numBuildings, self.n), name='flexPredLoad')
        buildLoad = cp.Variable((self.numBuildings, self.n), name='buildPredLoad')
        transLoad = cp.Variable((transNum, self.n), name='transPredLoad')

        usagePenalty = cp.Parameter((self.numBuildings, self.n), name='usagePenalty')
        loadBounds = cp.Parameter((self.numBuildings, self.n), name='loadBounds')
        dr_ref = cp.Parameter(self.n, name='pow_ref')
        trans_ref = cp.Parameter((transNum, self.n), name='trans_ref', nonneg=True)
        predLoad = cp.Parameter((self.numBuildings, self.n), name='predLoad')
        baseLoad = cp.Parameter((transNum, self.n), name='baseLoad', nonneg=True)

        W1 = 2
        W2 = 1
        W3 = 1

        flexMin = cp.Parameter((self.numBuildings, self.n), name='flexMin')

        objective = cp.Minimize(W1*cp.sum_squares(usagePenalty.T@flexLoad)
                                +W2*cp.sum_squares(flexLoad))
        constraints = []
        constraints.append(buildLoad == predLoad + flexLoad)
        constraints.append(buildLoad >= flexMin)
        constraints.append(buildLoad <= loadBounds)
        constraints.append(transLoad == self.transMap@buildLoad)
        constraints.append(transLoad + baseLoad <= trans_ref)
        constraints.append(transLoad + baseLoad <= self.transMap@loadBounds)
        constraints.append(cp.sum(transLoad, axis=0) <= dr_ref)

        self.prob = cp.Problem(objective, constraints)