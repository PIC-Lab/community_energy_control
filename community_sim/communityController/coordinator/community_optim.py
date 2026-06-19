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

        self.predictedLoad = np.zeros((self.numBuildings, self.nsteps))
        self.predictedFlex = np.zeros((self.numBuildings, self.nsteps, 2))
        self.baseLoad = np.zeros((len(self.transInfo.keys()), self.nsteps))
        # self.baseLoad = np.zeros((self.numBuildings, self.nsteps))
        self.countSinceChange = 0
        self.usagePenalty = np.zeros((self.numBuildings, self.nsteps))
        self.adjustValues = {'flexLoad': np.zeros((self.numBuildings, self.nsteps))}
        self.state = 0
        self.coord_queue = np.zeros((self.numBuildings, self.nsteps))
        self.queued = False

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
        # paramDict['predLoad'] = np.ones_like(self.predictedLoad)
        paramDict['baseLoad'] = np.ones((len(self.transInfo.keys()), self.nsteps))
        # paramDict['baseLoad'] = np.ones((self.numBuildings, self.nsteps))
        self.adjustProb.SolveProblem(paramDict, verbose=verbose)
        self.logger.debug("Adjust optimization initialized")
    
    def TransformerOverload(self):
        '''
        Checks if the community will overload the transformer in the near future
        '''
        predictedTransLoad = np.zeros((len(self.transInfo.keys()), self.nsteps))
        self.overloadList = []
        overload = False
        i = 0
        for key, value in self.transInfo.items():
            # predictedTransLoad[i,:] = self.baseLoad[i,:] + self.transMap[i,:]@self.predictedLoad
            predictedTransLoad[i,:] = self.baseLoad[i,:] + self.transMap[i,:]@self.predictedFlex[:,:,1]
            self.overloadList.append(predictedTransLoad[i,:] - value['rating'])
            # Check if overload occurs at any point
            if np.any((predictedTransLoad[i,:] - value['rating']) >= 0):
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
        # paramDict['flexMin'] = np.zeros((self.numBuildings, self.nsteps))
        # paramDict['flexMax'] = np.ones((self.numBuildings, self.nsteps)) * 100
        paramDict['flexMin'] = self.predictedFlex[:,:,0]
        paramDict['flexMax'] = self.predictedFlex[:,:,1]
        # paramDict['predLoad'] = self.predictedLoad
        # paramDict['baseLoad'] = np.ones((len(self.transInfo.keys()), self.nsteps))
        paramDict['baseLoad'] = self.baseLoad
        self.adjustProb.SolveProblem(paramDict, verbose=verbose)

        adjustValues = {}
        for key, value in self.adjustProb.prob.var_dict.items():
            adjustValues[key] = value.value

        if not(self.adjustProb.feasible):
            self.logger.info("Coordinator optimization infeasible")
            adjustValues = self.ResetValues()
        adjustValues['flexLoad'] = np.round(adjustValues['flexLoad'], 3)
        return adjustValues

    def Dispatch(self, adjustValues):
        '''
        Dispatch phase: Format desired consumption changes as control signals to be provided to each house
        '''
        self.logger.debug("Coordinator dispatch")
        for key, value in adjustValues.items():
            self.adjustValues[key] = value

        # self.usagePenalty[:,:-1] = self.usagePenalty[:,1:]
        # self.usagePenalty[:,-1] = np.sum(self.adjustValues['flexLoad'], axis=1) \
        #                           + (self.usagePenalty[:,-2] * pow(0.99,self.stepSize))
        # self.usagePenalty = np.clip(self.usagePenalty, a_min=0, a_max=100)

    def Step(self):
        '''
        Calls methods that should run every step

        :return: (bool) whether or not the coordinator updated this step
        '''
        self.logger.debug("Coordinator assess check")
        if self.Assess():
            self.state = 1
            adjustValues = self.Adjust()
        else:
            self.state = 0
            adjustValues = self.ResetValues()

        # if self.queued:
        #     self.coord_queue = np.minimum(self.coord_queue, adjustValues['flexLoad'][:,:self.coord_queue.shape[1]])
        #     adjustValues['flexLoad'][:,:self.coord_queue.shape[1]] = self.coord_queue
        #     self.coord_queue = self.coord_queue[:,1:]
        #     if np.all(adjustValues['flexLoad'][:,0] == 20):
        #         self.queued = False
        # else:
        #     self.queued = np.any(adjustValues['flexLoad'][:,0] != 20)

        self.Dispatch(adjustValues)

        return True
    
    def ResetValues(self):
        '''
        Method for resetting the values used by the other components of the coordination system (like flexLoad)
        '''
        adjustValues = {}
        adjustValues['flexLoad'] = np.zeros((self.numBuildings,self.nsteps))
        # adjustValues['totalLoad'] = self.predictedLoad
        adjustValues['totalLoad'] = self.predictedFlex[:,:,1] - adjustValues['flexLoad']
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

    # def DefineProblem(self):
    #     '''
    #     Define the optimization problem used in the adjust phase to choose which buildings will utilize their flexibility
    #     '''
    #     flexLoad = cp.Variable((self.numBuildings, self.n), name='flexLoad')
    #     totalLoad = cp.Variable((self.numBuildings, self.n), name='totalLoad')

    #     usagePenalty = cp.Parameter((self.numBuildings, self.n), name='usagePenalty')
    #     dr_ref = cp.Parameter(self.n, name='pow_ref')
    #     trans_ref = cp.Parameter((len(self.transLimits), self.n), name='trans_ref')
    #     predLoad = cp.Parameter((self.numBuildings, self.n), name='predLoad')
    #     baseLoad = cp.Parameter((len(self.transLimits), self.n), name='baseLoad')

    #     W1 = 2
    #     W2 = 1
    #     W3 = 1

    #     flexMax = cp.Parameter((self.numBuildings, self.n), name='flexMax')
    #     flexMin = cp.Parameter((self.numBuildings, self.n), name='flexMin')

    #     objective = cp.Minimize(W1*cp.sum_squares(usagePenalty.T@flexLoad)
    #                             +W2*cp.norm(flexLoad))
    #     constraints = []
    #     constraints.append(totalLoad <= flexMax)
    #     constraints.append(totalLoad >= flexMin)
    #     constraints.append(cp.sum(totalLoad, axis=0) <= dr_ref)
    #     constraints.append(totalLoad == predLoad + flexLoad)
    #     for i in range(0, len(self.transLimits)):
    #         constraints.append(self.transMap[i,:]@totalLoad + baseLoad[i,:] <= trans_ref[i,:])

    #     self.prob = cp.Problem(objective, constraints)

    def DefineProblem(self):
        '''
        Define the optimization problem used in the adjust phase to choose which buildings will utilize their flexibility
        '''
        flexLoad = cp.Variable((self.numBuildings, self.n), name='flexLoad')
        totalLoad = cp.Variable((self.numBuildings, self.n), name='totalLoad')

        usagePenalty = cp.Parameter((self.numBuildings, self.n), name='usagePenalty')
        dr_ref = cp.Parameter(self.n, name='pow_ref')
        trans_ref = cp.Parameter((len(self.transLimits), self.n), name='trans_ref')
        predLoad = cp.Parameter((len(self.transLimits), self.n), name='predLoad')
        baseLoad = cp.Parameter((len(self.transLimits), self.n), name='baseLoad')

        W1 = 2
        W2 = 1
        W3 = 1

        flexMax = cp.Parameter((self.numBuildings, self.n), name='flexMax')
        flexMin = cp.Parameter((self.numBuildings, self.n), name='flexMin')

        objective = cp.Minimize(W1*cp.sum_squares(usagePenalty.T@flexLoad)
                                +W2*cp.norm(flexLoad))
        constraints = []
        # constraints.append(totalLoad <= flexMax)
        constraints.append(totalLoad >= flexMin)
        constraints.append(cp.sum(totalLoad, axis=0) <= dr_ref)
        constraints.append(totalLoad == flexMax - flexLoad)
        constraints.append(flexLoad >= 0)
        for i in range(0, len(self.transLimits)):
            constraints.append(self.transMap[i,:]@totalLoad + baseLoad[i,:] <= trans_ref[i,:])
            # constraints.append(self.transMap[i,:]@totalLoad + baseLoad[i,:] >= -trans_ref[i,:])

        self.prob = cp.Problem(objective, constraints)