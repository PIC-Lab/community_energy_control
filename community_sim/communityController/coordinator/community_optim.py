import numpy as np
import cvxpy as cp
from communityController.coordinator.convex_problem import ConvexProblem

class Coordinator():
    '''
    Class for the community coordinator
    '''

    def __init__(self, numBuildings, n):
        '''
        Constructor
        '''
        self.numBuildings = numBuildings
        self.n = n

        self.assessProb = AssessOptimization()
        self.adjustProb = AdjustOptimization()    
    
    def TransformerOverload(self):
        '''
        Checks if the community will overload the transformer in the near future
        '''
        pass

    def DemandResponse(self):
        '''
        Decides if the community can/should respond to a demand response signal
        '''
        pass

    def Assess(self):
        '''
        Assess phase: Determines if the community can/should adjust planned consumption
        '''
        pass

    def Adjust(self):
        '''
        Adjust phase: Determine how a desired change in consumption will occur
        '''
        pass

    def Dispatch(self):
        '''
        Dispatch phase: Format desired consumption changes as control signals to be provided to each house
        '''
        pass

class AssessOptimization(ConvexProblem):
    '''
    Optimization problem for the assess phase
    '''

    def __init__(self):
        '''
        Constructor
        '''
        super().__init__()

    def DefineProblem(self):
        '''
        Define the optimization problem used in the adjust phase to choose which buildings will utilize their flexibility
        '''
        flexLoad = cp.Variable((self.numBuildings, self.n), name='flexLoad')

        # flexUsage = cp.Parameter((numBuildings,n), name='flexUsage')
        # usageTime = cp.Parameter((numBuildings,n), name='usageTime')
        # price = cp.Parameter(n, name='price')
        usagePenalty = cp.Parameter(self.numBuildings, name='usagePenalty')
        dr_reduce = cp.Parameter(name='dr_reduce')
        W1 = 2
        W2 = 1

        flexMax = cp.Parameter((self.numBuilding, self.n), name='flexMax')
        flexMin = cp.Parameter((self.numBuilding, self.n), name='flexMin')

        objective = cp.Minimize(W1*usagePenalty@flexLoad@np.ones(self.n)+W2*cp.max(flexLoad, axis=0)@np.ones(self.n))
        constraints = []
        constraints.append(flexLoad <= flexMax)
        constraints.append(flexLoad >= flexMin)
        constraints.append(cp.sum(flexLoad, axis=0) >= dr_reduce)

        self.adjustProb = cp.Problem(objective, constraints)

class AdjustOptimization(ConvexProblem):
    '''
    Optimization problem for the adjust phase
    '''

    def __init__(self):
        '''
        Constructor
        '''
        super().__init__()

    def DefineProblem(self):
        '''
        Define the optimization problem used in the assess phase to decide if responding to a demand response signal is worth it for the community
        '''
        pass