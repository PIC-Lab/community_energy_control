import cvxpy as cp

class ConvexProblem():
    '''
    General class for working with convex optimization problems in cvxpy
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.prob = None

    def GetVariableNames(self):
        '''
        Gets a list of the names of all defined variables
        '''
        return [elem.name for elem in self.prob.variables()]

    def GetParameterNames(self):
        '''
        Gets a list of the names of all defined parameters
        '''
        return [elem.name for elem in self.prob.parameters()]

    def DefineProblem(self):
        '''
        Define the convex optimization problem. Intended to be overloaded by a child class
        '''
        pass

    def SolveProblem(self, paramValues={}, verbose=False):
        '''
        Solves the optimization problem
        Inputs:
            paramValues: (dict{paramName: value}) values for problem parameters, defaults to an empty dict
            verbose: (bool) defaults to false
        '''
        for key, value in paramValues:
            param = self.FindParameterByName(key)
            param.value = value

        result = self.prob.solve(verbose=verbose)
        return result

    def FindVariableByName(self, name):
        '''
        Searches for a variable by name
        Inputs: 
            name: (str) variable name
        Outputs:
            (cvxpy.Variable object) desired variable or None if not found
        '''
        variables = self.prob.variables()
        temp = [elem for elem in variables if elem.name() == name]
        if len(temp) == 0:
            print("No element found with name %s" % name)
            return
        return temp[0]
    
    def FindParameterByName(self, name):
        '''
        Searches for a parameter by name
        Inputs: 
            name: (str) parameter name
        Outputs:
            (cvxpy.Parameter object) desired variable or None if not found
        '''
        parameters = self.prob.parameters()
        temp = [elem for elem in parameters if elem.name() == name]
        if len(temp) == 0:
            print("No element found with name %s" % name)
            return
        return temp[0]