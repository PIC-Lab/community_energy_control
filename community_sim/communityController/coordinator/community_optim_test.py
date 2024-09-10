import numpy as np
import pandas as pd

from community_optim import *

def Main():
    n = 24
    numBuildings = 5

    # flexMax = pd.read_csv()         # (numBuildings,n)
    # flexMin = pd.read_csv()         # (numBuildings,n)
    flexMax = np.ones((numBuildings,n))*1000
    flexMin = np.zeros((numBuildings,n))

    coordinator = Coordinator(numBuildings, n)
    coordinator.DefineAdjustProblem()

    paramValues = {
        'usagePenalty': np.array([1,1,3,0.5,4]),
        'dr_reduce': 100,
        'flexMax': flexMax,
        'flexMin': flexMin
    }

    coordinator.SolveProblem(paramValues, verbose=True)

    flexLoad = coordinator.FindVariableByName('flexLoad')
    print(flexLoad.value)

if __name__ == '__main__':
    Main()