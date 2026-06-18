import numpy as np
import pandas as pd

from community_optim import *

def Main():
    n = 60
    controlAliasList = ["2", "3", "4", "5", "9", "10", "12", "15", "17", "22", "24", "25", "26", "27", "28"]

    predictedLoad = np.ones((len(controlAliasList), n)) * 50
    predictedFlex = np.zeros((len(controlAliasList), n))

    coordinator = Coordinator(len(controlAliasList), n)
    coordinator.AdjustInit(verbose=True)
    print("Finished first run")

    coordinator.predictedLoad = predictedLoad
    coordinator.predictedFlexibility = predictedFlex
    coordinator.baseLoad = np.ones(coordinator.nsteps) * 10
    overload = coordinator.TransformerOverload()
    print(overload)
    print(coordinator.overloadList)
    adjustValues = coordinator.Adjust(verbose=True)
    coordinator.Dispatch(adjustValues)
    print(adjustValues)
    print(coordinator.reductionFactor)

if __name__ == '__main__':
    Main()