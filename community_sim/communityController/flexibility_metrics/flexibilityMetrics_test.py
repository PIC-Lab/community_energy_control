from flexibilityMetrics import *

def Main():
    flexPred = FlexibilityMetricPredictor()

    flexPred.TrainPredictor()
    # flexPred.LoadPredictor()

    flexPred.PlotPredictor()


if __name__ == '__main__':
    Main()