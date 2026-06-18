import modelConstructor
import matplotlib.pyplot as plt
import unittest

def Main():
    batModel = modelConstructor.BatteryModel(0.95, 10, 1, 1)
    

class TestBatteryModel(unittest.TestCase):

    def test_discharge(self):
        pass


if __name__ == '__main__':
    Main()