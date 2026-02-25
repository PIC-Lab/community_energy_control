from dotenv import dotenv_values
import json
import os
import numpy as np

from communityController.buildingController.buildingController import BuildingController
from communityController.coordinator.community_optim import Coordinator
from communityController.flexibility_metrics.flexibilityMetrics import FlexibilityMetricPredictor

class CommunityController:
    '''
    '''
    def __init__(self, controlAliasList, runName, testCase='MPC'):
        '''
        '''
        dirName = os.path.dirname(__file__)

        self.controlAliasList = controlAliasList
        self.nsteps = 60

        self.controllerList = []
        self.flexibilityList = []
        self.trajectoryList = {}
        self.coordDebug = {}

        self.predictedLoad = np.zeros((len(controlAliasList), self.nsteps))
        self.predictedFlex = np.zeros((len(controlAliasList), self.nsteps))

        self.ControllerInit(runName, testCase, dirName)
        self.CoordinatorInit(runName)
        self.FlexibilityInit(runName)

    def CoordinatorInit(self, runName):
        '''
        '''
        self.coordinator = Coordinator(len(self.controlAliasList), self.nsteps, 1)
        self.coordinator.AdjustInit()

    def ControllerInit(self, runName, testCase, dirName):
        '''
        '''
        # if self.mode == 'deploy':
        #     self.config = dotenv_values('../.env')
        
        # Get all devices from a building using API
        # Using json file for now
        with open(os.path.join(dirName, '../configs/buildingDeviceList.json')) as fp:
            buildingDevices = json.load(fp)
                
        # Create controller object
        for alias in self.controlAliasList:
            for building in buildingDevices:
                if building['house_id'] == alias:
                    devices = building['devices']
                    break
            self.controllerList.append(BuildingController(alias, devices, runName, testCase=testCase))

    def FlexibilityInit(self, runName):
        '''
        '''
        for alias in self.controlAliasList:
            self.flexibilityList.append(FlexibilityMetricPredictor(alias, runName))

    def Step(self, sensorValues, currentTime):
        '''
        Run every time step
        Returns a list of dicts containing the control events for each house. Matches controls API format except for device IDs
        '''
        # Forecast flexibility

        # Update coordinator
        self.coordinator.predictedLoad = self.predictedLoad
        self.coordinator.predictedFlexibility = self.predictedFlex
        self.coordinator.baseLoad = np.ones((2, self.coordinator.nsteps)) * 10
        self.coordinator.Step()
        coordinateSignals = self.predictedLoad - self.coordinator.adjustValues['flexLoad']

        # Update controllers
        controlEvents = []
        for i, alias in enumerate(self.controlAliasList):
            trajectories = self.controllerList[i].Step(sensorValues[alias], coordinateSignals[i,:], currentTime)
            self.trajectoryList[alias] = trajectories
            self.coordDebug[alias] = {}
            self.coordDebug[alias]['usagePenalty'] = self.coordinator.usagePenalty[i]
            self.coordDebug[alias]['flexLoad'] = self.coordinator.adjustValues['flexLoad'][i,:]
            # self.coordDebug[alias]['trans_ref'] = self.coordinator.adjustValues['trans_ref']
            # self.coordDebug[alias]['predLoad']
            # self.coordDebug[alias]['baseLoad']
            # self.coordDebug[alias]['pow_ref']
            self.predictedLoad[i,:] += trajectories['horizon_u_hvac'].detach().numpy()[0,:,0]
            controlEvents.append(self.controllerList[i].controlEvents)
        
        return controlEvents

        

        