import sys
from dotenv import dotenv_values
import json

from communityController.buildingController.buildingController import BuildingController
from communityController.coordinator.community_optim import Coordinator
from communityController.flexibility_metrics.flexibilityMetrics import FlexibilityMetricPredictor

class CommunityController:
    '''
    '''
    def __init__(self, controlAliasList):
        '''
        '''
        self.controlAliasList = controlAliasList

        self.controllerList = []
        self.flexibilityList = []

    def CoordinatorInit(self):
        '''
        '''
        self.coodinator = Coordinator()

    def ControllerInit(self):
        '''
        '''
        if self.mode == 'deploy':
            self.config = dotenv_values('../.env')
        
        # Get all devices from a building using API
        # Using json file for now
        with open('community_sim/buildingDeviceList.json') as fp:
            buildingDevices = json.load(fp)
                
        # Create controller object
        for alias in self.controlAliasList:
            for building in buildingDevices:
                if building['house_id'] == alias:
                    devices = building['devices']
                    break
            self.controllerList.append(BuildingController(alias, devices, self.mode, 'MPC'))

    def FlexibilityInit(self):
        '''
        '''
        for alias in self.controlAliasList:
            self.flexibilityList.append(FlexibilityMetricPredictor())

    def Step(self, values, currentTime):
        '''
        Run every time step
        Returns a list of dicts containing the control events for each house. Matches controls API format except for device IDs
        '''
        # Update controllers
        controlEvents = {}
        for i, alias in enumerate(self.controlAliasList):
            self.controllerList[i].Step(values, currentTime)

            controlEvents[alias] = self.controllerList[i].actuatorValues
        
        return controlEvents

        

        