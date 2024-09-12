import json
import pandas as pd
import numpy as np
import torch

from buildingController.thermal_node_model.modelConstructor import BuildingNode, ControllerSystem, Normalizer, ModeClassifier
from buildingController.thermal_node_model.runManager import RunManager

class BuildingController:
    """
    Class for a building controller
    Attributes:
        actuatorValues: (dict[float]) control decisions for each control point
        sensorValues: (dict[float]) current sensor values for each sensor
        buildingID: (str) id of the controller's building
        HVAC_mode: (int) operation mode of HVAC
        cl_system: (System) MPC model
        norm: (Normalizer) normalizer class used to norm/denorm values going in/out of the controller
        pv: (ndarray[float]) predicted PV generation
        dist: (ndarray[float]) predicted disturbances (outdoor temperature)
    """
    def __init__(self, id, devices, mode):
        """
        Constructor
        Parameters:
            id: (str) id of the controller's building
            testCase: (str) name of test case being run, defaults to MPC
        """
        self.actuatorValues = {'heatingSetpoint': 0, 'coolingSetpoint': 0}
        self.sensorValues = {'indoorAirTemp': 0}
        self.buildingID = id

        # HVAC Values
        self.HVAC_mode = 0

        # Load building specific parameters
        self.setpointInfo = {"heatSP": 18.88888888888889, "coolSP": 24.444444444444443, "deadband": 4.444444444444445}      # Default value, should be commented out

        self.tempBounds = np.ones((1,24*60+60, 2))       # Hard coded to be one day long, will need to change
        self.tempBounds[:,:,0] *= self.setpointInfo['heatSP'] - self.setpointInfo['deadband']     # min
        self.tempBounds[:,:,1] *= self.setpointInfo['coolSP'] + self.setpointInfo['deadband']     # max

    # Placeholder database functions (for deployment)
    def PushControlSignals(self):
        """
        Push control signals to actuators
        """
        pass

    def PullSensorValues(self):
        """
        Pull sensor data from database
        """
        pass

    def UpdateControl(self):
        """
        Main function to be run every time step, runs all control functions
        """

        # Pull data from sensors
        self.PullSensorValues()

        # Control functions
        self.HVAC_Control()

        # Push control signals to actuators
        self.PushControlSignals()

    # Control functions
    def HVAC_Control(self, forceMode=0, op_mode=4):
        """
        Deadband HVAC control for baseline case
        Inputs:
            forceMode: (int) Used to override control and disable heating (1) or cooling (-1), defaults to 0
            op_mode: (int) Used to disable control entirely (0), defaults to 4
        """
        temperature = self.sensorValues['indoorAirTemp']
        modeLock = 0
        match self.HVAC_mode:
            case 4:
                if temperature <= self.setpointInfo['heatSP'] + self.setpointInfo['deadband']:
                    modeLock = 1
            case 3:
                if temperature <= self.setpointInfo['heatSP'] + self.setpointInfo['deadband']:
                    if temperature >= self.setpointInfo['heatSP'] - self.setpointInfo['deadband']:
                        modeLock = 1
            case 2:
                if temperature >= self.setpointInfo['coolSP'] - self.setpointInfo['deadband']:
                    modeLock = 1
            case 1:
                if temperature >= self.setpointInfo['coolSP'] - self.setpointInfo['deadband']:
                    if temperature <= self.setpointInfo['coolSP'] + self.setpointInfo['deadband']:
                        modeLock = 1

        if modeLock == 0:
            if temperature <= self.setpointInfo['heatSP'] - self.setpointInfo['deadband']:
                self.HVAC_mode = 4
            elif temperature <= self.setpointInfo['heatSP']:
                self.HVAC_mode = 3
            elif temperature >= self.setpointInfo['coolSP'] + self.setpointInfo['deadband']:
                self.HVAC_mode = 2
            elif temperature >= self.setpointInfo['coolSP']:
                self.HVAC_mode = 1
            else:
                self.HVAC_mode = 0

        # Heat only mode
        if forceMode == 1:
            if (self.HVAC_mode == 1) or (self.HVAC_mode == 2):
                self.HVAC_mode = 0
        # Cool only mode
        elif forceMode == -1:
            if (self.HVAC_mode == 3) or (self.HVAC_mode == 4):
                self.HVAC_mode = 0
        # Disable control if not available
        if op_mode == 0:
            self.HVAC_mode = 0

        match self.HVAC_mode:
            case 0:
                self.actuatorValues['heatingSetpoint'] = 0
                self.actuatorValues['coolingSetpoint'] = 100
            case 1:
                self.actuatorValues['heatingSetpoint'] = 0
                self.actuatorValues['coolingSetpoint'] = 10
            case 2:
                self.actuatorValues['heatingSetpoint'] = 0
                self.actuatorValues['coolingSetpoint'] = 10
            case 3:
                self.actuatorValues['heatingSetpoint'] = 90
                self.actuatorValues['coolingSetpoint'] = 100
            case 4:
                self.actuatorValues['heatingSetpoint'] = 90
                self.actuatorValues['coolingSetpoint'] = 100