import pandas as pd
import numpy as np
import os

from communityController.buildingController.thermal_node_model.modelConstructor import BuildingNode, ControllerSystem, ModeClassifier
from communityController.buildingController.thermal_node_model.runManager import RunManager
from communityController.buildingController.thermal_node_model.neuro_util import Normalizer

import torch

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
    def __init__(self, id, devices):
        """
        Constructor
        Parameters:
            id: (str) id of the controller's building
            testCase: (str) name of test case being run, defaults to MPC
        """
        self.actuatorValues = {'heatingSetpoint': 0, 'coolingSetpoint': 0, 'battery': 3.8}
        self.sensorValues = {'indoorAirTemp': 0}
        self.buildingID = id
        self.devices = devices

        self.controlEvents = []

        self.dirName = os.path.dirname(__file__)

        # HVAC Values
        self.HVAC_mode = 0

        # MPC variables
        self.cl_system = None

        # Load building specific parameters
        self.setpointInfo = {"heatSP": 18.88888888888889, "coolSP": 24.444444444444443, "deadband": 4.444444444444445}      # Default value, should be commented out
        # Temporary, needs to be fixed later
        self.setpointSchedule = pd.read_csv(os.path.join(self.dirName, '../../setpointSchedule.csv')
                                            , header=None).to_numpy()[np.newaxis, :]
        self.weather = pd.read_csv(os.path.join(self.dirName, '../../results/summer/1_out.csv')
                                   , usecols=['Time', 'Site Outdoor Air Temperature'])      # Time column needs to be a timestamp
        self.weather.set_index(pd.to_datetime(self.weather['Time'], format="%Y-%m-%d %H:%M:%S"), inplace=True)
        self.weather.drop('Time', axis=1, inplace=True)

        # Run additional setup functions
        self.LoadMPC()
        self.count = 0
        self.HVAC_lock = 0
        self.HVAC_prevMode = 0

    # Placeholder database functions (for deployment)
    def PullSensorValues(self, sensorValues, coordinateSignals, currentTime):
        """
        Pull sensor data from database
        """
        # Update sensor values
        for key, value in sensorValues.items():
            self.sensorValues[key] = value

        # Prepare data for current horizon
        currentMinutes = currentTime.hour * 60 + currentTime.minute
        y = self.sensorValues['indoorAirTemp']
        self.stateData[0,0,self.y_idx] = y
        ymin = self.setpointSchedule[:,currentMinutes:currentMinutes+self.nsteps,:] - self.setpointInfo['deadband']
        ymax = self.setpointSchedule[:,currentMinutes:currentMinutes+self.nsteps,:] + self.setpointInfo['deadband']
        pos = self.weather.index.get_loc(currentTime)
        d = self.weather['Site Outdoor Air Temperature'].iloc[pos:pos+self.nsteps].to_numpy()[np.newaxis,:,np.newaxis]
        dr = coordinateSignals[np.newaxis,:,np.newaxis]
        self.horizonData = {'xn': torch.tensor(self.stateData, dtype=torch.float32),
                            'y': torch.tensor(np.array([y])[np.newaxis,:,np.newaxis], dtype=torch.float32),
                            'ymin': torch.tensor(ymin, dtype=torch.float32),
                            'ymax': torch.tensor(ymax, dtype=torch.float32),
                            'd': torch.tensor(d, dtype=torch.float32),
                            'dr': torch.tensor(dr, dtype=torch.float32)}

    def PushControlSignals(self):
        """
        Push control signals to actuators
        """
        self.controlEvents = {}
        self.controlEvents['location'] = self.buildingID
        self.controlEvents['devices'] = {'coolingSetpoint':  self.actuatorValues['coolingSetpoint'],
                                         'heatingSetpoint': self.actuatorValues['heatingSetpoint'],
                                         'battery': self.actuatorValues['battery']}

    def Step(self, sensorValues, coordinateSignals, currentTime):
        """
        Main function to be run every time step, runs all control functions
        """

        # Pull data from sensors
        self.PullSensorValues(sensorValues, coordinateSignals, currentTime)

        # Control functions
        trajectories = self.PredictiveControl()

        # Push control signals to actuators
        self.PushControlSignals()
        return trajectories

    # Control functions
    def PredictiveControl(self):
        '''
        Run predictive control for the prediction horizon
        Inputs:
            currentStep: (int) current step of simulation loop (simulation only)
        '''

        # Run control
        trajectories = self.cl_system(self.horizonData)

        control = trajectories['u'][0,0,0].detach()

        if self.HVAC_lock:
            self.count += 1
            if self.count > 3:
                self.count = 0
                self.HVAC_lock = 0
        else:
            if control >= 0.5:
                self.HVAC_mode = 1
                if self.HVAC_prevMode == 0:
                    self.HVAC_lock = 1
            else:
                self.HVAC_mode = 0
                if self.HVAC_prevMode == 1:
                    self.HVAC_lock = 1
        self.HVAC_prevMode = self.HVAC_mode

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
        return trajectories

    # Control setup functions
    def LoadMPC(self):
        '''
        Creates a controller object from the modelConstructor module and loads the saved weights
        '''

        device = torch.device("cpu")

        run = 'latestRun'
        run = 'alf_AllBuildings_1stPass'

        # Path relative to the directory the sim is being run in. Needs to be fixed
        filePath = os.path.join(self.dirName, 'thermal_node_model')

        manager = RunManager(run, saveDir=f'{filePath}/savedRuns')
        manager.LoadRunJson(run)
        norm = Normalizer()
        norm.load(f"{manager.runPath}norm/{self.buildingID}/")

        for key in manager.models.keys():
            if key.find('buildingThermal') != -1:
                thermalModelName = key
            elif key.find('classifier') != -1:
                classifierModelName = key
            elif key.find('controller') != -1:
                controllerModelName = key
            else:
                raise ValueError(f"Model name '{key}' does not meet expected naming conventions.")
            
        self.nsteps = manager.models[controllerModelName]['train_params']['nsteps']

        # Thermal model definition
        # Building thermal model
        initParams = manager.models[thermalModelName]['init_params']
        buildingThermal = BuildingNode(nx=initParams['nx'],
                                       nu=initParams['nu'],
                                       nd=initParams['nd'],
                                       manager=manager,
                                       name=thermalModelName,
                                       device=device,
                                       debugLevel = 0,
                                       saveDir=f"{manager.runPath+thermalModelName}/{self.buildingID}")
        buildingThermal.CreateModel()

        buildingThermal.TrainModel(dataset=None, load=True, test=False)

        self.stateData = np.zeros((1,1,initParams['nx']))

        # Classifier model definition
        initParams = manager.models[classifierModelName]['init_params']
        classifier = ModeClassifier(nm=initParams['nm'],
                                    nu=initParams['nu'],
                                    manager=manager,
                                    name=classifierModelName,
                                    device=device,
                                    debugLevel = 0,
                                    saveDir=f"{manager.runPath+classifierModelName}")
        classifier.CreateModel()

        classifier.TrainModel(dataset=None, load=True, test=False)

        # Controller model definition
        initParams = manager.models[controllerModelName]['init_params']
        controlSystem = ControllerSystem(nx=initParams['nx'],
                                         nu=initParams['nu'],
                                         nd=initParams['nd'],
                                         nd_obs=initParams['nd_obs'],
                                         ny=initParams['ny'],
                                         y_idx=initParams['y_idx'],
                                         d_idx=initParams['d_idx'],
                                         manager=manager,
                                         name=controllerModelName,
                                         norm=norm,
                                         thermalModel=buildingThermal.model,
                                         classifier=classifier.model,
                                         device=device,
                                         debugLevel=0,
                                         saveDir=f"{manager.runPath+controllerModelName}/{self.buildingID}")
        controlSystem.CreateModel()

        controlSystem.TrainModel(dataset=None, tempMin=None, tempMax=None, load=True, test=False)

        self.y_idx = initParams['y_idx']
        self.cl_system = controlSystem.system