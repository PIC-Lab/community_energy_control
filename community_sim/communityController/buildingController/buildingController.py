import pandas as pd
import numpy as np
import os
import datetime as dt
import cvxpy as cp
import json

from communityController.buildingController.thermal_node_model.modelConstructor import BuildingNode, ControllerSystem, Normalizer
from communityController.buildingController.thermal_node_model.runManager import RunManager

import torch

class BuildingController:
    """
    Class for a building controller
    Attributes:
        actuatorValues: (dict[float]) control decisions for each control point
        sensorValues: (dict[float]) current sensor values for each sensor
        buildingID: (str) id of the controller's building
        HVAC_mode: (int) operation mode of HVAC
        cl_system: (System) DPC model
        norm: (Normalizer) normalizer class used to norm/denorm values going in/out of the controller
        pv: (ndarray[float]) predicted PV generation
        dist: (ndarray[float]) predicted disturbances (outdoor temperature)
    """
    def __init__(self, id, devices, runName, logger, train=False, testCase='DPC', nstepsOverride=None):
        """
        Constructor
        Parameters:
            id: (str) id of the controller's building
            testCase: (str) name of test case being run, defaults to DPC
        """
        dirName = os.path.dirname(__file__)

        self.actuatorValues = {'heatingSetpoint': 18, 'coolingSetpoint': 24, 'battery': 0}
        self.sensorValues = {'indoorAirTemp': 21, 'batterySOC': 8.2}
        self.buildingID = id
        self.devices = devices
        self.runName = runName
        self.logger = logger
        self.testCase = testCase
        self.nstepsOverride = nstepsOverride
        self.step_mins = 1

        self.controlEvents = {}

        # HVAC Values
        self.HVAC_mode = 0

        # DPC variables
        self.cl_system = None

        # Load building specific parameters
        self.setpointInfo = {"heatSP": 18.88888888888889, "coolSP": 24.444444444444443, "deadband": 0.5*3}      # Default value, should be commented out
        # Temporary, needs to be fixed later
        self.setpointSchedule = pd.read_csv(os.path.join(dirName, '../../setpointSchedule.csv')
                                            , header=None).to_numpy()[np.newaxis, :]
        self.weather = pd.read_csv(os.path.join(dirName, '../../results/2024_weather.csv'), usecols=['Time', 'Site Outdoor Air Temperature'])      # Time column needs to be a timestamp
        self.weather.set_index(pd.to_datetime(self.weather['Time'], format="%Y-%m-%d %H:%M:%S"), inplace=True)
        self.weather.drop('Time', axis=1, inplace=True)
        self.socSchedule = pd.read_csv(os.path.join(dirName, 'thermal_node_model/socSchedule.csv')).to_numpy() * 16.4

        # Run additional setup functions
        if testCase == 'DPC':
            if not(train):
                self.LoadDPC(dirName)
        elif testCase == 'MPC':
            self.CreateMPC()
            self.feasible = True
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
        if self.testCase == 'DPC':
            currentMinutes = currentTime.hour * 60 + currentTime.minute
            y = self.sensorValues['indoorAirTemp']
            self.stateData[0,0,self.y_idx] = y
            ymin = self.setpointSchedule[:,currentMinutes:currentMinutes+self.nsteps,:] - self.setpointInfo['deadband']
            ymax = self.setpointSchedule[:,currentMinutes:currentMinutes+self.nsteps,:] + self.setpointInfo['deadband']
            pos = self.weather.index.get_loc(currentTime)
            d = self.weather['Site Outdoor Air Temperature'].iloc[pos:pos+self.nsteps].to_numpy()[np.newaxis,:,np.newaxis]
            dr = coordinateSignals[np.newaxis,:,np.newaxis]
            self.horizonData = {'yn': torch.tensor(self.stateData, dtype=torch.float32),
                                'y': torch.tensor(np.array([y])[np.newaxis,:,np.newaxis], dtype=torch.float32),
                                'ymin': torch.tensor(ymin, dtype=torch.float32),
                                'ymax': torch.tensor(ymax, dtype=torch.float32),
                                'd': torch.tensor(d, dtype=torch.float32),
                                'powerRef': torch.tensor(dr, dtype=torch.float32),
                                'cost': torch.tensor(BuildingController.TOUPricing(currentTime, self.nsteps)[np.newaxis,:,np.newaxis], dtype=torch.float32),
                                'stored': torch.tensor(np.array([self.sensorValues['batterySOC']])[np.newaxis,:,np.newaxis], dtype=torch.float32),
                                'batRef': torch.tensor(self.socSchedule.take(range(currentMinutes, currentMinutes+self.nsteps), axis=0, mode='wrap')[np.newaxis,:], dtype=torch.float32),
                                'batMax': torch.tensor(np.ones((1,self.nsteps+1,1))*8.0, dtype=torch.float32),
                                'name': 'horizon'}
            
            normDict = {'yn': 'y', 'y': 'y', 'ymin': 'y', 'ymax': 'y', 'd': 'd', 'cost': 'p'}

            self.horizonData = self.norm.norm(self.horizonData, keys=normDict)

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
        # trajectories = self.PredictiveControl()
        trajectories = self.SimpleMPC(currentTime)

        # Push control signals to actuators
        if not(self.testCase == 'base'):
            self.PushControlSignals()
        return trajectories

    # Control functions
    def RandomControl(self):
        if self.HVAC_lock:
            self.count += 1
            if self.count > 30:
                self.count = 0
                self.HVAC_lock = 0
        else:
                self.actuatorValues['heatingSetpoint'] = 0
                self.actuatorValues['coolingSetpoint'] = np.random.randint(18, 23)
                self.HVAC_lock = 1

    def PredictiveControl(self):
        '''
        Run predictive control for the prediction horizon
        Inputs:
            currentStep: (int) current step of simulation loop (simulation only)
        '''

        # Run control
        trajectories = self.cl_system(self.horizonData)

        self.actuatorValues['battery'] = trajectories['horizon_u_bat'][0,0,0].detach().item()

        control = trajectories['horizon_u_hvac'][0,0,0].detach()

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
                self.actuatorValues['coolingSetpoint'] = 16
            case 2:
                self.actuatorValues['heatingSetpoint'] = 0
                self.actuatorValues['coolingSetpoint'] = 16
            case 3:
                self.actuatorValues['heatingSetpoint'] = 30
                self.actuatorValues['coolingSetpoint'] = 100
            case 4:
                self.actuatorValues['heatingSetpoint'] = 30
                self.actuatorValues['coolingSetpoint'] = 100

        normDict = {'horizon_y': 'y', 'horizon_ymin': 'y', 'horizon_ymax': 'y', 'horizon_d': 'd', 'horizon_cost': 'p'}
        
        return self.norm.denorm(trajectories, keys=normDict)
    
    def SimpleMPC(self, currentTime):
        currentMinutes = currentTime.hour * 60 + currentTime.minute
        
        pos = self.weather.index.get_loc(currentTime)
        d = self.weather['Site Outdoor Air Temperature'].iloc[pos:pos+self.nsteps].to_numpy()[:,np.newaxis]

        self.horizonData = {
            'y0': np.array([self.sensorValues['indoorAirTemp']]),
            'y_min': self.setpointSchedule[0,currentMinutes:currentMinutes+self.nsteps,:] - self.setpointInfo['deadband'],
            'y_max': self.setpointSchedule[0,currentMinutes:currentMinutes+self.nsteps,:] + self.setpointInfo['deadband'],
            'd': d,
            'stored0': np.array([self.sensorValues['batterySOC']]),
            'bat_min': np.ones((self.nsteps,1))*1.5,
            'bat_max': np.ones((self.nsteps,1))*15,
            'cost': BuildingController.TOUPricing(currentTime, self.nsteps)[:,np.newaxis],
            'power_ref': np.ones((self.nsteps,1))*50
        }
        if (currentMinutes >= 740) and (currentMinutes < 800):
            self.horizonData['power_ref'] = np.ones((self.nsteps,1))*5

        for key, param in self.prob.param_dict.items():
            param.value = self.horizonData[key]

        self.prob.solve(solver='SCIP')
        control_traj = self.prob.var_dict

        if control_traj['u_hvac'].value is None:
            self.logger.warn(f"MPC optization failed for {self.buildingID}")
            if self.sensorValues['indoorAirTemp'] < self.setpointSchedule[0,currentMinutes,0] - self.setpointInfo['deadband']:
                control = np.ones(1)
            elif self.sensorValues['indoorAirTemp'] > self.setpointSchedule[0,currentMinutes,0] + self.setpointInfo['deadband']:
                control = np.ones(1) * -1
            else:
                control = np.zeros(1)
            self.feasible = False
        else:
            control = control_traj['u_hvac'].value[0,0]
            self.feasible = True

        if self.HVAC_lock:
            self.count += self.step_mins
            if self.count > 3:
                self.count = 0
                self.HVAC_lock = 0
        else:
            if control >= 0.5:
                self.HVAC_mode = 3
                # if self.HVAC_prevMode == 0:
                #     self.HVAC_lock = 1
            elif control <= -0.5:
                self.HVAC_mode = 1
                # if self.HVAC_prevMode == 0:
                #     self.HVAC_lock = 1
            else:
                self.HVAC_mode = 0
                # if self.HVAC_prevMode == 1:
                #     self.HVAC_lock = 1
        self.HVAC_prevMode = self.HVAC_mode

        match self.HVAC_mode:
            case 0:
                self.actuatorValues['heatingSetpoint'] = 0
                self.actuatorValues['coolingSetpoint'] = 100
            case 1:
                self.actuatorValues['heatingSetpoint'] = 0
                self.actuatorValues['coolingSetpoint'] = self.setpointSchedule[0,currentMinutes,0] - self.setpointInfo['deadband']
            case 2:
                self.actuatorValues['heatingSetpoint'] = 0
                self.actuatorValues['coolingSetpoint'] = self.setpointSchedule[0,currentMinutes,0] - self.setpointInfo['deadband']
            case 3:
                self.actuatorValues['heatingSetpoint'] = self.setpointSchedule[0,currentMinutes,0] + self.setpointInfo['deadband']
                self.actuatorValues['coolingSetpoint'] = 100
            case 4:
                self.actuatorValues['heatingSetpoint'] = self.setpointSchedule[0,currentMinutes,0] + self.setpointInfo['deadband']
                self.actuatorValues['coolingSetpoint'] = 100

        trajectories = {}
        for key, value in self.prob.param_dict.items():
            trajectories[key] = value.value

        if control_traj['u_hvac'].value is None:
            trajectories['y'] = self.horizonData['y0'][np.newaxis,:]
            trajectories['u_hvac'] = control[np.newaxis,:]
            trajectories['stored'] = self.horizonData['stored0'][np.newaxis,:]
            trajectories['u_bat'] = np.zeros((self.nsteps,1))
            trajectories['u_tot'] = trajectories['u_bat'] + np.abs(trajectories['u_hvac']) * 6
        else:
            for key, value in self.prob.var_dict.items():
                trajectories[key] = value.value

        return trajectories

    # Control setup functions
    def LoadDPC(self, dirName):
        '''
        Creates a controller object from the modelConstructor module and loads the saved weights
        '''

        device = torch.device("cpu")

        # Path relative to the directory the sim is being run in. Needs to be fixed
        filePath = os.path.join(dirName, 'thermal_node_model')

        manager = RunManager(self.runName, saveDir=f'{filePath}/deployModels')
        manager.LoadRunJson(self.runName)
        self.norm = Normalizer()
        self.norm.load(f"{manager.runPath}norm/{self.buildingID}/")

        for key in manager.models.keys():
            if key.find('buildingThermal') != -1:
                thermalModelName = key
            elif key.find('controller') != -1:
                controllerModelName = key
            else:
                self.logger.warn(f"Model name '{key}' does not meet expected naming conventions. Key will be ignored.")
            
        self.nsteps = manager.models[controllerModelName]['train_params']['nsteps']

        # Thermal model definition
        # Building thermal model
        initParams = manager.models[thermalModelName]['init_params']
        buildingThermal = BuildingNode(nx=initParams['nx'],
                                       nu=initParams['nu'],
                                       nd=initParams['nd'],
                                       manager=manager,
                                       norm=self.norm,
                                       name=thermalModelName,
                                       device=device,
                                       debugLevel = 0,
                                       saveDir=f"{manager.runPath+thermalModelName}/{self.buildingID}")
        buildingThermal.CreateModel()

        buildingThermal.TrainModel(dataset=None, load=True, test=False)

        self.stateData = np.zeros((1,1,initParams['nx']))

        # Controller model definition
        initParams = manager.models[controllerModelName]['init_params']
        controlSystem = ControllerSystem(nx=initParams['nx'],
                                         nu=initParams['nu'],
                                         nd=initParams['nd'],
                                         nd_obs=initParams['nd_obs'],
                                         ni=2,
                                         ny=initParams['ny'],
                                         y_idx=initParams['y_idx'],
                                         d_idx=initParams['d_idx'],
                                         manager=manager,
                                         name=controllerModelName,
                                         norm=self.norm,
                                         thermalModel=buildingThermal.problem,
                                         device=device,
                                         debugLevel=0,
                                         saveDir=f"{manager.runPath+controllerModelName}/{self.buildingID}")
        controlSystem.CreateModel()

        controlSystem.TrainModel(dataset=None, tempMin=None, tempMax=None, load=True, test=False)

        self.y_idx = initParams['y_idx']
        self.cl_system = controlSystem.problem

    def CreateMPC(self):
        RC = 400
        alpha = 0.08

        # with open('../thermal_rc_model/buildings_tuned.json') as fp:
        #     buildingRC = json.load(fp)

        # RC = buildingRC[self.buildingID]['RC']
        # alpha = buildingRC[self.buildingID]['alpha']

        manager = RunManager(self.runName, saveDir='deployModels')
        manager.LoadRunJson(self.runName)

        for key in manager.models.keys():
            if key.find('buildingNODE') != -1:
                thermalModelName = key
            elif key.find('controller') != -1:
                controllerModelName = key
            else:
                self.logger.warn(f"Building {self.buildingID}: Model name '{key}' does not meet expected naming conventions. Key will be ignored.")
            
        if self.nstepsOverride is None:
            self.nsteps = manager.models[controllerModelName]['train_params']['nsteps']
        else:
            self.nsteps = self.nstepsOverride

        self.nsteps_eff = int(self.nsteps / self.step_mins)

        # outTemp = pd.read_csv('./results/basecase_week/1_out.csv', usecols=['Site Outdoor Air Temperature'])
        # waterHeater = pd.read_csv(f'./results/basecase_week/{self.buildingID}_out.csv', usecols=['WaterSystems:Electricity'])
        # self.dist = np.column_stack([outTemp.to_numpy(), waterHeater.to_numpy()])
        # self.dist = np.tile(self.dist, (2,1))
        # self.dist = self.dist[np.newaxis,:]

        self.norm = Normalizer()         # type:ignore
        self.norm.load(manager.runPath+f'norm/{self.buildingID}/')

        u_hvac = cp.Variable((self.nsteps_eff,1), integer=True, name='u_hvac')
        u_bat = cp.Variable((self.nsteps_eff,1), name='u_bat')
        u_tot = cp.Variable((self.nsteps_eff,1), name='u_tot')
        y = cp.Variable((self.nsteps_eff,1), name='y')
        stored = cp.Variable((self.nsteps_eff,1), name='stored')

        y0 = cp.Parameter((1), name='y0')
        ymin = cp.Parameter((self.nsteps_eff,1), name='y_min')
        ymax = cp.Parameter((self.nsteps_eff,1), name='y_max')
        d = cp.Parameter((self.nsteps_eff,1), name='d')
        stored0 = cp.Parameter((1), name='stored0')
        batmin = cp.Parameter((self.nsteps_eff,1), name='bat_min')
        batmax = cp.Parameter((self.nsteps_eff,1), name='bat_max')
        cost = cp.Parameter((self.nsteps_eff,1), name='cost')
        power_ref = cp.Parameter((self.nsteps_eff,1), name='power_ref')

        objective = cp.Minimize(1.0*cost.T@u_tot)
        # objective = cp.Minimize(1.0*cost.T@u_hvac)

        constraints = [u_hvac >= -1, u_hvac <=0,
                       y[1:] >= ymin[1:], y[1:] <= ymax[1:],
                    #    y <= 30,
                       y[0] == y0,
                       u_tot == u_hvac * -6 + u_bat,
                       u_bat <= 9.6, u_bat >= u_hvac * 6,
                       stored[0] == stored0,
                       stored >= batmin, stored <= batmax,
                       u_tot <= power_ref
                    ]
        for i in range(1, self.nsteps_eff):
            constraints.append(y[i] == (-1/(RC)*y[i-1] + 1/(RC)*d[i-1] + alpha*u_hvac[i-1]*6)*self.step_mins + y[i-1])
            constraints.append(stored[i] == stored[i-1] + u_bat[i] * self.step_mins / 60)

        self.prob = cp.Problem(objective, constraints)

    @staticmethod
    def TOUPricing(date, nsteps):
        """
        Gets the time of use pricing depending on the date
        Parameters:
            date: datetime of the desired day
            timeSteps: int of the number of timesteps in a single day (96 for 15 min intervals)
        Returns:
            list of ints of the hourly cost of energy in cents/kWh
        """
        # Prices in cents/kWh
        # Summer: May 1st to September 30th
        summer = [0.07884, 0.21277]
        summer = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 19, 19, 28, 28, 28, 28, 10, 10, 10, 10, 10]
        
        # Winter: October 1st to April 30th
        winter = [0.06792, 0.18331]
        winter = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 14, 14, 17, 17, 17, 17, 10, 10, 10, 10, 10]
        
        limit1 = dt.datetime(date.year, 5, 1)
        limit2 = dt.datetime(date.year, 10, 1)
        
        if (date >= limit1) & (date < limit2):
            price = summer
        else:
            price = winter

        tempList = []
        for i in range(0,nsteps):
            if date.minute + i >= 60:
                if date.hour >= 23:
                    tempList.append(price[0])
                else:
                    tempList.append(price[date.hour+1])
            else:
                tempList.append(price[date.hour])
        return np.array(tempList)