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
    def __init__(self, id, devices, runName, logger, weather, sp_cool, sp_heat, baseLoad, train=False, testCase='DPC', nsteps=60, stepSize=1):
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
        self.nstepsOverride = nsteps
        self.step_mins = stepSize

        self.controlEvents = {}

        # HVAC Values
        self.HVAC_mode = 0
        self.hvac_offset = 3
        self.hvac_queue = np.zeros((self.hvac_offset,1))

        # Battery Values
        self.batSize = 16.4
        self.inverterSize = 9.6
        self.bat_hvac_queue = np.zeros((self.hvac_offset,1))

        # DPC variables
        self.cl_system = None

        # Load building specific parameters
        self.setpointInfo = {"deadband": 0.5}      # Default value
        # Temporary, needs to be fixed later, only used in DPC
        self.setpointSchedule = pd.read_csv(os.path.join(dirName, '../../sim_schedules/setpointSchedule.csv'),
                                            header=None).to_numpy()[np.newaxis, :]
        self.sp_cool_sched = sp_cool[:,np.newaxis]
        self.sp_heat_sched = sp_heat[:,np.newaxis]

        self.weather = weather
        # Temporary, needs to be fixed later, only used in DPC
        self.socSchedule = pd.read_csv(os.path.join(dirName, 'thermal_node_model/socSchedule.csv')).to_numpy() * 16.4

        self.chargeSchedule = np.zeros((24*int(60/self.step_mins),1))
        self.chargeSchedule[600:1020] = np.ones((420,1))*-4
        self.chargeSchedule[1020:1260] = np.ones((240,1))
        self.chargeSchedule = np.concatenate([self.chargeSchedule, self.chargeSchedule], axis=0)

        # Should be shifted up to the community controller
        # self.baseLoad = pd.read_csv(os.path.join(dirName, '../../sim_schedules/base_load_ws.csv'), usecols=[id]).values
        self.baseLoad = baseLoad[:,np.newaxis]

        # Run additional setup functions
        if self.testCase == 'DPC':
            if not(train):
                self.LoadDPC(dirName)
        elif self.testCase == 'MPC':
            self.CreateMPC()
            self.feasible = True
        elif self.testCase == 'MPC_alt':
            self.CreateMPC_alt()
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

        currentMinutes = int((currentTime.hour * 60 + currentTime.minute) / self.step_mins)

        pos = self.weather.index.get_loc(currentTime)
        d = self.weather['Site Outdoor Air Temperature'].iloc[pos:pos+self.nsteps_eff].to_numpy()[:,np.newaxis]
        if np.any(d <= 10):
            self.HVAC_mode = 3
        elif np.any(d >= 27):
            self.HVAC_mode = 1
        else:
            # self.HVAC_mode = 0
            self.HVAC_mode = 1

        # Prepare data for current horizon
        if self.testCase == 'DPC':
            y = self.sensorValues['indoorAirTemp']
            self.stateData[0,0,self.y_idx] = y
            ymin = self.setpointSchedule[:,currentMinutes:currentMinutes+self.nsteps_eff,:] - self.setpointInfo['deadband']
            ymax = self.setpointSchedule[:,currentMinutes:currentMinutes+self.nsteps_eff,:] + self.setpointInfo['deadband']
            # pos = self.weather.index.get_loc(currentTime)
            # d = self.weather['Site Outdoor Air Temperature'].iloc[pos:pos+self.nsteps_eff].to_numpy()[np.newaxis,:,np.newaxis]
            dr = coordinateSignals[np.newaxis,:,np.newaxis]
            self.horizonData = {'yn': torch.tensor(self.stateData, dtype=torch.float32),
                                'y': torch.tensor(np.array([y])[np.newaxis,:,np.newaxis], dtype=torch.float32),
                                'ymin': torch.tensor(ymin, dtype=torch.float32),
                                'ymax': torch.tensor(ymax, dtype=torch.float32),
                                'd': torch.tensor(d, dtype=torch.float32),
                                'powerRef': torch.tensor(dr, dtype=torch.float32),
                                'cost': torch.tensor(BuildingController.TOUPricing(currentTime, self.nsteps_eff)[np.newaxis,:,np.newaxis], dtype=torch.float32),
                                'stored': torch.tensor(np.array([self.sensorValues['batterySOC']])[np.newaxis,:,np.newaxis], dtype=torch.float32),
                                'batRef': torch.tensor(self.socSchedule.take(range(currentMinutes, currentMinutes+self.nsteps_eff), axis=0, mode='wrap')[np.newaxis,:], dtype=torch.float32),
                                'batMax': torch.tensor(np.ones((1,self.nsteps_eff+1,1))*8.0, dtype=torch.float32),
                                'name': 'horizon'}
            
            normDict = {'yn': 'y', 'y': 'y', 'ymin': 'y', 'ymax': 'y', 'd': 'd', 'cost': 'p'}

            self.horizonData = self.norm.norm(self.horizonData, keys=normDict)
        elif self.testCase == 'MPC_alt':  
            # pos = self.weather.index.get_loc(currentTime)
            # d = self.weather['Site Outdoor Air Temperature'].iloc[pos:pos+self.nsteps_eff].to_numpy()[:,np.newaxis]

            self.horizonData = {
                'y0': np.array([self.sensorValues['indoorAirTemp']]),
                'd': d,
                'hvac_prev': self.hvac_queue,
                'stored0': np.array([self.sensorValues['batterySOC']]),
                'bat_min': np.ones((self.nsteps_eff,1))*0.1*self.batSize,
                'bat_max': np.ones((self.nsteps_eff,1))*15,
                'cost': BuildingController.TOUPricing(currentTime, self.nsteps_eff)[:,np.newaxis],
                # 'power_ref': self.power_ref[currentMinutes:currentMinutes+self.nsteps,:],
                'power_ref': coordinateSignals[:int(self.nsteps_eff / self.step_mins)][:,np.newaxis],
                'bat_ref': self.socSchedule.take(range(currentMinutes, currentMinutes+self.nsteps_eff), axis=0, mode='wrap'),
                'charge_incen': self.chargeSchedule[currentMinutes:currentMinutes+self.nsteps_eff,:],
                'base_load': self.baseLoad[currentMinutes:currentMinutes+self.nsteps_eff,:],
            }

            if self.HVAC_mode == 1:
                self.horizonData['y_ref'] = self.sp_cool_sched[currentMinutes:currentMinutes+self.nsteps_eff,:] + self.setpointInfo['deadband']
            elif self.HVAC_mode == 3:
                self.horizonData['y_ref'] = self.sp_heat_sched[currentMinutes:currentMinutes+self.nsteps_eff,:] - self.setpointInfo['deadband']

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
        if self.testCase == 'DPC':
            trajectories = self.PredictiveControl()
        elif self.testCase == 'MPC':
            trajectories = self.SimpleMPC(currentTime)
        elif self.testCase == 'MPC_alt':
            trajectories = self.SimpleMPC_alt()

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
    
    def SimpleMPC_alt(self):

        # Select hvac optimization
        if self.HVAC_mode == 1:
            self.prob = self.prob_cool
        elif self.HVAC_mode == 3:
            self.prob = self.prob_heat
            
        # Run optimization
        for key, param in self.prob.param_dict.items():
            param.value = self.horizonData[key]
        try:
            self.prob.solve(solver='clarabel')
        except cp.SolverError:
            self.logger.warn("Solver error occurred")
        control_traj = self.prob.var_dict

        # Infeasible/error handling
        if control_traj['u_hvac'].value is None:
            self.logger.warn(f"MPC optization failed for {self.buildingID}")
            self.feasible = False
            if self.HVAC_mode == 1:
                self.actuatorValues['coolingSetpoint'] = self.horizonData['y_ref'][0,0] + self.setpointInfo['deadband']
                self.actuatorValues['heatingSetpoint'] = 10
            elif self.HVAC_mode == 3:
                self.actuatorValues['coolingSetpoint'] = 35
                self.actuatorValues['heatingSetpoint'] = self.horizonData['y_ref'][0,0] - self.setpointInfo['deadband']
        # Set setpoint based on u_hvac
        else:
            if self.count == 0:
                if (control_traj['u_hvac'].value[0,0] > 0.5):
                    if self.HVAC_mode == 1:
                        self.actuatorValues['coolingSetpoint'] = control_traj['y'].value[1,0]
                        self.actuatorValues['heatingSetpoint'] = 0
                    elif self.HVAC_mode == 3:
                        self.actuatorValues['coolingSetpoint'] = 40
                        self.actuatorValues['heatingSetpoint'] = control_traj['y'].value[1,0]
                    # self.count += 1
                else:
                    if self.HVAC_mode == 1:
                        self.actuatorValues['coolingSetpoint'] = control_traj['y'].value[1,0] + 0.8
                        self.actuatorValues['heatingSetpoint'] = 0
                    elif self.HVAC_mode == 3:
                        self.actuatorValues['coolingSetpoint'] = 40
                        self.actuatorValues['heatingSetpoint'] = control_traj['y'].value[1,0] - 0.8
                    # self.count += 1
            else:
                self.count += 1
                if self.count >= 5:
                    self.count = 0

        # Hard limits on setpoint
        if self.actuatorValues['coolingSetpoint'] <= 10:
            self.actuatorValues['coolingSetpoint'] = 10
        if self.actuatorValues['heatingSetpoint'] >= 30:
            self.actuatorValues['heatingSetpoint'] = 30

        # Move variables and parameters into one dict
        trajectories = {}
        for key, value in self.prob.param_dict.items():
            trajectories[key] = value.value

        if control_traj['u_tot'].value is None:
            trajectories['y'] = self.horizonData['y0'][np.newaxis,:]
            trajectories['u_hvac'] = np.zeros((self.nsteps_eff,1))
            trajectories['u_bat_hvac'] = np.zeros((self.nsteps_eff,1))
            trajectories['stored'] = self.horizonData['stored0'][np.newaxis,:]
            trajectories['u_bat'] = np.zeros((self.nsteps_eff,1))
            trajectories['u_tot'] = trajectories['u_bat'] + trajectories['u_hvac'] + trajectories['base_load']
            trajectories['u_load'] = trajectories['u_hvac'] + trajectories['base_load']
        else:
            for key, value in self.prob.var_dict.items():
                trajectories[key] = value.value

        # u_hvac fixing
        trajectories['u_hvac_old'] = trajectories['u_hvac']
        if self.HVAC_mode == 1:
            trajectories['u_hvac'] = np.clip(trajectories['u_hvac'], a_min=0, a_max=2)
            # trajectories['u_hvac'] = np.where(trajectories['u_hvac'] > 1, np.ones((self.nsteps_eff,1))*2, trajectories['u_hvac'])
        elif self.HVAC_mode == 3:
            trajectories['u_hvac'] = np.clip(trajectories['u_hvac'], a_min=0, a_max=7)
            # trajectories['u_hvac'] = np.where(trajectories['u_hvac'] > 2, np.ones((self.nsteps_eff,1))*7, trajectories['u_hvac'])

        self.hvac_queue[:-1] = self.hvac_queue[1:]
        self.hvac_queue[-1] = trajectories['u_hvac'][0,0]

        # Run battery fixing optimization
        # hvac_shift = np.concat((self.hvac_queue, trajectories['u_hvac'][:-self.hvac_offset]), axis=0)
        # hvac_fix = hvac_shift.copy()
        # for i in range(hvac_shift.shape[0]-3):
        #     if (hvac_shift[i] == 7) and (hvac_shift[i+1] < 7):
        #         hvac_fix[i+1] = 7
        #         hvac_fix[i+2] = 7
        #         hvac_fix[i+3] = 7
        # self.horizonData['u_load'] = hvac_fix + self.horizonData['base_load']
        # self.horizonData['u_tot_old'] = trajectories['u_tot']
        # for key, param in self.prob_bat.param_dict.items():
        #     param.value = self.horizonData[key]
        # try:
        #     self.prob_bat.solve(solver='clarabel')
        # except cp.SolverError:
        #     self.logger.warn("Solver error occurred")

        # trajectories['u_hvac'] = hvac_fix
        # trajectories['u_bat_old'] = trajectories['u_bat']
        # trajectories['u_tot_old'] = trajectories['u_tot']
        # trajectories['u_load_old'] = trajectories['u_load']
        # if not(self.prob_bat.var_dict['u_bat'].value is None):
        #     trajectories['u_bat'] = self.prob_bat.var_dict['u_bat'].value
        #     trajectories['u_tot'] = self.prob_bat.var_dict['u_tot'].value
        #     trajectories['u_load'] = self.prob_bat.param_dict['u_load'].value

        # Update battery actuator value
        if (trajectories['u_bat'][0,0] < 0) and \
        (self.sensorValues['batterySOC'] <= self.horizonData['bat_min'][0,0] + 0.05):
        # if (self.bat_hvac_queue[0] < 0) and \
        # (self.sensorValues['batterySOC'] <= self.horizonData['bat_min'][0,0] + 0.05):
        #     self.bat_hvac_queue[:-1] = self.bat_hvac_queue[1:]
        #     self.bat_hvac_queue[-1] = 0
            self.actuatorValues['battery'] = 0
        else:
            # self.bat_hvac_queue[:-1] = self.bat_hvac_queue[1:]
            # self.bat_hvac_queue[-1] = trajectories['u_bat_hvac'][0,0]
            # self.actuatorValues['battery'] = self.bat_hvac_queue[0]
            self.actuatorValues['battery'] = trajectories['u_bat'][0,0]
        
        self.feasible = True

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

    def CreateMPC_alt(self):
        RC = 400
        alpha = 0.08

        dirName = os.path.dirname(__file__)
        with open(os.path.join(dirName, 'thermal_rc_model/buildings_tuned_map.json')) as fp:
            buildingRC = json.load(fp)
        RC = buildingRC[self.buildingID]['RC']
        alpha = buildingRC[self.buildingID]['alpha']

        # manager = RunManager(self.runName, saveDir='deployModels')
        # manager.LoadRunJson(self.runName)

        # for key in manager.models.keys():
        #     if key.find('buildingNODE') != -1:
        #         thermalModelName = key
        #     elif key.find('controller') != -1:
        #         controllerModelName = key
        #     else:
        #         self.logger.warn(f"Building {self.buildingID}: Model name '{key}' does not meet expected naming conventions. Key will be ignored.")
            
        # if self.nstepsOverride is None:
        #     self.nsteps = manager.models[controllerModelName]['train_params']['nsteps']
        # else:
        #     self.nsteps = self.nstepsOverride

        self.nsteps_eff = int(self.nstepsOverride / self.step_mins)

        self.prob_cool = self.MPC_Cool(RC, alpha)
        self.logger.debug(f"Building {self.buildingID} cooling optimization")
        self.logger.debug(f"Is DPP? {self.prob_cool.is_dcp(dpp=True)}")
        self.logger.debug(f"Is DCP? {self.prob_cool.is_dcp(dpp=False)}")
        self.prob_heat = self.MPC_Heat(RC, alpha)
        self.logger.debug(f"Building {self.buildingID} heating optimization")
        self.logger.debug(f"Is DPP? {self.prob_heat.is_dcp(dpp=True)}")
        self.logger.debug(f"Is DCP? {self.prob_heat.is_dcp(dpp=False)}")
        self.prob_bat = self.MPC_Bat()

    def MPC_Cool(self, RC, alpha):
        u_hvac = cp.Variable((self.nsteps_eff,1), name='u_hvac', bounds=[0,10])
        u_hvac_shift = cp.Variable((self.nsteps_eff,1), name='u_hvac_shift', bounds=[0,10])
        u_bat = cp.Variable((self.nsteps_eff,1), name='u_bat')
        # u_bat_hvac = cp.Variable((self.nsteps_eff,1), name='u_bat_hvac')
        u_load = cp.Variable((self.nsteps_eff,1), name='u_load')
        u_tot = cp.Variable((self.nsteps_eff,1), name='u_tot')
        y = cp.Variable((self.nsteps_eff,1), name='y')
        stored = cp.Variable((self.nsteps_eff,1), name='stored')

        y0 = cp.Parameter((1), name='y0')
        yref = cp.Parameter((self.nsteps_eff,1), name='y_ref')
        d = cp.Parameter((self.nsteps_eff,1), name='d')
        hvac_prev = cp.Parameter((self.hvac_offset,1), name='hvac_prev')
        stored0 = cp.Parameter((1), name='stored0')
        batmin = cp.Parameter((self.nsteps_eff,1), name='bat_min')
        batmax = cp.Parameter((self.nsteps_eff,1), name='bat_max')
        cost = cp.Parameter((self.nsteps_eff,1), name='cost')
        power_ref = cp.Parameter((self.nsteps_eff,1), name='power_ref')
        charge_incen = cp.Parameter((self.nsteps_eff,1), name='charge_incen')
        base_load = cp.Parameter((self.nsteps_eff,1), name='base_load')

        objective = cp.Minimize(
                        3.0*cost.T@(u_load)
                        +1.0*np.ones(self.nsteps_eff-1)@cp.power(cp.diff(u_hvac),2)
                        +0.1*np.ones(self.nsteps_eff-1)@cp.power(cp.diff(u_bat),2)
                        +0.1*cp.norm(u_hvac)
                        +0.1*cp.norm(u_bat)
                        +8.0*charge_incen.T@u_bat
                        )

        constraints = [y[0] == y0,
                       y[1:] <= yref[1:],
                       u_hvac_shift[:self.hvac_offset] == hvac_prev,
                       u_hvac_shift[self.hvac_offset:] == u_hvac[:-self.hvac_offset],
                       u_load == u_hvac_shift + u_bat,
                    #    u_tot == u_hvac + u_bat + u_bat_hvac + base_load,
                       u_tot == u_hvac_shift + u_bat + base_load,
                       u_bat <= self.inverterSize, u_bat >= -(u_load),
                    #    u_bat <= self.inverterSize, u_bat >= -(base_load),
                    #    u_bat_hvac <= 0, u_bat_hvac >= -(u_hvac),
                       stored[0] == stored0,
                       stored[1:] >= batmin[1:],
                       stored[1:] <= batmax[1:],
                       u_tot <= power_ref
                    ]
        for i in range(1, self.nsteps_eff):
            constraints.append(y[i] == (-1/(RC)*y[i-1] + 1/(RC)*d[i-1] + alpha*-u_hvac[i-1])*self.step_mins + y[i-1])
            constraints.append(stored[i] == stored[i-1] + u_bat[i-1] * self.step_mins / 60)

        prob = cp.Problem(objective, constraints)

        return prob
    
    def MPC_Heat(self, RC, alpha):
        u_hvac = cp.Variable((self.nsteps_eff,1), name='u_hvac', bounds=[0,10])
        u_hvac_shift = cp.Variable((self.nsteps_eff,1), name='u_hvac_shift', bounds=[0,10])
        u_bat = cp.Variable((self.nsteps_eff,1), name='u_bat')
        # u_bat_hvac = cp.Variable((self.nsteps_eff,1), name='u_bat_hvac')
        u_load = cp.Variable((self.nsteps_eff,1), name='u_load')
        u_tot = cp.Variable((self.nsteps_eff,1), name='u_tot')
        y = cp.Variable((self.nsteps_eff,1), name='y')
        stored = cp.Variable((self.nsteps_eff,1), name='stored')

        y0 = cp.Parameter((1), name='y0')
        yref = cp.Parameter((self.nsteps_eff,1), name='y_ref')
        d = cp.Parameter((self.nsteps_eff,1), name='d')
        hvac_prev = cp.Parameter((self.hvac_offset,1), name='hvac_prev')
        stored0 = cp.Parameter((1), name='stored0')
        batmin = cp.Parameter((self.nsteps_eff,1), name='bat_min')
        batmax = cp.Parameter((self.nsteps_eff,1), name='bat_max')
        cost = cp.Parameter((self.nsteps_eff,1), name='cost')
        power_ref = cp.Parameter((self.nsteps_eff,1), name='power_ref')
        charge_incen = cp.Parameter((self.nsteps_eff,1), name='charge_incen')
        base_load = cp.Parameter((self.nsteps_eff,1), name='base_load')

        objective = cp.Minimize(
                        3.0*cost.T@(u_load)
                        +1.0*np.ones(self.nsteps_eff-1)@cp.power(cp.diff(u_hvac),2)
                        +0.1*np.ones(self.nsteps_eff-1)@cp.power(cp.diff(u_bat),2)
                        +0.1*cp.norm(u_hvac)
                        +0.1*cp.norm(u_bat)
                        +8.0*charge_incen.T@u_bat
                        )

        constraints = [y[0] == y0,
                       y[1:] >= yref[1:],
                       u_hvac_shift[:self.hvac_offset] == hvac_prev,
                       u_hvac_shift[self.hvac_offset:] == u_hvac[:-self.hvac_offset],
                       u_load == u_hvac_shift + base_load,
                    #    u_tot == u_hvac + u_bat + u_bat_hvac + base_load,
                       u_tot == u_hvac_shift + u_bat  + base_load,
                       u_bat <= self.inverterSize, u_bat >= -(u_load),
                    #    u_bat <= self.inverterSize, u_bat >= -(base_load),
                    #    u_bat_hvac <= 0, u_bat_hvac >= -(u_hvac),
                       stored[0] == stored0,
                       stored[1:] >= batmin[1:],
                       stored[1:] <= batmax[1:],
                       u_tot <= power_ref,
                    #    u_hvac >= 0,
                    #    u_hvac <= 10,
                    ]
        for i in range(1, self.nsteps_eff):
            constraints.append(y[i] == (-1/(RC)*y[i-1] + 1/(RC)*d[i-1] + alpha*u_hvac[i-1])*self.step_mins + y[i-1])
            constraints.append(stored[i] == stored[i-1] + u_bat[i-1] * self.step_mins / 60)

        prob = cp.Problem(objective, constraints)

        return prob
    
    def MPC_Bat(self):
        u_bat = cp.Variable((self.nsteps_eff,1), name='u_bat')
        u_tot = cp.Variable((self.nsteps_eff,1), name='u_tot')
        stored = cp.Variable((self.nsteps_eff,1), name='stored')

        u_load = cp.Parameter((self.nsteps_eff,1), name='u_load')
        u_tot_old = cp.Parameter((self.nsteps_eff,1), name='u_tot_old')
        stored0 = cp.Parameter((1), name='stored0')
        batmin = cp.Parameter((self.nsteps_eff,1), name='bat_min')
        batmax = cp.Parameter((self.nsteps_eff,1), name='bat_max')
        power_ref = cp.Parameter((self.nsteps_eff,1), name='power_ref')
        charge_incen = cp.Parameter((self.nsteps_eff,1), name='charge_incen')

        objective = cp.Minimize(
                        # +0.1*np.ones(self.nsteps_eff-1)@cp.power(cp.diff(u_bat),2)
                        # +0.1*cp.norm(u_bat)
                        # +8.0*charge_incen.T@u_bat
                        +1.0*cp.norm(u_tot_old-u_tot)
                        )

        constraints = [
                       u_tot == u_bat + u_load,
                       u_bat <= self.inverterSize, u_bat >= -(u_load),
                       stored[0] == stored0,
                       stored[1:] >= batmin[1:],
                       stored[1:] <= batmax[1:],
                       u_tot <= power_ref,
                    ]
        for i in range(1, self.nsteps_eff):
            constraints.append(stored[i] == stored[i-1] + u_bat[i-1] * self.step_mins / 60)

        prob = cp.Problem(objective, constraints)

        return prob

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