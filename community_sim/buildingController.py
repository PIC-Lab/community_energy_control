import json
import pandas as pd
import numpy as np
import sys

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
    def __init__(self, id, testCase='MPC'):
        """
        Constructor
        Parameters:
            id: (str) id of the controller's building
            testCase: (str) name of test case being run, defaults to MPC
        """
        self.actuatorValues = {'heatingSetpoint': 0, 'coolingSetpoint': 0}
        self.sensorValues = {'indoorAirTemp': 0}
        if id.isnumeric():
            self.buildingID = id
        else:
            self.buildingID = id[0:id.find('_')]
        self.controllerID = id

        # HVAC Values
        self.HVAC_mode = 0

        # MPC variables
        self.cl_system = None
        self.norm = None
        self.pv = None
        self.dist = None

        # Load building specific parameters
        with open('buildingSetpoints.json') as jsonData:
            buildingInfo = json.load(jsonData)
        # self.setpointInfo = buildingInfo[self.controllerID]       # defines temperature bounds
        self.setpointInfo = {"heatSP": 18.88888888888889, "coolSP": 24.444444444444443, "deadband": 4.444444444444445}      # Default value, should be commented out

        self.tempBounds = np.ones((1,24*60+60, 2))       # Hard coded to be one day long, will need to change
        self.tempBounds[:,:,0] *= self.setpointInfo['heatSP'] - self.setpointInfo['deadband']     # min
        self.tempBounds[:,:,1] *= self.setpointInfo['coolSP'] + self.setpointInfo['deadband']     # max

        # Run additional setup functions
        if testCase == 'MPC':
            self.LoadMPC()
            self.count = 0
            self.HVAC_lock = 0
            self.HVAC_prevMode = 0

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
        self.PredictiveControl()

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

        # match self.HVAC_mode:
        #     case 0:
        #         self.actuatorValues['heatingSetpoint'] = 0
        #         self.actuatorValues['coolingSetpoint'] = 100
        #     case 1:
        #         self.actuatorValues['heatingSetpoint'] = 0
        #         self.actuatorValues['coolingSetpoint'] = self.setpointInfo['coolSP'] + 2 * self.setpointInfo['deadband']
        #     case 2:
        #         self.actuatorValues['heatingSetpoint'] = 0
        #         self.actuatorValues['coolingSetpoint'] = self.setpointInfo['coolSP'] + 2 * self.setpointInfo['deadband']
        #     case 3:
        #         self.actuatorValues['heatingSetpoint'] = self.setpointInfo['heatSP'] - 2 * self.setpointInfo['deadband']
        #         self.actuatorValues['coolingSetpoint'] = 100
        #     case 4:
        #         self.actuatorValues['heatingSetpoint'] = self.setpointInfo['heatSP'] - 2 * self.setpointInfo['deadband']
        #         self.actuatorValues['coolingSetpoint'] = 100

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

    def PredictiveControl(self, currentStep):
        '''
        Run predictive control for the prediction horizon
        Inputs:
            currentStep: (int) current step of simulation loop (simulation only)
        '''
        # Prepare dataset
        print('stateData', self.stateData.shape)
        y = self.sensorValues['indoorAirTemp']
        self.stateData[0,0,self.y_idx] = y
        ymin = self.tempBounds[:,currentStep:currentStep+self.nsteps,0:1]
        ymax = self.tempBounds[:,currentStep:currentStep+self.nsteps,1:2]
        d = self.dist[:,currentStep:currentStep+self.nsteps,:]
        data = {'xn': torch.tensor(self.stateData, dtype=torch.float32),
            'y': torch.tensor(np.array([y])[np.newaxis,:,np.newaxis], dtype=torch.float32),
            'ymin': torch.tensor(ymin, dtype=torch.float32),
            'ymax': torch.tensor(ymax, dtype=torch.float32),
            'd': torch.tensor(d, dtype=torch.float32)}
        
        # Run control
        trajectories = self.cl_system(data)
        print('trajectories', trajectories['xn'].shape)
        self.stateData = trajectories['xn'][:,-2:-1,:]

        control = trajectories['u'][0,0,0].detach()
        # if (control == 0):                  # off
        #     self.HVAC_mode = 0
        # elif (control == 1):                # cool
        #     self.HVAC_mode = 1  
        # else:
        #     print("HVAC Mode Error")

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
        # Import controller model classes
        sys.path.insert(1, '../thermal_rc_model')
        from modelConstructor import BuildingRC          # type:ignore
        from modelConstructor import ControllerSystem    # type:ignore
        from modelConstructor import Normalizer          # type:ignore
        from modelConstructor import ModeClassifier      # type:ignore
        import runManager                                # type:ignore

        device = torch.device("cpu")

        run = 'latestRun'
        # run = 'alf_AB_classifier_goated_edit'
        # run = 'alf_SB_classifier_splitPV_1'

        manager = runManager.RunManager(run, saveDir='../thermal_rc_model/savedRuns')
        manager.LoadRunJson(run)

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

        A_full = pd.read_csv('../thermal_rc_model/models/Envelope_OCHRE_matrixA.csv', index_col=0)
        B_full = pd.read_csv('../thermal_rc_model/models/Envelope_OCHRE_matrixB.csv', index_col=0)
        B_reduced = B_full.loc[:, ['H_LIV']]
        F_reduced = B_full.loc[:,['T_EXT', 'T_GND']]
        C_full = pd.read_csv('../thermal_rc_model/models/Envelope_OCHRE_matrixC.csv', index_col=0)
        states = A_full.columns.to_list()
        self.y_idx = states.index('T_LIV')

        buildingData = pd.read_csv('../thermal_rc_model/building4_data.csv', usecols=['indoor temp', 'outdoor temp'], nrows=57600)
        ochData = pd.read_parquet('../thermal_rc_model/Envelope_OCHRE.parquet', engine='pyarrow', columns=['Time', 'T_GND'] + states)
        self.dist = np.column_stack([buildingData['outdoor temp'], ochData['T_GND']])
        self.dist = self.dist[np.newaxis,:]
        self.stateData = ochData.loc[:,states].iloc[0:1,:].to_numpy()
        self.stateData[0,self.y_idx] = buildingData.loc[:, 'indoor temp'].iloc[0]
        self.stateData = self.stateData[np.newaxis,:]

        # Thermal model definition
        thermalModel = BuildingRC(nx=33,
                                    ny=1,
                                nu=1,
                                nd=2,
                                A=torch.tensor(A_full.to_numpy(), dtype=torch.float32),
                                B=torch.tensor(B_reduced.to_numpy(), dtype=torch.float32),
                                F=torch.tensor(F_reduced.to_numpy(), dtype=torch.float32),
                                C=torch.tensor(C_full.to_numpy(), dtype=torch.float32),
                                manager=manager,
                                name=thermalModelName,
                                device=device,
                                debugLevel=0,
                                saveDir=f"../thermal_rc_model/{manager.runPath+thermalModelName}/{self.buildingID}",)
        thermalModel.CreateModel()

        thermalModel.TrainModel(None, load=True, test=False)

        # Classifier model definition
        classifier = ModeClassifier(nm=1,
                                    nu=1,
                                    manager=manager,
                                    name=classifierModelName,
                                    device=device,
                                    debugLevel = 0,
                                    saveDir=f"{manager.runPath+classifierModelName}")
        classifier.CreateModel()

        classifier.TrainModel(None, load=True, test=False)

        # Controller model definition
        controlSystem = ControllerSystem(nx=33,
                                         nu=1,
                                         nd=2,
                                         nd_obs=1,
                                         ny=1,
                                         y_idx=[self.y_idx],
                                         d_idx=[0],
                                         manager=manager,
                                         name=controllerModelName,
                                         thermalModel=thermalModel.model,
                                         classifier=classifier.model,
                                         device=device,
                                         debugLevel=0,
                                         saveDir=f"../thermal_rc_model/{manager.runPath+controllerModelName}/{self.buildingID}")
        controlSystem.CreateModel()

        controlSystem.TrainModel(None, None, None, load=True, test=False)
        self.cl_system = controlSystem.system