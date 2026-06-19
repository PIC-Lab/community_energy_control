from dotenv import dotenv_values
import json
import os
import numpy as np
import requests
import pandas as pd

from communityController.buildingController.buildingController import BuildingController
from communityController.coordinator.community_optim import Coordinator
from communityController.flexibility_metrics.flexibilityMetrics import FlexibilityMetricPredictor

class CommunityController:
    '''
    '''
    def __init__(self, controlAliasList, runName, logger, baseLoad, testCase='DPC', deploy=False,
                 nstepsCoord=240, stepSizeCoord=1, nstepsBuild=60, stepSizeBuild=1):
        '''
        '''
        dirName = os.path.dirname(__file__)

        self.controlAliasList = controlAliasList

        self.nsteps = nstepsCoord
        self.stepSize = stepSizeCoord

        self.stepCount = stepSizeCoord

        # self.nstepsBuild = nstepsBuild
        # self.stepSizeBuild = stepSizeBuild
        self.stepSize_ratio = int(self.stepSize / stepSizeBuild)
        self.nstepsBuild_eff = int(nstepsBuild / self.stepSize_ratio)
        
        self.logger = logger
        self.testCase = testCase
        self.deploy = deploy

        self.controllerList = []
        self.flexibilityList = []
        self.trajectoryList = {}

        self.coordDebug = {}

        self.predictedLoad = np.zeros((len(controlAliasList), self.nsteps))
        self.predictedFlex = np.zeros((len(controlAliasList), self.nsteps, 2))
        self.coordinateSignals_eff = np.zeros((len(controlAliasList), nstepsBuild))

        self.baseTransLoad = None

        config = dotenv_values(os.path.join(dirName,'.env'))

        if self.deploy:
            self.nws_userAgent = config['NWS_USERAGENT']
            self.WeatherInit(config['LATITUDE'], config['LONGITUDE'])

        self.ControllerInit(runName, testCase, baseLoad, nstepsBuild, stepSizeBuild, dirName)
        self.CoordinatorInit(runName)
        self.FlexibilityInit(nstepsCoord, stepSizeCoord, runName)

    def WeatherInit(self, latitude, longitude):
        r = requests.get(f"https://api.weather.gov/points/{latitude},{longitude}", headers={"UserAgent": f"({self.nws_userAgent})"})
        response = json.loads(r)
        self.weather_loc = {
            'gridId': response['properties']['gridId'],
            'gridX': response['properties']['gridX'],
            'gridY': response['properties']['gridY']
        }

    def WeatherPull(self):
        if self.deploy:     # Deployment environment
            # ----- UNFINISHED -----
            r = requests.get(f"https://api.weather.gov/gridpoints/{self.weather_loc['gridId']}/{self.weather_loc['gridX']},{self.weather_loc['gridY']}/forecast/hourly?units=us",
                             headers={"UserAgent": f"({self.nws_userAgent})"})
            response = json.loads(r)
            response['properties']['periods']
            # ----------------------
        else:               # Simulation environment
            dirName = os.path.dirname(__file__)
            self.weather = pd.read_csv(os.path.join(dirName, '../sim_schedules/2024_weather.csv'), usecols=['Time', 'Site Outdoor Air Temperature'])
            self.weather.set_index(pd.to_datetime(self.weather['Time'], format="%Y-%m-%d %H:%M:%S"), inplace=True)
            self.weather.drop('Time', axis=1, inplace=True)

    def SetpointPull(self):
        if self.deploy:         # Deployment environment
            pass
        else:
            dirName = os.path.dirname(__file__)
            sp_cool = pd.read_csv(os.path.join(dirName,'../sim_schedules/sp_cool_sched.csv'), header=0, index_col=None)
            sp_heat = pd.read_csv(os.path.join(dirName,'../sim_schedules/sp_heat_sched.csv'), header=0, index_col=None)
            
        return (sp_cool, sp_heat)

    def CoordinatorInit(self, runName):
        '''
        '''
        self.coordinator = Coordinator(self.controlAliasList, self.nsteps, 1, self.logger)
        self.coordinator.AdjustInit()
        self.logger.debug("Coordinator initialized")

    def ControllerInit(self, runName, testCase, baseLoad, nsteps, stepSize, dirName):
        '''
        '''
        # if self.mode == 'deploy':
        #     self.config = dotenv_values('../.env')

        self.logger.debug(self.controlAliasList)
        
        # Get all devices from a building using API
        # Using json file for now
        with open(os.path.join(dirName, '../configs/buildingDeviceList.json')) as fp:
            buildingDevices = json.load(fp)

        self.WeatherPull()
        sp_cool, sp_heat = self.SetpointPull()
                
        # Create controller object
        for alias in self.controlAliasList:
            for building in buildingDevices:
                if building['house_id'] == alias:
                    devices = building['devices']
                    break
            self.controllerList.append(BuildingController(id=alias,
                                                          devices=devices,
                                                          runName=runName,
                                                          logger=self.logger,
                                                          weather=self.weather,
                                                          sp_cool=sp_cool[alias].values,
                                                          sp_heat=sp_heat[alias].values,
                                                          baseLoad=baseLoad[alias].values,
                                                          testCase=testCase,
                                                          nsteps=nsteps,
                                                          stepSize=stepSize))
            self.logger.debug(f"Created building controller for {alias}")
        
        self.logger.debug("Building controllers initialized")

    def FlexibilityInit(self, nsteps, stepSize, runName):
        '''
        '''
        for alias in self.controlAliasList:
            self.flexibilityList.append(FlexibilityMetricPredictor(alias, nsteps, stepSize, runName, self.logger))
        self.logger.debug("Flexibility predictor initialized")

    def Step(self, sensorValues, currentTime):
        '''
        Run every time step
        Returns a list of dicts containing the control events for each house. Matches controls API format except for device IDs
        '''
        currentMinutes = int((currentTime.hour * 60 + currentTime.minute) / self.stepSize)

        # Forecast flexibility
        for i, alias in enumerate(self.controlAliasList):
            self.flexibilityList[i].Step(currentTime)
            self.predictedFlex[i,:,:] = self.flexibilityList[i].flexBounds

        # Update coordinator
        self.stepCount += 1
        if self.stepCount >= self.stepSize:
            self.coordinator.predictedLoad = self.predictedLoad
            self.coordinator.predictedFlex = self.predictedFlex

            self.logger.debug("Coordinator step")
            self.coordinator.baseLoad = self.baseTransLoad[:,currentMinutes:currentMinutes+self.nsteps]
            self.coordinator.Step()
            # coordinateSignals = self.predictedLoad + self.coordinator.adjustValues['flexLoad'] + 0.1
            # coordinateSignals = np.where(self.coordinator.adjustValues['flexLoad'] == 0,
            #                             np.ones_like(self.predictedLoad)*20 ,
            #                             self.predictedLoad + self.coordinator.adjustValues['flexLoad'])
            coordinateSignals = self.coordinator.adjustValues['buildBoundLoad']
            self.coordinateSignals_eff = np.repeat(coordinateSignals, self.stepSize_ratio, axis=1)
            self.stepCount = 0

        # Update controllers
        self.logger.debug("Building controllers step")
        controlEvents = []
        for i, alias in enumerate(self.controlAliasList):
            trajectories = self.controllerList[i].Step(sensorValues[alias], self.coordinateSignals_eff[i,:], currentTime)

            # Building level data collection
            self.trajectoryList[alias] = trajectories
            self.coordDebug[alias] = {}
            self.coordDebug[alias]['usagePenalty'] = self.coordinator.usagePenalty[i]
            self.coordDebug[alias]['flexLoad'] = self.coordinator.adjustValues['flexLoad'][i,:]
            self.coordDebug[alias]['flexBoundLoad'] = self.coordinator.adjustValues['flexBoundLoad'][i,:]
            self.coordDebug[alias]['flexPredLoad'] = self.coordinator.adjustValues['flexPredLoad'][i,:]
            self.coordDebug[alias]['predLoad'] = self.coordinator.predictedLoad[i,:]
            self.coordDebug[alias]['lowerBound'] = self.coordinator.predictedFlex[i,:,0]
            self.coordDebug[alias]['upperBound'] = self.coordinator.predictedFlex[i,:,1]

            if self.testCase == 'DPC':
                self.predictedLoad[i,:] = trajectories['horizon_u_tot'].detach().numpy()[0,:,0]
            elif (self.testCase == 'MPC') or (self.testCase == 'MPC_alt'):
                # self.predictedLoad[i,:self.nstepsBuild] = trajectories['u_tot'][:self.nstepsBuild,0]
                self.predictedLoad[i,:self.nstepsBuild_eff] = trajectories['u_tot'].reshape(self.nstepsBuild_eff, self.stepSize_ratio).mean(axis=1)
            if self.nstepsBuild_eff < self.nsteps:      # Forecast horizon of buildings shorter than coordinator
                self.predictedLoad[i,self.nstepsBuild_eff:] = np.zeros(self.nsteps-self.nstepsBuild_eff)
            controlEvents.append(self.controllerList[i].controlEvents)

        # Shift coordinate signals forward 1 building timestep
        self.coordinateSignals_eff[:,:-1] = self.coordinateSignals_eff[:,1:]
        self.coordinateSignals_eff[:,-1] = self.coordinateSignals_eff[:,-2]

        # Coordinator level data collection
        self.logger.debug("Coordinator data collection")
        self.coordDebug['gen'] = {}
        for i in range(len(self.coordinator.transInfo.keys())):
            self.coordDebug['gen'][f"trans{i+1} lim"] = self.coordinator.transLim[i:i+1]
            # totalLoad = self.coordinator.adjustValues['flexLoad'] + self.coordinator.predictedLoad
            self.coordDebug['gen'][f"trans{i+1} bound load"] = self.coordinator.adjustValues['transBoundLoad'][i,:]
            self.coordDebug['gen'][f"trans{i+1} pred load"] = self.coordinator.adjustValues['transPredLoad'][i,:]
            self.coordDebug['gen'][f"trans{i+1} base load"] = self.coordinator.baseLoad[i,:]
        self.coordDebug['gen']['coord feas'] = [self.coordinator.adjustProb.feasible]
        self.coordDebug['gen']['coord state'] = [self.coordinator.state]
        
        return controlEvents

        

        