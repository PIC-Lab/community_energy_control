import torch
import numpy as np
import pandas as pd
import shutil
import json
import resource

import runManager
from modelConstructor_projGrad import *

def Main():
    # Setting memory limits
    resource.setrlimit(resource.RLIMIT_AS, (int(200 * 1e9), int(250 * 1e9)))

    # torch.manual_seed(0)
    # If cuda is available, run on GPU
    if torch.cuda.is_available():
        dev = 'cuda:0'
        print("CUDA is available, running on GPU")
    else:
        dev = "cpu"
        print("CUDA is not available, running on CPU")
    device = torch.device(dev)
    # device = torch.device("cpu")

    # Check if to load a previous run
    print("---Run Manager---")
    userInput = input('Would you like to load a previous run? (y/n) ')
    if userInput.lower() == 'y':
        loadRun = True
        # Should put some error checking here at some point
        name = input('What run would you like to load? (case sensitive) ')
        if name == '':
            name = 'latestRun'
            print(f"No name given, defaulting to {name}")
    else:
        loadRun = False
        # name = "alf_AllBuildings"
        name = 'projGrad_fine_6'

    manager = runManager.RunManager(name, saveToLatest=True)

    # Load a previous run based on a name
    if loadRun:
        manager.LoadRunJson(name)
        loadThermal = True
        loadClass = True
        loadMPC = True

        for key in manager.models.keys():
            if key.find('buildingThermal') != -1:
                thermalModelName = key
                expModelName = key
            elif key.find('classifier') != -1:
                classifierModelName = key
            elif key.find('setpoint') != -1:
                setpointName = key
            elif key.find('controller') != -1:
                controllerModelName = key
            else:
                raise ValueError(f"Model name '{key}' does not meet expected naming conventions.")
            
    # Populate run manager for saving later
    else:
        # Check if specific models should be loaded
        manager.PrepNewRun()
        print('---Models---')
        userInput = input('Do you want to load the building thermal model? (y/n) ')
        if userInput.lower() == 'y':
            loadThermal = True
        else:
            loadThermal = False

        userInput = input('Do you want to load the classifier model? (y/n) ')
        if userInput.lower() == 'y':
            loadClass = True
        else:
            loadClass = False

        userInput = input('Do you want to load the controller model? (y/n) ')
        if userInput.lower() == 'y':
            loadMPC = True
        else:
            loadMPC = False

        # ----- Set model parameters -----
        # Building thermal model
        thermalModelName = "buildingThermal"
        manager.models[thermalModelName] = {
            'hsizes': [200,200],
            'train_params': {
                'max_epochs': 1000,
                'patience': 50,
                'warmup': 100,
                'lr': 0.001,
                'nsteps': 60,
                'batch_size': 30
            }
        }

        # expModelName = "experimental buildingThermal"
        # manager.models[expModelName] = {
        #     'hsizes': [200,200],
        #     'train_params': {
        #         'max_epochs': 1000,
        #         'patience': 50,
        #         'warmup': 100,
        #         'lr': 0.0005,
        #         'nsteps': 60,
        #         'batch_size': 50
        #     }
        # }

        # Classifier model
        classifierModelName = "classifier"
        manager.models[classifierModelName] = {
            'hsizes': [64],
            'train_params': {
                'max_epochs': 500,
                'patience': 50,
                'warmup': 100,
                'lr': 0.001,
                'nsteps': 30,
                'batch_size': 10
            }
        }

        setpointName = "setpoint"
        manager.models[classifierModelName] = {
            'hsizes': [64],
            'train_params': {
                'max_epochs': 500,
                'patience': 50,
                'warmup': 100,
                'lr': 0.001,
                'nsteps': 30,
                'batch_size': 10
            }
        }
        
        # Controller model
        controllerModelName = "controller"
        manager.models[controllerModelName] = {
            'weights': {'dr_loss': 6.0, 'cost_loss': 5.0, 'delta_loss': 1.0,
                        'hvac_loss': 0.5, 'bat_loss': 2.0, 'bat_max_loss': 0.7,
                        'x_min': 20.0, 'x_max': 20.0, 'bat_min': 15.0},
            # 'hsizes': [32,32],
            # 'hsizes': [64,64],
            'hsizes': [200,200],
            'train_params': {
                'max_epochs': 200,
                'patience': 30,
                'warmup': 50,
                'lr': 0.001,
                'nsteps': 60,
                'batch_size': 20,
                'n_samples': 200,
                'projHVAC_steps': 10,
                'projHVAC_size': 35,
                'projBat_steps': 15,
                'projBat_size': 10
            }
        }
        # -----------------------------

        if loadThermal or loadClass or loadMPC:
            modelRunName = input('What run should the model(s) be loaded from? ')
            if modelRunName == '':
                # ----- Default run name -----
                modelRunName = 'alf_AllBuildings'
                # ----------------------------
                print(f"No name given, defaulting to {modelRunName}")
            manager_modelLoad = runManager.RunManager(modelRunName)

            if loadThermal:
                for key in manager_modelLoad.models.keys():
                    if key.find('buildingThermal') != -1:
                        thermalModelName = key
                shutil.copytree(f'{manager.saveDir}/{modelRunName}/{thermalModelName}', manager.runPath+thermalModelName)
                manager.LoadModel(modelRunName, thermalModelName)
            if loadClass:
                for key in manager_modelLoad.models.keys():
                    if key.find('classifier') != -1:
                        classifierModelName = key
                shutil.copytree(f'{manager.saveDir}/{modelRunName}/{classifierModelName}', manager.runPath+classifierModelName)
                manager.LoadModel(modelRunName, classifierModelName)
            if loadMPC:
                for key in manager_modelLoad.models.keys():
                    if key.find('controller') != -1:
                        controllerModelName = key
                shutil.copytree(f'{manager.saveDir}/{modelRunName}/{controllerModelName}', manager.runPath+controllerModelName)
                manager.LoadModel(modelRunName, controllerModelName)

        # ----- Set dataset parameters -----
        manager.dataset['path'] = '../../../results/summer/'
        # manager.dataset['path'] = 'Saved Figures/SF_HP.csv'
        manager.dataset['sliceBool'] = True
        manager.dataset['slice_idx'] = [0, 57600]
        # ----------------------------------

        manager.tempBounds = [16.0, 24.0]
        
        manager.WriteRunJson()

    # Get building ids
    with open('../../../simParams.json') as fp:
        simParams = json.load(fp)
    buildingModels = Path('../../../building_models/')
    buildings = []
    for file in buildingModels.iterdir():
        if (file / 'workflow.osw').exists() and (file.name in simParams['controlledAliases']):
            buildings.append(file.name)

    buildings = ['4']

    # ------------ Train classifier ------------
    tempList = []
    while len(tempList) < manager.dataset['slice_idx'][1]:
        value = np.random.uniform(0, 1)
        duration = np.random.randint(1,30)
        tempList.extend([value] * duration)

    tempList = tempList[:manager.dataset['slice_idx'][1]]

    classifierData = {}
    classifierData['U'] = np.array(tempList)[:,np.newaxis]
    classifierData['M'] = classifierData['U'] > 0.5
    
    classifier = ModeClassifier(nm=classifierData['M'].shape[1],
                                    nu=classifierData['U'].shape[1],
                                    manager=manager,
                                    name=classifierModelName,
                                    device=device,
                                    debugLevel = DebugLevel.EPOCH_LOSS,
                                    saveDir=f"{manager.runPath+classifierModelName}")
    classifier.CreateModel()

    classifier.TrainModel(classifierData, loadClass)

    classifier.TestModel()

    # setpointData = {}

    # setpoint = SetpointPredictor(nu=setpointData['M'].shape[1],
    #                                 nu=setpointData['U'].shape[1],
    #                                 manager=manager,
    #                                 name=setpointName,
    #                                 device=device,
    #                                 debugLevel = DebugLevel.EPOCH_LOSS,
    #                                 saveDir=f"{manager.runPath+setpointName}")
    # setpoint.CreateModel()

    # setpoint.TrainModel(setpointData, loadClass)

    # setpoint.TestModel()

    manager.models["classifier"]["init_params"] = {'nm': classifier.nm, 'nu': classifier.nu}
    # ------------------------------------------

    tol = 1e-6              # Tolerance when determining if training loss improved at all
    bestLoss = 1e5          # Best achieved loss over multiple training attempts
    maxIterations = 1       # Maximum attempts allowed for finding the best training loss

    count = 0   
    attempts = 0
    i = 0
    while(i < len(buildings)):
        building = buildings[i]
        print(f"Training models for building {building}, round {count}")
        alfData = pd.read_csv(manager.dataset['path']+f'{building}_out.csv', usecols=['Time', 'living space Air Temperature', 'cooling setpoint', 'Electricity:HVAC', 'Site Outdoor Air Temperature'], nrows=57600)
        dates = pd.to_datetime(alfData['Time'], format='%Y-%m-%d %H:%M:%S')
        alfData['Price'] = dates.apply(lambda x: TOUPricing(x, timeSteps=1440))

        if manager.dataset['sliceBool']:
            raw_dataset = alfData.iloc[manager.dataset['slice_idx'][0]:manager.dataset['slice_idx'][1]].loc[:,['living space Air Temperature', 'cooling setpoint', 'Electricity:HVAC', 'Site Outdoor Air Temperature', 'Price']].copy()
        else:
            raw_dataset = alfData.loc[:, ['living space Air Temperature', 'cooling setpoint', 'Electricity:HVAC', 'Site Outdoor Air Temperature', 'Price']].copy()

        # upData = pd.DataFrame(index=dates, columns=raw_dataset.columns, data=raw_dataset.to_numpy())
        
        # upData = upData.reindex(index=upsampleDates)
        # upData.interpolate(inplace=True)
        # raw_dataset = upData

        print(raw_dataset.describe())

        norm = Normalizer()
        norm.add_data(raw_dataset)
        norm.add_data(raw_dataset, keys=['y', 'u', 'x', 'd', 'p'])
        norm.dataInfo['p']['min'] = 0
        norm.save(f"{manager.runPath}norm/{building}/")
        dataset_norm = norm.norm(raw_dataset, keys=['y', 'u', 'x', 'd', 'p'])

        print(dataset_norm.describe())

        dataset = {}
        dataset['X'] = dataset_norm['living space Air Temperature'].to_numpy()[:, np.newaxis]
        dataset['U'] = dataset_norm['Electricity:HVAC'].to_numpy()[:, np.newaxis]
        dataset['D'] = dataset_norm['Site Outdoor Air Temperature'].to_numpy()[:, np.newaxis]
        dataset['I'] = dataset_norm['Price'].to_numpy()[:,np.newaxis]

        # dataset = {}
        # dataset['X'] = np.stack([dataset_norm['living space Air Temperature'].to_numpy(), dataset_norm['Electricity:HVAC'].to_numpy()], axis=1)
        # dataset['U'] = dataset_norm['cooling setpoint'].to_numpy()[:, np.newaxis]
        # dataset['D'] = dataset_norm['Site Outdoor Air Temperature'].to_numpy()[:, np.newaxis]
        # dataset['I'] = dataset_norm['Price'].to_numpy()[:,np.newaxis]

        # Bounds on indoor temperature
        tempMin = torch.tensor(norm.norm(manager.tempBounds[0], keys=['y'])).to(device=device)
        tempMax = torch.tensor(norm.norm(manager.tempBounds[1], keys=['y'])).to(device=device)

        # ---------- Train thermal model -----------
        if (attempts == 0) and (count == 0):
            buildingThermal = BuildingNode(nx=dataset['X'].shape[1],
                                    nu=dataset['U'].shape[1],
                                    nd=dataset['D'].shape[1],
                                    manager=manager,
                                    name=thermalModelName,
                                    device=device,
                                    debugLevel = DebugLevel.EPOCH_VALUES,
                                    saveDir=f"{manager.runPath+thermalModelName}/{building}")
            # buildingThermal = ExperimentalThermalModel(nx=dataset['X'].shape[1],
            #                         nu=dataset['U'].shape[1],
            #                         nd=dataset['D'].shape[1],
            #                         manager=manager,
            #                         name=expModelName,
            #                         device=device,
            #                         debugLevel = DebugLevel.EPOCH_VALUES,
            #                         saveDir=f"{manager.runPath+expModelName}/{building}")
            buildingThermal.CreateModel()

            buildingThermal.TrainModel(dataset, loadThermal)

            buildingThermal.TestModel()
            
            manager.models["buildingThermal"]["init_params"] = {'nx': buildingThermal.nx, 'nu': buildingThermal.nu, 'nd': buildingThermal.nd}
        # ------------------------------------------

        # ------------ Train controller ------------
        controlSystem = ControllerSystem(nx=dataset['X'].shape[1],
                                        nu=dataset['U'].shape[1],
                                        nd=dataset['D'].shape[1],
                                        nd_obs=1,
                                        ni=2,
                                        ny=1,
                                        y_idx=[0],
                                        d_idx=[0],
                                        manager=manager,
                                        name=controllerModelName,
                                        norm=norm,
                                        thermalModel=buildingThermal.model,
                                        classifier=classifier.model,
                                        device=device,
                                        debugLevel=DebugLevel.EPOCH_VALUES,
                                        saveDir=f"{manager.runPath+controllerModelName}/{building}")
        controlSystem.CreateModel()

        controlSystem.TrainModel(dataset, tempMin, tempMax, loadMPC)

        controlSystem.TestModel(dataset, tempMin, tempMax)

        manager.models["controller"]["init_params"] = {'nx': controlSystem.nx, 'nu': controlSystem.nu, 'nd': controlSystem.nd,
                                                       'nd_obs': controlSystem.nd_obs, 'ny': controlSystem.ny,
                                                       'y_idx': controlSystem.y_idx, 'd_idx': controlSystem.d_idx}
        # ------------------------------------------

        # Skip repeat training if loading a previously saved run
        if loadRun or (loadThermal and loadClass and loadMPC):
            i += 1
            continue
        # Repeat training if it got stuck somewhere
        # This section is spaghetti code, make it better at some point
        loss_df = pd.read_csv(controlSystem.saveDir+'/loss.csv')
        temp_df = np.abs(loss_df.iloc[0] - loss_df) < tol
        if temp_df.all(axis=None):
            if attempts >= 3:
                print(f"No improvement for building {building} over {attempts} attempts. Giving up")
                nextBuilding = True
            else:
                print(f"No improvement, training building {building} again, attempt {attempts}")
                nextBuilding = False
                attempts += 1
        else:
            if count >= maxIterations:
                nextBuilding = True
            else:
                count += 1
                nextBuilding = False
        if bestLoss > loss_df['dev_loss'].min():
            shutil.copytree(controlSystem.saveDir, f"{controlSystem.saveDir}/best",
                            ignore=shutil.ignore_patterns('best'), dirs_exist_ok=True)
            bestLoss = loss_df['dev_loss'].min()
            print(f"New best run, best loss updated to {bestLoss}")
        if nextBuilding:
            print(f"Finished with building {i+1}, moving to next building")
            shutil.copytree(f"{controlSystem.saveDir}/best", controlSystem.saveDir, dirs_exist_ok=True)
            shutil.rmtree(f"{controlSystem.saveDir}/best")
            i += 1
            count = 0
            bestLoss = 1e5
            attempts = 0

    # Update run json with init params for each block (Used when loading for deployment)
    manager.WriteRunJson()

if __name__ == '__main__':
    Main()