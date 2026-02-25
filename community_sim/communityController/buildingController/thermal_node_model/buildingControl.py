import torch
import numpy as np
import pandas as pd
import shutil
import json
import resource

import runManager
from modelConstructor import *

def Main():
    # Setting memory limits
    # resource.setrlimit(resource.RLIMIT_AS, (int(200 * 1e9), int(250 * 1e9)))

    print("--- Run Manager ---")
    userInput = input('Would you like to load a previous run? (y/n) ')
    if userInput.lower() == 'y':
        loadRun = True
        saveToLatest = True
        device = torch.device('cpu')
        # Should put some error checking here at some point
        name = input('What run would you like to load? (case sensitive) ')
        if name == '':
            name = 'latestRun'
            print(f"No name given, defaulting to {name}")
    else:
        print("--- CUDA Setup ---")
        if torch.cuda.is_available():
            print(f"CUDA is available. There are {torch.cuda.device_count()} GPUs available. Which would you like to run on?")
            looping = True
            while looping:
                devNum = input("Enter a number, will default to 0 if left blank. You may also enter 'cpu' if desired. ")
                if devNum == '':
                    dev = 'cuda:0'
                    looping = False
                elif devNum.lower() == 'cpu':
                    dev = 'cpu'
                    looping = False
                else:
                    try:
                        assert int(devNum) < torch.cuda.device_count()
                        dev = f'cuda:{devNum}'
                        looping = False
                    except (ValueError, AssertionError):
                        print(f'{devNum} is not a valid cuda device.')
        else:
            dev = "cpu"
            print("CUDA is not available, running on CPU")
        device = torch.device(dev)
        loadRun = False
        userInput = input('Are you doing training runs in parallel? (y/n) ')
        if userInput.lower() == 'y':
            saveToLatest = False
            name = input('What name should this train be saved under? ')
            if name == '':
                name = "alf_AllBuildings"
                print(f"No name given, defaulting to {name}")
        else:
            name = "alf_AllBuildings"
            saveToLatest = True

    manager = runManager.RunManager(name, saveToLatest=saveToLatest)

    # Load a previous run based on a name
    if loadRun:
        manager.LoadRunJson(name)
        loadThermal = True
        loadMPC = True

        for key in manager.models.keys():
            if key.find('buildingThermal') != -1:
                thermalModelName = key
            elif key.find('controller') != -1:
                controllerModelName = key
            else:
                print(f"Model name '{key}' does not meet expected naming conventions. Key will be ignored.")

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

        userInput = input('Do you want to load the controller model? (y/n) ')
        if userInput.lower() == 'y':
            loadMPC = True
        else:
            loadMPC = False
            userInput = input("Should this run include SDA gradient projection? (y/n) ")
            if userInput.lower() == 'y':
                gradProj = True
            else:
                gradProj = False

        # ----- Set model parameters -----
        # Building thermal model
        thermalModelName = "buildingThermal"
        manager.models[thermalModelName] = {
            'hsizes': [100],
            'train_params': {
                'max_epochs': 500,
                'patience': 20,
                'warmup': 50,
                'lr': 0.003,
                'nsteps': 60,
                'batch_size': 30
            }
        }
        
        # Controller model
        controllerModelName = "controller"
        manager.models[controllerModelName] = {
            'weights': {'cost_loss': 2.0, 'delta_loss': 2.0,
                        'follow_limit': 15.0, 'coordRef': 15.0,
                        'x_min': 20.0, 'x_max': 20.0, 'bat_min': 15.0, 'bat_max': 10.0},
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
            }
        }
        # -----------------------------

        if loadThermal or loadMPC:
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
    with open('../../../configs/simParams.json') as fp:
        simParams = json.load(fp)
    buildingModels = Path('../../../building_models/')
    buildings = []
    for file in buildingModels.iterdir():
        if (file / 'workflow.osw').exists() and (file.name in simParams['controlledAliases']):
            buildings.append(file.name)

    tol = 1e-6              # Tolerance when determining if training loss improved at all
    bestLoss = 1e5          # Best achieved loss over multiple training attempts
    maxIterations = 2       # Maximum attempts allowed for finding the best training loss

    count = 1 
    attempts = 0
    i = 0
    while(i < len(buildings)):
        building = buildings[i]
        print(f"Training models for building {building}, round {count}")
        alfData = pd.read_csv(manager.dataset['path']+f'{building}_out.csv', usecols=['Time', 'living space Air Temperature', 'cooling setpoint', 'Electricity:HVAC', 'Site Outdoor Air Temperature'], nrows=57600)
        dates = pd.to_datetime(alfData['Time'], format='%Y-%m-%d %H:%M:%S')
        alfData['Price'] = dates.apply(lambda x: TOUPricing(x, timeSteps=1440))

        if manager.dataset['sliceBool']:
            raw_dataset = alfData.iloc[manager.dataset['slice_idx'][0]:manager.dataset['slice_idx'][1]].loc[:,['living space Air Temperature', 'Electricity:HVAC', 'Site Outdoor Air Temperature', 'Price']].copy()
        else:
            raw_dataset = alfData.loc[:, ['living space Air Temperature', 'Electricity:HVAC', 'Site Outdoor Air Temperature', 'Price']].copy()

        # upData = pd.DataFrame(index=dates, columns=raw_dataset.columns, data=raw_dataset.to_numpy())
        
        # upData = upData.reindex(index=upsampleDates)
        # upData.interpolate(inplace=True)
        # raw_dataset = upData

        print(raw_dataset.describe())

        norm = Normalizer()
        norm.add_data(raw_dataset)
        norm.add_data(raw_dataset, keys=['y', 'u', 'd', 'p'])
        norm.dataInfo['p']['min'] = 0
        norm.save(f"{manager.runPath}norm/{building}/")
        dataset_norm = norm.norm(raw_dataset, keys=['y', 'u', 'd', 'p'])

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
        if (attempts == 0) and (count == 1):
            buildingThermal = BuildingNode(nx=dataset['X'].shape[1],
                                    nu=dataset['U'].shape[1],
                                    nd=dataset['D'].shape[1],
                                    manager=manager,
                                    norm=norm,
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
                                        thermalModel=buildingThermal.problem,
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
        if loadRun or (loadThermal and loadMPC):
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
            count = 1
            bestLoss = 1e5
            attempts = 0

    # Update run json with init params for each block (Used when loading for deployment)
    manager.WriteRunJson()

if __name__ == '__main__':
    Main()