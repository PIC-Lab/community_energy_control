import torch
import numpy as np
import pandas as pd
import shutil
from pathlib import Path

import runManager
from modelConstructor import *

def Main():
    # torch.manual_seed(0)
    # If cuda is available, run on GPU
    if torch.cuda.is_available():
        dev = 'cuda:1'
        print("CUDA is available, running on GPU")
    else:
        dev = "cpu"
        print("CUDA is not available, running on CPU")
    device = torch.device(dev)
    device = torch.device("cpu")

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
        name = "alf_AllBuildings"

    manager = runManager.RunManager(name)

    # Load a previous run based on a name
    if loadRun:
        manager.LoadRunJson(name)
        loadThermal = True
        loadClass = True
        loadMPC = True

        for key in manager.models.keys():
            if key.find('buildingThermal') != -1:
                thermalModelName = key
            elif key.find('classifier') != -1:
                classifierModelName = key
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

        # Classifier model
        classifierModelName = "classifier"
        manager.models[classifierModelName] = {
            'hsizes': [64, 64],
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
            'weights': {'action_loss': 0.01, 'dr_loss': 0.1,
                        'x_min': 10.0, 'x_max': 10.0},
            # 'hsizes': [32,32],
            # 'hsizes': [64,64],
            'hsizes': [100,100],
            'train_params': {
                'max_epochs': 200,
                'patience': 30,
                'warmup': 50,
                'lr': 0.01,
                'nsteps': 30,
                'batch_size': 50,
                'n_samples': 1000
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
        manager.dataset['path'] = 'building4_data.csv'
        manager.dataset['sliceBool'] = True
        manager.dataset['slice_idx'] = [0, 57600]
        # ----------------------------------

        manager.tempBounds = [16.0, 24.0]
        
        manager.WriteRunJson()

    # Get building ids
    # buildingModels = Path('../community_sim/building_models/')
    # buildings = []
    # for file in buildingModels.iterdir():
    #     if (file / 'workflow.osw').exists():
    #         buildings.append(file.name)
    buildings = ['4']

    # Train classifier
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

    tol = 1e-6
    iterations = 1
    i = 0
    count = 0
    bestLoss = 1e5
    attempts = 0
    while(i < len(buildings)):
        building = buildings[i]
        print(f"Training models for building {building}, round {count}")
        alfData = pd.read_csv('../community_sim/results/summer/4_out.csv', usecols=['living space Air Temperature', 'Electricity:HVAC', 'Site Outdoor Air Temperature'], nrows=57600)
        # buildingData = pd.read_csv('building4_data.csv', usecols=['indoor temp', 'outdoor temp'], nrows=57600)

        if manager.dataset['sliceBool']:
            raw_dataset = alfData[manager.dataset['slice_idx'][0]:manager.dataset['slice_idx'][1]].copy()
        else:
            raw_dataset = alfData.copy()

        print(raw_dataset.describe())

        dataset = {}
        dataset['X'] = raw_dataset['living space Air Temperature'].to_numpy()[:, np.newaxis]
        dataset['U'] = raw_dataset['Electricity:HVAC'].to_numpy()[:, np.newaxis]
        dataset['D'] = raw_dataset['Site Outdoor Air Temperature'].to_numpy()[:, np.newaxis]

        # Bounds on indoor temperature
        tempMin = torch.tensor(manager.tempBounds[0]).to(device=device)
        tempMax = torch.tensor(manager.tempBounds[1]).to(device=device)

        if (attempts == 0) and (count == 0):
            # Building thermal model
            buildingThermal = BuildingNode(nx=dataset['X'].shape[1],
                                    nu=dataset['U'].shape[1],
                                    nd=dataset['D'].shape[1],
                                    manager=manager,
                                    name=thermalModelName,
                                    device=device,
                                    debugLevel = DebugLevel.NO,
                                    saveDir=f"{manager.runPath+thermalModelName}/{building}")
        buildingThermal.CreateModel()

        buildingThermal.TrainModel(dataset, loadThermal)

        buildingThermal.TestModel()

        # Controller
        controlSystem = ControllerSystem(nx=dataset['X'].shape[1],
                                        nu=dataset['U'].shape[1],
                                        nd=dataset['D'].shape[1],
                                        nd_obs=1,
                                        ny=1,
                                        y_idx=[0],
                                        d_idx=[0],
                                        manager=manager,
                                        name=controllerModelName,
                                        thermalModel=buildingThermal.model,
                                        classifier=classifier.model,
                                        device=device,
                                        debugLevel=DebugLevel.EPOCH_VALUES,
                                        saveDir=f"{manager.runPath+controllerModelName}/{building}")
        controlSystem.CreateModel()

        controlSystem.TrainModel(dataset, tempMin, tempMax, loadMPC)

        controlSystem.TestModel(dataset, tempMin, tempMax)

        # Skip repeat training if loading a previously saved run
        if loadRun or (loadThermal and loadClass and loadMPC):
            i += 1
            continue
        # Repeat training if it got stuck somewhere
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
            if count >= iterations:
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

if __name__ == '__main__':
    Main()