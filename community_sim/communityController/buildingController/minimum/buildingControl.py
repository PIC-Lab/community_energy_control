import torch
import numpy as np
import pandas as pd

from modelConstructor import *

def Main():
    # torch.manual_seed(0)

    # ----- Set model parameters -----
    # Building thermal model
    thermal_params = {
        'train_params': {
            'max_epochs': 1000,
            'patience': 100,
            'warmup': 200,
            'lr': 0.001,
            'nsteps': 30,
            'batch_size': 30
        }
    }
    
    # Controller model
    controller_params = {
        'weights': {'action_loss': 0.01, 'du_loss': 0.1,
                    'x_min': 10.0, 'x_max': 10.0},
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
    
    alfData = pd.read_csv('data files/4_out.csv', usecols=['living space Air Temperature', 'Heating:Electricity', 'Site Outdoor Air Temperature'], nrows=57600)
    buildingData = pd.read_csv('data files/building4_data.csv', usecols=['indoor temp', 'outdoor temp'], nrows=57600)
    A_full = pd.read_csv('data files/matrixA.csv', index_col=0)
    B_full = pd.read_csv('data files/matrixB.csv', index_col=0)
    B_reduced = B_full.loc[:, ['H_LIV']]
    F_reduced = B_full.loc[:,['T_EXT', 'T_GND']]
    C_full = pd.read_csv('data files/matrixC.csv', index_col=0)

    states = A_full.columns.to_list()
    y_idx = states.index('T_LIV')
    ochData = pd.read_parquet('data files/Envelope_OCHRE.parquet', engine='pyarrow', columns=['Time', 'T_GND'] + states)
    stateData = ochData.loc[:,states].to_numpy()
    stateData[:,y_idx] = buildingData['indoor temp'].to_numpy()

    dataset = {}
    dataset['X'] = stateData
    dataset['Y'] = buildingData['indoor temp'].to_numpy()[:, np.newaxis]
    dataset['U'] = np.column_stack([np.zeros_like(alfData['Heating:Electricity'])])
    dataset['D'] = np.column_stack([buildingData['outdoor temp'], ochData['T_GND']])

    stateDataSim = np.zeros_like(ochData.loc[:,states])
    stateDataSim[:,y_idx] = alfData['living space Air Temperature'].to_numpy()
    datasetSim = {}
    datasetSim['X'] = stateDataSim
    datasetSim['Y'] = alfData['living space Air Temperature'].to_numpy()[:, np.newaxis]
    datasetSim['U'] = alfData['Heating:Electricity'].to_numpy()[:, np.newaxis]
    datasetSim['U'] = datasetSim['U'] > 0.5
    datasetSim['D'] = np.column_stack([alfData['Site Outdoor Air Temperature'], ochData['T_GND']])

    # Bounds on indoor temperature
    tempMin = torch.tensor(16.0)
    tempMax = torch.tensor(24.0)

    # Building thermal model
    buildingThermal = BuildingRC(nx=dataset['X'].shape[1],
                                ny=dataset['Y'].shape[1],
                                nu=dataset['U'].shape[1],
                                nd=dataset['D'].shape[1],
                                params=thermal_params,
                                A=torch.tensor(A_full.to_numpy(), dtype=torch.float32),
                                B=torch.tensor(B_reduced.to_numpy(), dtype=torch.float32),
                                F=torch.tensor(F_reduced.to_numpy(), dtype=torch.float32),
                                C=torch.tensor(C_full.to_numpy(), dtype=torch.float32),
                                name='thermal')

    buildingThermal.CreateModel()

    buildingThermal.TrainModel(dataset, trainMode='Natural')

    buildingThermal.TestModel(label='Natural')

    buildingThermal.TrainModel(datasetSim, trainMode='forced')

    buildingThermal.TestModel(label='Full')

    # Controller
    controlSystem = ControllerSystem(nx=datasetSim['X'].shape[1],
                                    nu=datasetSim['U'].shape[1],
                                    nd=datasetSim['D'].shape[1],
                                    nd_obs=1,
                                    ny=datasetSim['Y'].shape[1],
                                    y_idx=[y_idx],
                                    d_idx=[0],
                                    params=controller_params,
                                    name='controller',
                                    thermalModel=buildingThermal.model)
    controlSystem.CreateModel()

    controlSystem.TrainModel(datasetSim, tempMin, tempMax)

    controlSystem.TestModel(datasetSim, tempMin)

if __name__ == '__main__':
    Main()