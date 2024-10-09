# Numpy + plotting utilities + ordered dicts
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd

# Standard PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Neuromancer imports
from neuromancer.psl.coupled_systems import *
from neuromancer.dynamics import integrators, ode, physics, interpolation
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.problem import Problem
from neuromancer.loss import PenaltyLoss
from neuromancer.system import Node, System
from neuromancer.loggers import BasicLogger
from neuromancer.trainer import Trainer
from neuromancer.modules import blocks
from neuromancer.callbacks import Callback

def Main():
    # Fix seeds for reproducibility
    np.random.seed(200)
    torch.manual_seed(0)
    load = False

    nx = 3
    nu = 2
    ts = 1
    nsteps = 60
    bs = 100

    # df = pd.read_csv('building4_data.csv', nrows=57600, usecols=['outdoor temp', 'indoor temp'])
    # dataset = pd.DataFrame()
    # dataset['X1'] = df['indoor temp']
    # dataset['U1'] = df['outdoor temp']
    # dataset['U2'] = np.zeros_like(df['outdoor temp'])
    # print(dataset.describe())

    callback = Callback_Basic()

    df = pd.read_csv('1_out.csv', nrows=57600, usecols=['Site Outdoor Air Temperature', 'living space Air Temperature', 'air source heat pump airloop Discharge Air Temp Sensor'])
    dataset = pd.DataFrame()
    dataset['X1'] = df['living space Air Temperature']
    dataset['U1'] = df['Site Outdoor Air Temperature']
    dataset['U2'] = df['air source heat pump airloop Discharge Air Temp Sensor']
    print(dataset.describe())
    
    # Possible include an on/off input

    trainLoader, devLoader, testData = GetData(dataset, nsteps, bs, nx, nu)

    # Create RC network
    numZones = 3
    zones = [physics.RCNode(C=nn.Parameter(torch.tensor(5.0)),scaling=1.0e-4) for i in range(numZones)]
    heaters = [physics.SourceSink() for i in range(numZones)]
    outside = [physics.SourceSink()]
    agents = zones + heaters + outside

    map = physics.map_from_agents(agents)

    couplings = []
    # Couple zone to outside
    couplings.append(physics.DeltaTemp(R=nn.Parameter(torch.tensor(100.0)),pins=[[0,2]]))
    couplings.append(physics.DeltaTemp(R=nn.Parameter(torch.tensor(100.0)),pins=[[0,2]]))
    # Couple zone to heater
    couplings.append(physics.DeltaTemp(R=nn.Parameter(torch.tensor(0.01)),pins=[[0,1]]))

    model_ode = ode.GeneralNetworkedODE(
        map = map,
        agents = agents,
        couplings = couplings,
        insize = nx+nu,
        outsize = nx,
        inductive_bias="compositional"
    )

    fx_int = integrators.RK2(model_ode, h=1.0)

    dynamics_model = System([Node(fx_int,['xn', 'U'],['xn'], name='RC')])

    x = variable("X")
    xhat = variable("xn")[:, :-1, 0:1]

    reference_loss = ((xhat == x)^2)
    reference_loss.name = "ref_loss"

    objectives = [reference_loss]
    constraints = []

    loss = PenaltyLoss(objectives, constraints)

    problem = Problem([dynamics_model], loss)

    if load:
        problem.load_state_dict(torch.load('testRC/best_model_state_dict.pth', weights_only=True))
        problem.eval()
    else:
        optimizer = torch.optim.AdamW(problem.parameters(), lr = 0.001)
        logger = BasicLogger(args=None, savedir='testRC', verbosity=1,
                            stdout=["dev_loss","train_loss"])
        
        trainer = Trainer(
            problem,
            trainLoader,
            devLoader,
            testData,
            optimizer,
            epochs=1000,
            patience=100,
            warmup=50,
            eval_metric="dev_loss",
            train_metric="train_loss",
            dev_metric="dev_loss",
            test_metric="dev_loss",
            logger=logger, 
            callback=callback
        )

        best_model = trainer.train()

    trajectories = dynamics_model(testData)
    pred_traj = trajectories['xn'][:,:-1,:].detach().numpy().reshape(-1, nx)

    true_traj = testData['X'].detach().numpy().reshape(-1,nx)
    input_traj = testData['U'].detach().numpy().reshape(-1,nu)

    fig, ax = plt.subplots(2, figsize=(10,5))
    u = torch.from_numpy(input_traj).float()
    sol = torch.zeros((true_traj.shape[0],1))
    ic = torch.tensor(true_traj[0,:])
    for j in range(sol.shape[0]-1):
        if j==0:
            sol[[0],:] = ic.float()
            sol[[j+1],:] = fx_int(sol[[j],:],u[[j],:])
        else:
            sol[[j+1],:] = fx_int(sol[[j],:],u[[j],:])

        
    ax[0].plot(sol.detach().numpy(), label='model', color='black')
    ax[0].plot(pred_traj, label='pred')
    ax[0].plot(true_traj, label='data', color='red')
    ax[1].plot(input_traj, label='input')
    for x in ax:
        x.set_xlim([0,1000])
    plt.legend()
    fig.savefig('./Saved Figures/temperaturePred', dpi=fig.dpi)

    loss_df = pd.DataFrame(callback.loss)
    loss_df = loss_df[['train_loss', 'dev_loss']]
    loss_df['train_loss'] = loss_df['train_loss'].apply(lambda x: x.detach().item())
    loss_df['dev_loss'] = loss_df['dev_loss'].apply(lambda x: x.detach().item())
    fig = plt.figure()
    plt.plot(loss_df)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loss_df.keys())
    plt.tight_layout()
    fig.savefig('./Saved Figures/building_loss', dpi=fig.dpi)
    plt.close(fig)

    for zone in zones:
        print(zone.C)

    for couple in couplings:
        print(couple.R)

def GetData(data_df, nsteps, bs, nx, nu):
    n = len(data_df)
    mean = data_df.mean()
    std = data_df.std()
    data_df = (data_df - mean) / std
    data_df.fillna(0, inplace=True)
    
    train_df = data_df[:int(np.round(n*0.7))]
    dev_df = data_df[int(np.round(n*0.7)):int(np.round(n*0.9))]
    test_df = data_df[int(np.round(n*0.9)):]

    nbatch = len(test_df)//nsteps

    train = {}
    train['X'] = np.column_stack([train_df[f'X{i}'] for i in range(1,nx+1)])
    train['U'] = np.column_stack([train_df[f'U{i}'] for i in range(1,nu+1)])

    dev = {}
    dev['X'] = np.column_stack([dev_df[f'X{i}'] for i in range(1,nx+1)])
    dev['U'] = np.column_stack([dev_df[f'U{i}'] for i in range(1,nu+1)])

    test = {}
    test['X'] = np.column_stack([test_df[f'X{i}'] for i in range(1,nx+1)])
    test['U'] = np.column_stack([test_df[f'U{i}'] for i in range(1,nu+1)])


    trainX = train['X'][:n].reshape(nbatch*7, nsteps, nx)
    trainX = torch.tensor(trainX, dtype=torch.float32)
    trainU = train['U'][:n].reshape(nbatch*7, nsteps, nu)
    trainU = torch.tensor(trainU, dtype=torch.float32)
    trainData = DictDataset({'X': trainX, 'xn': trainX[:,0:1,:],
                             'U': trainU}, name='train')
    trainLoader = DataLoader(trainData, batch_size=bs,
                             collate_fn=trainData.collate_fn, shuffle=True)
    
    devX = dev['X'][:n].reshape(nbatch*2, nsteps, nx)
    devX = torch.tensor(devX, dtype=torch.float32)
    devU = dev['U'][:n].reshape(nbatch*2, nsteps, nu)
    devU = torch.tensor(devU, dtype=torch.float32)
    devData = DictDataset({'X': devX, 'xn': devX[:,0:1,:],
                             'U': devU}, name='dev')
    devLoader = DataLoader(devData, batch_size=bs,
                             collate_fn=devData.collate_fn, shuffle=True)
    
    testX = test['X'][:n].reshape(nbatch, nsteps, nx)
    testX = torch.tensor(testX, dtype=torch.float32)
    testU = test['U'][:n].reshape(nbatch, nsteps, nu)
    testU = torch.tensor(testU, dtype=torch.float32)
    testData = {'X': testX, 'xn': testX[:,0:1,:], 'U': testU}

    return trainLoader, devLoader, testData

class Callback_Basic(Callback):
    def __init__(self):

        # Monitoring loss by epoch
        self.loss = []

    def end_eval(self, trainer, output):
        self.loss.append(output)


if __name__ == '__main__':
    Main()