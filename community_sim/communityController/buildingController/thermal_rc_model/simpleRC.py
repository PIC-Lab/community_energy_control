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

    nx = 1
    nu = 2
    ts = 1
    nsteps = 60
    bs = 32

    callback = Callback_Basic()

    # df = pd.read_csv('1_out.csv', nrows=57600, usecols=['Site Outdoor Air Temperature', 'living space Air Temperature', 'air source heat pump airloop Discharge Air Temp Sensor'])
    df = pd.read_csv('1_out.csv', nrows=57600, usecols=['Site Outdoor Air Temperature', 'living space Air Temperature', 'Cooling:Electricity', 'Whole Building Electricity'])
    dataset = pd.DataFrame()
    dataset['X1'] = df['living space Air Temperature']
    dataset['U1'] = df['Site Outdoor Air Temperature']
    dataset['U2'] = df['Cooling:Electricity']
    # dataset['U3'] = df['Whole Building Electricity']
    print(dataset.describe())
    
    # Possible include an on/off input

    trainLoader, devLoader, testData = GetData(dataset, nsteps, bs, nx, nu)
    mean = dataset.mean()
    std = dataset.std()

    rcModel = RCNetwork(insize=nx+nu, outsize=nx)
    fxRK4 = integrators.RK4(rcModel, h=ts)
    dynamics_model = System([Node(fxRK4, ['xn', 'U'], ['xn'])], nsteps=nsteps)

    x = variable("X")
    xhat = variable('xn')[:,:-1,:]

    xFD = x[:,1:,:] - x[:,:-1,:]
    xhatFD = xhat[:,1:,:] - xhat[:,:-1,:]

    fd_loss = 2.0*((xFD == xhatFD)^2)
    fd_loss.name = 'FD_loss'

    reference_loss = ((xhat == x)^2)
    reference_loss.name = "ref_loss"

    objectives = [reference_loss, fd_loss]
    constraints = []
    loss = PenaltyLoss(objectives, constraints)
    problem = Problem([dynamics_model], loss)

    optimizer = torch.optim.AdamW(problem.parameters(), lr=0.01)
    logger = BasicLogger(args=None, savedir='simpleRC', verbosity=1,
                            stdout=["dev_loss","train_loss"])
    trainer = Trainer(
        problem,
        trainLoader,
        devLoader,
        testData,
        optimizer,
        callback=callback,
        logger=logger,
        patience=20,
        warmup=20,
        epochs=200,
        eval_metric='dev_loss',
        train_metric='train_loss',
        dev_metric='dev_loss',
        test_metric='dev_loss'
    )

    best_model = trainer.train()
    problem.load_state_dict(best_model)

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
            sol[[j+1],:] = fxRK4(sol[[j],:],u[[j],:])
        else:
            sol[[j+1],:] = fxRK4(sol[[j],:],u[[j],:])

        
    ax[0].plot(sol.detach().numpy() * std[1] + mean[1], label='model', color='black')
    ax[0].plot(pred_traj * std[1] + mean[1], label='pred')
    ax[0].plot(true_traj * std[1] + mean[1], label='data', color='red')
    ax[0].legend()
    ax[1].plot(input_traj, label='input')
    for x in ax:
        x.set_xlim([0,1000])
    plt.legend()
    fig.savefig('./Saved Figures/simpleTemperaturePred', dpi=fig.dpi)

    test_mae = np.mean(np.abs(pred_traj - true_traj))
    print(f'Test mae: {test_mae}')

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
    fig.savefig('./Saved Figures/simpleBuilding_loss', dpi=fig.dpi)
    plt.close(fig)

    print('Parameter values')
    print(f"R: {rcModel.R.item()}")
    print(f"C: {rcModel.C.item()}")
    print(f"a: {rcModel.a.item()}")

# adam, 0.01, 10, 1, 1

class RCNetwork(ode.ODESystem):
    def __init__(self, insize=2, outsize=2):
        super().__init__(insize=insize, outsize=outsize)
        self.R = nn.Parameter(torch.tensor([10.0]), requires_grad=True)
        self.C = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.a = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        # self.p = nn.Parameter(torch.tensor([5.0]), requires_grad=True)
        # self.f = nn.Parameter(torch.tensor([5.0]), requires_grad=True)

    def ode_equations(self, x, u):
        x1 = x[:, [0]]
        # x2 = x[:, [-1]]
        u1 = u[:,[0]]
        u2 = u[:, [1]]
        # u3 = u[:, [-1]]
        dx1 = -1/(self.C*self.R)*x1 + 1/(self.C*self.R)*u1 + self.a*u2
        return torch.cat([dx1], dim=1)

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