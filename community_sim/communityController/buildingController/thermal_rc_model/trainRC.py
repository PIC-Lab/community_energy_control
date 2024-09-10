# Numpy + plotting utilities + ordered dicts
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd
import datetime as dt

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

def Main():
    # Fix seeds for reproducibility
    # np.random.seed(200)
    # torch.manual_seed(0)
    alfData = pd.read_csv('1_out.csv', usecols=['living space Air Temperature', 'Heating:Electricity', 'Cooling:Electricity', 'Site Outdoor Air Temperature'], nrows=57600)

    A_full = pd.read_csv('models/Envelope_OCHRE_matrixA.csv', index_col=0)
    B_full = pd.read_csv('models/Envelope_OCHRE_matrixB.csv', index_col=0)
    B_reduced = B_full.loc[:, ['H_LIV', 'T_EXT', 'T_GND']]
    C_full = pd.read_csv('models/Envelope_OCHRE_matrixC.csv', index_col=0)

    states = A_full.columns.to_list()
    ochData = pd.read_parquet('Envelope_OCHRE.parquet', engine='pyarrow', columns=['Time', 'T_GND'] + states)
    stateData = ochData.loc[:,states]
    stateData['T_LIV'] = alfData['living space Air Temperature'].to_numpy()
    
    A = torch.tensor(A_full.to_numpy(), dtype=torch.float32)
    B = torch.tensor(B_reduced.to_numpy(), dtype=torch.float32)
    C = torch.tensor(C_full.to_numpy(), dtype=torch.float32)

    nx = A.shape[0]
    nu = B.shape[1]
    ny = 1

    nsteps = 30
    nbatch = len(alfData) // nsteps
    batch_size = 30

    dataset = {}
    dataset['X'] = stateData.to_numpy()
    dataset['Y'] = alfData['living space Air Temperature'].to_numpy()[:, np.newaxis]
    dataset['U'] = np.column_stack([alfData['Heating:Electricity'] - alfData['Cooling:Electricity'],
                                    alfData['Site Outdoor Air Temperature'],
                                    ochData['T_GND']])

    n = len(dataset['X'])

    train = {}
    train['X'] = dataset['X'][:int(np.round(n*0.7)),:]
    train['Y'] = dataset['Y'][:int(np.round(n*0.7)),:]
    train['U'] = dataset['U'][:int(np.round(n*0.7)),:]

    dev = {}
    dev['X'] = dataset['X'][int(np.round(n*0.7)):int(np.round(n*0.9)),:]
    dev['Y'] = dataset['Y'][int(np.round(n*0.7)):int(np.round(n*0.9)),:]
    dev['U'] = dataset['U'][int(np.round(n*0.7)):int(np.round(n*0.9)),:]

    test = {}
    test['X'] = dataset['X'][int(np.round(n*0.9)):,:]
    test['Y'] = dataset['Y'][int(np.round(n*0.9)):,:]
    test['U'] = dataset['U'][int(np.round(n*0.9)):,:]

    nbatch = len(test['X']) // nsteps

    trainX = torch.tensor(train['X'].reshape(nbatch*7, nsteps, nx), dtype=torch.float32)
    trainU = torch.tensor(train['U'].reshape(nbatch*7, nsteps, nu), dtype=torch.float32)
    trainY = torch.tensor(train['Y'].reshape(nbatch*7, nsteps, ny), dtype=torch.float32)
    trainData = DictDataset({'x': trainX, 'xn': trainX[:,0:1,:],
                             'y': trainY, 'u': trainU}, name='train')
    for key, value in trainData.datadict.items():
        print(key, value.shape)
    trainLoader = DataLoader(trainData, batch_size=batch_size,
                            collate_fn=trainData.collate_fn, shuffle=True)
    
    devX = torch.tensor(dev['X'].reshape(nbatch*2, nsteps, nx), dtype=torch.float32)
    devU = torch.tensor(dev['U'].reshape(nbatch*2, nsteps, nu), dtype=torch.float32)
    devY = torch.tensor(dev['X'][:,-3].reshape(nbatch*2, nsteps, ny), dtype=torch.float32)
    devData = DictDataset({'x': devX, 'xn': devX[:,0:1,:],
                           'y': devY, 'u': devU}, name='dev')
    devLoader = DataLoader(devData, batch_size=batch_size,
                            collate_fn=devData.collate_fn, shuffle=True)
    
    # test_nsteps = nsteps * 4
    # test_nbatch = len(test['X']) // test_nsteps

    # testX = torch.tensor(test['X'].reshape(test_nbatch, test_nsteps, nx))
    # testU = torch.tensor(test['U'].reshape(test_nbatch, test_nsteps, nu))
    # testY = torch.tensor(test['X'][:,-3].reshape(test_nbatch, test_nsteps, ny))
    # testData = {'x': testX, 'xn': testX[:,0:1,:], 'y': testY, 'u': testU}
    
    testX = torch.tensor(test['X'].reshape(nbatch, nsteps, nx), dtype=torch.float32)
    testU = torch.tensor(test['U'].reshape(nbatch, nsteps, nu), dtype=torch.float32)
    testY = torch.tensor(test['X'][:,-3].reshape(nbatch, nsteps, ny), dtype=torch.float32)
    testData = {'x': testX, 'xn': testX[:,0:1,:], 'y': testY, 'u': testU}

    base_ynext = lambda x: x @ C.T
    base_output_model = Node(base_ynext, ['xn'], ['yn'], name='base_out_obs')

    base_xnext = lambda x, u: x @ A.T + u @ B.T
    base_ssm = Node(base_xnext, ['xn', 'u'], ['xn'])
    baseSystem = System([base_ssm, base_output_model], nsteps=nsteps)

    initialAdj = Node(IC_Adjust(nx), ['xn_0'], ['xn'], name='initialCond')

    xnext = SSM(A.clone(), B.clone(), nx, nu)
    state_model = Node(xnext, ['xn', 'u'], ['xn'], name ='base_SSM')

    ynext = lambda x: x @ C.T
    output_model = Node(ynext, ['xn'], ['yn'], name='out_obs')

    net = blocks.MLP(
         insize=ny,
         outsize=ny,
         hsizes=[32],
         nonlin=nn.GELU
    )

    outputFudge = Node(net, ['yn'], ['yfud'], name='out_fud')

    system = System([state_model, output_model], nsteps=nsteps)
    system.show('ssm.png')

    y = variable('y')
    yhat = variable('yn')
    # x = variable('x')[:,:,-3:-2]
    # xhat = variable('xn')[:,:-1,-3:-2]

    referenceLoss = 5.*(yhat == y)^2
    referenceLoss.name = 'ref_loss'

    onestepLoss = 1.*(yhat[:,1,:] == y[:,1,:])^2
    onestepLoss.name = 'onestep_loss'

    objectives = [referenceLoss, onestepLoss]
    constraints = []

    loss = PenaltyLoss(objectives, constraints)

    problem = Problem([system], loss)
    problem.show('problem.png')

    optimizer = torch.optim.Adam(problem.parameters(), lr = 0.01)
    logger = BasicLogger(args=None,
                            savedir='trained SS Model',
                            verbosity=1,
                            stdout=['dev_loss', 'train_loss'])
    
    trainer = Trainer(problem,
                        trainLoader,
                        devLoader,
                        testData,
                        optimizer,
                        patience=20,
                        warmup=50,
                        epochs=500,
                        eval_metric='dev_loss',
                        train_metric='train_loss',
                        dev_metric='dev_loss',
                        test_metric='dev_loss',
                        logger=logger)

    best_model = trainer.train()
    problem.load_state_dict(best_model)
    
    # Testing
    system.nsteps = testData['x'].shape[1]
    trajectories = system(testData)
    print('Trained')
    for key, value in trajectories.items():
            print(key, value.shape)

    baseSystem.nsteps = testData['x'].shape[1]
    base_trajectories = baseSystem(testData)
    print('Base')
    for key, value in base_trajectories.items():
            print(key, value.shape)

    pred_traj = trajectories['xn'][:,:-1,:].detach().cpu().numpy().reshape(-1,nx)
    base_traj = base_trajectories['xn'][:,:-1,:].detach().cpu().numpy().reshape(-1,nx)
    true_traj = testData['x'].detach().cpu().numpy().reshape(-1,nx)
    pred_y = trajectories['yn'].detach().cpu().numpy().reshape(-1,ny)
    base_y = base_trajectories['yn'].detach().cpu().numpy().reshape(-1,ny)
    true_y = testData['y'].detach().cpu().numpy().reshape(-1,ny)
    input_traj = testData['u'].detach().cpu().numpy().reshape(-1,nu)
    pred_traj, true_traj, base_traj = pred_traj.transpose(1,0), true_traj.transpose(1,0), base_traj.transpose(1,0)

    figsize = 25
    fig,ax = plt.subplots(2+ny+nu, figsize=(figsize, figsize))
    for i in range(nx):
        ax[0].plot(pred_traj[i,:])
    ax[0].tick_params(labelbottom=False, labelsize=figsize)
    ax[0].set_title("Predicted States", fontsize=figsize)
    # ax[0].set(xlim=(0,150))

    for i in range(nx):
        ax[1].plot(base_traj[i,:])
    ax[1].tick_params(labelbottom=False, labelsize=figsize)
    ax[1].set_title("Base States", fontsize=figsize)
    # ax[1].set(xlim=(0,150))

    # ax[2].plot(true_traj[-3,:], 'c', linewidth=4.0, label="True")
    # ax[2].plot(pred_traj[-3,:], 'm', linewidth=2.0, label='Pred')
    # ax[2].plot(base_traj[-3,:])
    ax[2].tick_params(labelbottom=False, labelsize=figsize)
    ax[2].plot(true_y, 'c', linewidth=4.0, label="True")
    ax[2].plot(pred_y, 'm', linewidth=2.0, label='Pred')
    ax[2].plot(base_y)
    # ax[2].set(xlim=(0,150))
    for i in range(nu):
        ax[3+i].plot(input_traj[:,i], linewidth=4.0)
        ax[3+i].set_xlabel('$time$', fontsize=figsize)
        ax[3+i].set_ylabel('$u$', rotation=0, labelpad=20, fontsize=figsize)
        ax[3+i].tick_params(labelbottom=False, labelsize=figsize)
        # ax[3+i].set(xlim=(0,150))
    ax[-1].tick_params(labelbottom=True, labelsize=figsize)
    plt.tight_layout()
    fig.savefig('model_rollout', dpi=fig.dpi)

    fig = plt.figure(figsize=(15,6))
    plt.plot(true_traj[-3,:], 'c', linewidth=4.0, label="True")
    plt.plot(pred_traj[-3,:], 'm', linewidth=2.0, label='Pred')
    # plt.plot(base_traj[-3,:])
    plt.tight_layout()
    plt.savefig('temp.png', dpi=fig.dpi)

class SSM(nn.Module):
    def __init__(self, A, B, nx, nu):
        super().__init__()
        self.A = nn.Parameter(A, requires_grad=True)
        self.B = nn.Parameter(B, requires_grad=True)
        self.nx = nx
        self.nu = nu
        self.in_features = nx+nu
        self.out_features = nx

    def forward(self, x, u):
        assert len(x.shape) == 2
        assert len(u.shape) == 2

        x = x @ self.A.T + u @ self.B.T

        return x
    
class IC_Adjust(nn.Module):
    def __init__(self, xSize):
        super().__init__()
        params = []
        for i in range(xSize):
            params.append(nn.Parameter(torch.tensor(1.0), requires_grad=True))
        params = torch.tensor(params)
        self.a = torch.diag(params)

    def forward(self, x):
        x_new = x @ self.a
        return x_new

if __name__ == '__main__':
    Main()