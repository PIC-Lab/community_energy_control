import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.loggers import BasicLogger

class BuildingRC():
    def __init__(self, nx, ny, nu, nd, A, B, F, C, params, name):
        '''
        Constructor function
        Inputs:
            nx: (int) number of states
            nu: (int) number of control signals
            nd: (int) number of disturbances
            name: (str) name of the model
        '''
        self.nx = nx
        self.ny = ny
        self.nu = nu
        self.nd = nd

        self.A = A
        self.B = B
        self.F = F
        self.C = C

        train_params = params['train_params']
        self.nsteps = train_params['nsteps']
        self.batch_size = train_params['batch_size']
        self.max_epochs = train_params['max_epochs']
        self.patience = train_params['patience']
        self.warmup = train_params['warmup']
        self.lr = train_params['lr']

        self.saveDir = name

        self.model = None
        self.problem = None
        self.testData = None
        self.loss = None

    def PrepareDataset(self, dataset):
        '''
        Prepare data for training and testing of building thermal model
        Inputs:
            dataset: (dict) normalized dataset split into states (X), inputs (U), and disturbances (D)
        Outputs:
            trainLoader: (DataLoader) DataLoader object for training
            devLoader: (DataLoader) DataLoader object for validation
            testData: (dict) Batched dataset designated for testing
        '''
        n = len(dataset['X'])

        train = {}
        train['X'] = dataset['X'][:int(np.round(n*0.7)),:]
        train['Y'] = dataset['Y'][:int(np.round(n*0.7)),:]
        train['U'] = dataset['U'][:int(np.round(n*0.7)),:]
        train['D'] = dataset['D'][:int(np.round(n*0.7)),:]

        dev = {}
        dev['X'] = dataset['X'][int(np.round(n*0.7)):int(np.round(n*0.9)),:]
        dev['Y'] = dataset['Y'][int(np.round(n*0.7)):int(np.round(n*0.9)),:]
        dev['U'] = dataset['U'][int(np.round(n*0.7)):int(np.round(n*0.9)),:]
        dev['D'] = dataset['D'][int(np.round(n*0.7)):int(np.round(n*0.9)),:]

        test = {}
        test['X'] = dataset['X'][int(np.round(n*0.9)):,:]
        test['Y'] = dataset['Y'][int(np.round(n*0.9)):,:]
        test['U'] = dataset['U'][int(np.round(n*0.9)):,:]
        test['D'] = dataset['D'][int(np.round(n*0.9)):,:]

        nbatch = len(test['X']) // self.nsteps
        print("nbatch:", nbatch)

        trainX = torch.tensor(train['X'].reshape(nbatch*7, self.nsteps, self.nx), dtype=torch.float32)
        trainU = torch.tensor(train['U'].reshape(nbatch*7, self.nsteps, self.nu), dtype=torch.float32)
        trainY = torch.tensor(train['Y'].reshape(nbatch*7, self.nsteps, self.ny), dtype=torch.float32)
        trainD = torch.tensor(train['D'].reshape(nbatch*7, self.nsteps, self.nd), dtype=torch.float32)
        trainData = DictDataset({'xn': trainX[:,0:1,:],
                                'yn': trainY, 'u': trainU, 'd': trainD}, name='train')
        for key, value in trainData.datadict.items():
            print(key, value.shape)
        trainLoader = DataLoader(trainData, batch_size=self.batch_size,
                                collate_fn=trainData.collate_fn, shuffle=True)
        
        devX = torch.tensor(dev['X'].reshape(nbatch*2, self.nsteps, self.nx), dtype=torch.float32)
        devU = torch.tensor(dev['U'].reshape(nbatch*2, self.nsteps, self.nu), dtype=torch.float32)
        devY = torch.tensor(dev['Y'].reshape(nbatch*2, self.nsteps, self.ny), dtype=torch.float32)
        devD = torch.tensor(dev['D'].reshape(nbatch*2, self.nsteps, self.nd), dtype=torch.float32)
        devData = DictDataset({'xn': devX[:,0:1,:],
                            'yn': devY, 'u': devU, 'd': devD}, name='dev')
        devLoader = DataLoader(devData, batch_size=self.batch_size,
                                collate_fn=devData.collate_fn, shuffle=True)
        
        testX = torch.tensor(test['X'].reshape(nbatch, self.nsteps, self.nx), dtype=torch.float32)
        testU = torch.tensor(test['U'].reshape(nbatch, self.nsteps, self.nu), dtype=torch.float32)
        testY = torch.tensor(test['Y'].reshape(nbatch, self.nsteps, self.ny), dtype=torch.float32)
        testD = torch.tensor(test['D'].reshape(nbatch, self.nsteps, self.nd), dtype=torch.float32)
        testData = {'xn': testX[:,0:1,:], 'yn': testY, 'u': testU, 'd': testD}

        return trainLoader, devLoader, testData

    def CreateModel(self):
        '''Defines building thermal system'''

        # self.B = blocks.MLP(self.nu, self.nx, bias=True, linear_map=torch.nn.Linear,
        #                nonlin=torch.nn.ReLU, hsizes=[80,80])

        # xnext = SSM(self.A.clone(), self.B, self.F.clone(), self.nx, self.nu, self.nd, self.device)
        xnext = SSM(self.A, self.B, self.F, self.nx, self.nu, self.nd)
        state_model = Node(xnext, ['xn', 'u', 'd'], ['xn', 'lamb'], name ='base_SSM')

        ynext = lambda x: x @ self.C.T
        output_model = Node(ynext, ['xn'], ['y'], name='out_obs')

        system = System([state_model, output_model], nsteps=self.nsteps, name='RC_Thermal')
        self.model = system

        y = variable('y')
        y_true = variable('yn')

        referenceLoss = 5.*(y_true == y)^2
        referenceLoss.name = 'ref_loss'

        onestepLoss = 1.*(y_true[:,1,:] == y[:,1,:])^2
        onestepLoss.name = 'onestep_loss'

        objectives = [referenceLoss, onestepLoss]
        constraints = []

        loss = PenaltyLoss(objectives, constraints)

        self.problem = Problem([system], loss)

    def TrainModel(self, dataset, trainMode='full'):
        '''
        Trains building thermal system or loads from file
        Inputs:
            dataset: (dict) normalized dataset split into states (X), inputs (U), and disturbances (D)
            load: (bool) load model from file instead of training
        '''
        # All states and input matrices are trainable

        if trainMode.lower() == 'full':
            self.model.nodes[0].callable.A.requires_grad = True
            self.model.nodes[0].callable.B.requires_grad = True
            self.model.nodes[0].callable.F.requires_grad = True
        # Only states and disturbance matrices are trainable
        elif trainMode.lower() == 'natural':
            self.model.nodes[0].callable.A.requires_grad = True
            self.model.nodes[0].callable.B.requires_grad = False
            self.model.nodes[0].callable.F.requires_grad = True
        # Only forced input matrices are trainable
        elif trainMode.lower() == 'forced':
            self.model.nodes[0].callable.A.requires_grad = False
            self.model.nodes[0].callable.B.requires_grad = True
            self.model.nodes[0].callable.F.requires_grad = False
        elif trainMode.lower() == 'none':
            self.model.nodes[0].callable.A.requires_grad = False
            self.model.nodes[0].callable.B.requires_grad = False
            self.model.nodes[0].callable.F.requires_grad = False
        else:
            ValueError('Unexpected training mode for thermal model')

        trainLoader, devLoader, testData = self.PrepareDataset(dataset)
        self.testData = testData

        optimizer = torch.optim.Adam(self.problem.parameters(), lr = self.lr)
        logger = BasicLogger(args=None,
                                savedir=self.saveDir,
                                verbosity=1,
                                stdout=['dev_loss', 'train_loss'])
        
        trainer = Trainer(self.problem,
                            trainLoader,
                            devLoader,
                            testData,
                            optimizer,
                            patience=self.patience,
                            warmup=self.warmup,
                            epochs=self.max_epochs,
                            eval_metric='dev_loss',
                            train_metric='train_loss',
                            dev_metric='dev_loss',
                            test_metric='dev_loss',
                            logger=logger)

        print("----- Training thermal model -----")
        best_model = trainer.train()
        self.problem.load_state_dict(best_model)

    def TestModel(self, label=''):
        '''Plots the testing of the building thermal model'''
        dynamics_model = self.model
        dynamics_model.nsteps = self.testData['yn'].shape[1]

        trajectories = dynamics_model(self.testData)

        pred_traj = trajectories['xn'][:,:-1,:].detach().numpy().reshape(-1,self.nx)
        pred_y = trajectories['y'].detach().numpy().reshape(-1,self.ny)
        true_y = self.testData['yn'].detach().numpy().reshape(-1,self.ny)
        input_traj = self.testData['u'].detach().numpy().reshape(-1,self.nu)
        dist_traj = self.testData['d'].detach().numpy().reshape(-1,self.nd)

        figsize = 25
        lw = 2.0
        fig,ax = plt.subplots(1+self.ny+self.nu+self.nd, figsize=(figsize, figsize))
        for i in range(self.nx):
            ax[0].plot(pred_traj[i,:])
        ax[0].tick_params(labelbottom=False, labelsize=figsize)
        ax[0].set_title("Predicted States", fontsize=figsize)

        ax[1].tick_params(labelbottom=False, labelsize=figsize)
        ax[1].plot(true_y, 'c', linewidth=lw, label="True")
        ax[1].plot(pred_y, 'm', linewidth=lw, label='Pred')
        for i in range(self.nu):
            ax[1+self.ny+i].plot(input_traj[:,i], linewidth=lw)
            ax[1+self.ny+i].set_xlabel('$time$', fontsize=figsize)
            ax[1+self.ny+i].set_ylabel('$u$', rotation=0, labelpad=20, fontsize=figsize)
            ax[1+self.ny+i].tick_params(labelbottom=False, labelsize=figsize)
        for i in range(self.nd):
            ax[1+self.ny+self.nu+i].plot(dist_traj[:,i], linewidth=lw)
            ax[1+self.ny+self.nu+i].set_xlabel('$time$', fontsize=figsize)
            ax[1+self.ny+self.nu+i].set_ylabel('$d$', rotation=0, labelpad=20, fontsize=figsize)
            ax[1+self.ny+self.nu+i].tick_params(labelbottom=False, labelsize=figsize)
        ax[-1].tick_params(labelbottom=True, labelsize=figsize)
        plt.tight_layout()
        fig.savefig(self.saveDir+'/model_rollout'+label, dpi=fig.dpi)

        figsize = 6
        lw = 2.0
        fig = plt.figure(figsize=(figsize,4))
        plt.plot(true_y, 'c', linewidth=lw, label="True")
        plt.plot(pred_y, 'm', linewidth=lw, label='Pred')
        plt.xlabel('Time [mins]')
        plt.ylabel("Temperature [C]")
        plt.legend()
        plt.tight_layout()
        fig.savefig(self.saveDir+'/justTemp'+label, dpi=fig.dpi)
        plt.close(fig)

class ControllerSystem():
    def __init__(self, nx, nu, nd, nd_obs, ny, y_idx, d_idx, params, name, thermalModel):
        '''
        Constructor function
        Inputs:
            nx: (int) number of states
            nu: (int) number of control signals
            nd: (int) number of disturbances
            nd_obs: (int) number of observable disturbances
            ni: (int) number of information inputs
            ny: (int) number of controlled outputs
            y_idx: (list[int]) indices of observed states
            d_idx: (list[int]) indices of observable disturbances
            name: (str) name of the model
            thermalModel: (neuromancer.system or Node) trained building thermal model
        '''
        self.nx = nx
        self.nu = nu
        self.nd = nd
        self.nd_obs = nd_obs
        self.ny = ny
        self.nref = ny
        self.y_idx = y_idx
        self.d_idx = d_idx

        self.weights = params['weights']
        self.hsizes = params['hsizes']

        train_params = params['train_params']
        self.nsteps = train_params['nsteps']
        self.n_samples = train_params['n_samples']
        self.batch_size = train_params['batch_size']
        self.max_epochs = train_params['max_epochs']
        self.patience = train_params['patience']
        self.warmup = train_params['warmup']
        self.lr = train_params['lr']

        self.thermalModel = thermalModel

        self.saveDir = name

        self.rng = np.random.default_rng(seed=60)

        self.system = None
        self.problem = None
        self.loss = None

    def CreateModel(self):
        '''Defines control system'''
        net = blocks.MLP_bounds(
        insize=self.ny + 2*self.nref + self.nd_obs,
        outsize=self.nu,
        hsizes=self.hsizes,
        nonlin=nn.GELU,
        min=0,
        max=1)

        policy = Node(net, ['y', 'ymin', 'ymax', 'd_obs'], ['u'], name='policy')

        # Freeze thermal model
        for p in self.thermalModel.nodes[0].parameters():
            p.requires_grad=False

        dist_model = lambda d: d[:, self.d_idx]
        dist_obs = Node(dist_model, ['d'], ['d_obs'], name='dist_obs')

        # closed loop system model
        self.system = System([dist_obs, policy, self.thermalModel.nodes[0], self.thermalModel.nodes[1]],
                        nsteps=self.nsteps,
                        name='cl_system')
        
        # DPC objectives and constraints
        # variables
        y = variable('y')
        u = variable('u')
        ymin = variable('ymin')
        ymax = variable('ymax')

        # objectives
        action_loss = u.minimize(weight=self.weights['action_loss'], name='action_loss')
        du_loss = self.weights['du_loss'] * (u[:,:-1,:] - u[:,1:,:] == torch.tensor(np.zeros((self.batch_size, self.nsteps-1, self.nu)))) # delta u minimization

        # thermal comfort constraints
        state_lower_bound_penalty = self.weights['x_min'] * (y > ymin)
        state_upper_bound_penalty = self.weights['x_max'] * (y < ymax)

        # objectives and constraints names for nicer plot
        state_lower_bound_penalty.name = 'x_min'
        state_upper_bound_penalty.name = 'x_max'

        # list of constraints and objectives
        objectives = [action_loss, du_loss]
        constraints = [state_lower_bound_penalty, state_upper_bound_penalty]

        # Problem construction
        nodes = [self.system]
        loss = PenaltyLoss(objectives, constraints)
        self.problem = Problem(nodes, loss)

    def TrainModel(self, dataset, tempMin, tempMax):
        '''
        Trains control system
        Inputs:
            dataset: (dict) normalized dataset split into states (X), information inputs (I), and disturbances (D)
            tempMin: (float)
            tempMax: (float)
        '''
        xmin_range = torch.distributions.Uniform(tempMin, tempMax)

        trainLoader, devLoader = [
            self.PrepareDataset(dataset, xmin_range, 
                            name=name) for name in ("train", "dev")
        ]

        logger = BasicLogger(args=None, savedir=self.saveDir, verbosity=1,
                         stdout=['dev_loss', 'train_loss'])

        optimizer = torch.optim.AdamW(self.problem.parameters(), lr=self.lr)
        trainer = Trainer(
            self.problem, trainLoader,
            devLoader,
            optimizer=optimizer,
            epochs=self.max_epochs,
            train_metric='train_loss',
            eval_metric='dev_loss',
            logger=logger,
            patience=self.patience,
            warmup=self.warmup
        )

        print("----- Training controller -----")
        best_model = trainer.train()
        trainer.model.load_state_dict(best_model)
        
    def TestModel(self, dataset, tempMin):
        '''Plots the testing of the controller system'''
        nsteps_test = 2000

        # generate reference
        np_refs = (tempMin + 4.0) * np.ones(nsteps_test+1)
        ymin_val = torch.tensor(np_refs, dtype=torch.float32).reshape(1, nsteps_test+1, 1)
        ymax_val = ymin_val + 2.0
        # get disturbance signal
        start_idx = self.rng.integers(0, len(dataset['D'])-nsteps_test)
        torch_dist = torch.tensor(self.Get_D(dataset['D'], nsteps_test+1, start_idx),
                                  dtype=torch.float32).unsqueeze(0)
        # initial data for closed loop sim
        x0 = torch.tensor(self.Get_X0(dataset['X']),
                          dtype=torch.float32).reshape(1,1,self.nx)
        data = {'xn': x0,
                'y': x0[:,:,self.y_idx],
                'ymin': ymin_val,
                'ymax': ymax_val,
                'd': torch_dist}

        self.system.nsteps = nsteps_test
        trajectories = self.system(data)

        # constraints bounds
        Ymin = trajectories['ymin'].reshape(nsteps_test+1, self.nref)
        Ymax = trajectories['ymax'].reshape(nsteps_test+1, self.nref)

        numPlots = 3
        fig, ax = plt.subplots(numPlots, figsize=(20,16))
        ax[0].plot(trajectories['y'].detach().reshape(nsteps_test+1, self.ny), linewidth=3)
        ax[0].plot(Ymin, '--', linewidth=3, c='k')
        ax[0].plot(Ymax, '--', linewidth=3, c='k')
        ax[0].set_ylabel('y', fontsize=26)
        # ax[0].set(ylim=[10,30])
        ax[1].plot(trajectories['u'].detach().reshape(nsteps_test, self.nu), linewidth=3)
        ax[1].set_ylabel('u', fontsize=26)
        ax[1].set(ylim=(-0.1,1.1))
        ax[2].plot(trajectories['d'].detach().reshape(nsteps_test+1, self.nd), linewidth=3)
        ax[2].set_ylabel('d', fontsize=26)
        ax[2].set_xlabel('Time [mins]', fontsize=26)
        for i in range(numPlots):
            ax[i].grid(True)
            ax[i].tick_params(axis='x', labelsize=26)
            ax[i].tick_params(axis='y', labelsize=26)
            ax[i].set_xlim(0, nsteps_test)
        plt.tight_layout()
        plt.savefig(self.saveDir+'/controller_rollout', dpi=fig.dpi)
        plt.close(fig)

        numPlots = 2
        lw = 3
        fs = 26
        fig, ax = plt.subplots(numPlots, figsize=(10,8))
        ax[0].plot(trajectories['y'].detach().reshape(nsteps_test+1, self.ny), linewidth=lw)
        ax[0].plot(Ymin, '--', linewidth=lw, c='k')
        ax[0].plot(Ymax, '--', linewidth=lw, c='k')
        ax[0].set_ylabel('y', fontsize=fs)
        ax[1].plot(trajectories['u'].detach().reshape(nsteps_test, self.nu), linewidth=lw)
        ax[1].set_ylabel('u', fontsize=fs)
        ax[1].set(ylim=(-0.1,1.1))
        for i in range(numPlots):
            ax[i].grid(True)
            ax[i].tick_params(axis='x', labelsize=26)
            ax[i].tick_params(axis='y', labelsize=26)
            ax[i].set_xlim(0, nsteps_test)
        plt.tight_layout()
        plt.savefig(self.saveDir+'/simplified_rollout', dpi=fig.dpi)
        plt.close(fig)

    def PrepareDataset(self, dataset, xmin_range, name='train'):
        '''
        Prepare data for training and testing of controller model
        Inputs:
            dataset: (dict) normalized dataset split into states (X), information inputs (I), and disturbances (D)
        Outputs:
            (DataLoader) DataLoader object with batched dataset
        '''
        # sampled references for training the policy
        batched_xmin = xmin_range.sample((self.n_samples, 1, self.nref)).repeat(1, self.nsteps+1, 1)
        batched_range = torch.tensor(self.rng.uniform(low=1.0, high = 8.0, size=(self.n_samples, 1, self.nref)), dtype=torch.float32).repeat(1, self.nsteps+1, 1)
        batched_xmax = batched_xmin + batched_range

        # sampled disturbance trajectories from simulation model
        temp_d = []
        for _ in range(self.n_samples):
            start_idx = self.rng.integers(0, len(dataset['D'])-1-self.nsteps)
            temp_d.append(torch.tensor(self.Get_D(dataset['D'], self.nsteps, start_idx),
                                      dtype=torch.float32))
        batched_dist = torch.stack(temp_d)

        # sampled initial conditions
        batched_x0 = torch.stack([torch.tensor(self.Get_X0(dataset['X']),
                                               dtype=torch.float32).unsqueeze(0)
                                               for _ in range(self.n_samples)])

        data = DictDataset(
            {"xn": batched_x0,
            "y": batched_x0[:,:,self.y_idx],
            "ymin": batched_xmin,
            "ymax": batched_xmax,
            "d": batched_dist},
            name=name,
        )
        for key, value in data.datadict.items():
            print(key, value.shape)

        return DataLoader(data, batch_size=self.batch_size, collate_fn=data.collate_fn, shuffle=False)

    def Get_X0(self, data):
        '''
        Randomly samples state data to create a series of initial states
        Inputs:
            dat: (ndarray/tensor) state data
        Outputs:
            list of initial states
        '''
        # brackets because there is only one state currently
        return np.array(self.rng.uniform(low=np.min(data, axis=0), high=np.max(data, axis=0)))

    def Get_D(self, data, nsim, start_idx):
        '''
        Samples a slice of disturbance data
        Inputs:
            data: (ndarray/tensor) disturbance data
            nsim: (int) length of slice to sample
            start_idx: (int) index to start slice at
        Outputs:
            Selected slice of disturbance data
        '''
        return data[start_idx:start_idx+nsim, :]
    
class SSM(nn.Module):
    def __init__(self, A, B, F, nx, nu, nd):
        super().__init__()
        self.A = nn.Parameter(A, requires_grad=True)
        self.B = nn.Parameter(B, requires_grad=True)
        self.F = nn.Parameter(F, requires_grad=True)
        self.in_features = nx + nu + nd
        self.out_features = nx

    def forward(self, x, u, d):
        assert len(x.shape) == 2, x.shape
        assert len(u.shape) == 2
        assert len(d.shape) == 2

        # x = x @ self.A.T + self.B(u) + d @ self.F.T
        x = x @ self.A.T + self.B.T + d @ self.F.T

        return (x)