import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import copy
from pathlib import Path
from enum import Enum
import numbers
import imageio
import os
import shutil

import neuromancer.psl as psl
from neuromancer.system import Node, System
from neuromancer.modules import blocks, solvers
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.loss import BarrierLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.dynamics import integrators
from neuromancer.loggers import BasicLogger
from neuromancer.callbacks import Callback

class BuildingRC():
    def __init__(self, nx, ny, nu, nd, A, B, F, C, manager, name, device, debugLevel, saveDir=None):
        '''
        Constructor function
        Inputs:
            nx: (int) number of states
            nu: (int) number of control signals
            nd: (int) number of disturbances
            manager (RunManager) object that keeps track of various model parameters
            name: (str) name of the model
            device: (torch.device) device to run training on
            debugLevel: (int or DebugLevel) sets the level of debug detail outputted from training
            saveDir: (str) relative path to where the model should be saved/loaded from, defaults to value of 'name'
        '''
        self.nx = nx
        self.ny = ny
        self.nu = nu
        self.nd = nd

        self.A = A.to(device=device)
        self.B = B.to(device=device)
        self.F = F.to(device=device)
        self.C = C.to(device=device)

        self.manager = manager

        train_params = manager.models[name]['train_params']
        self.nsteps = train_params['nsteps']
        self.batch_size = train_params['batch_size']
        self.max_epochs = train_params['max_epochs']
        self.patience = train_params['patience']
        self.warmup = train_params['warmup']
        self.lr = train_params['lr']
        self.device = device

        self.callback = Callback_NODE(debugLevel)

        if saveDir is None:
            self.saveDir = name
        else:
            self.saveDir = saveDir

        self.runName = manager.name

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
        
        testX = torch.tensor(test['X'].reshape(nbatch, self.nsteps, self.nx), dtype=torch.float32, device=self.device)
        testU = torch.tensor(test['U'].reshape(nbatch, self.nsteps, self.nu), dtype=torch.float32, device=self.device)
        testY = torch.tensor(test['Y'].reshape(nbatch, self.nsteps, self.ny), dtype=torch.float32, device=self.device)
        testD = torch.tensor(test['D'].reshape(nbatch, self.nsteps, self.nd), dtype=torch.float32, device=self.device)
        testData = {'xn': testX[:,0:1,:], 'yn': testY, 'u': testU, 'd': testD}

        return trainLoader, devLoader, testData

    def CreateModel(self):
        '''Defines building thermal system'''
        # base_ynext = lambda x: x @ self.C.T
        # base_output_model = Node(base_ynext, ['xn'], ['yn'], name='base_out_obs')

        # base_xnext = lambda x, u: x @ A.T + u @ B.T
        # base_ssm = Node(base_xnext, ['xn', 'u'], ['xn'])
        # baseSystem = System([base_ssm, base_output_model], nsteps=nsteps)

        # initialAdj = Node(IC_Adjust(nx), ['xn_0'], ['xn'], name='initialCond')
        # self.A = blocks.MLP(self.nx, self.nx, bias=True, linear_map=torch.nn.Linear,
        #                nonlin=torch.nn.ReLU, hsizes=[40,40])
        self.B = blocks.MLP(self.nu, self.nx, bias=True, linear_map=torch.nn.Linear,
                       nonlin=torch.nn.ReLU, hsizes=[80,80])
        # self.F = blocks.MLP(self.nd, self.nx, bias=True, linear_map=torch.nn.Linear,
        #                nonlin=torch.nn.ReLU, hsizes=[40,40])

        xnext = SSM(self.A.clone(), self.B, self.F.clone(), self.nx, self.nu, self.nd, self.device)
        # xnext = SSM(self.A, self.B, self.F, self.nx, self.nu, self.nd)
        state_model = Node(xnext, ['xn', 'u', 'd'], ['xn', 'lamb'], name ='base_SSM')

        ynext = lambda x: x @ self.C.T
        output_model = Node(ynext, ['xn'], ['y'], name='out_obs')

        system = System([state_model, output_model], nsteps=self.nsteps, name='RC_Thermal')
        system.to(device=self.device)
        self.model = system
        if self.callback.debugLevel > DebugLevel.NO:
            system.show(self.manager.runPath+'thermal_model.png')

        y = variable('y')
        y_true = variable('yn')
        lamb = variable('lamb')

        referenceLoss = 5.*(y_true == y)^2
        referenceLoss.name = 'ref_loss'

        onestepLoss = 1.*(y_true[:,1,:] == y[:,1,:])^2
        onestepLoss.name = 'onestep_loss'

        stableLoss = 10.*(lamb == 0)

        objectives = [referenceLoss, onestepLoss]
        constraints = []

        loss = PenaltyLoss(objectives, constraints)

        self.problem = Problem([system], loss)
        if self.callback.debugLevel > DebugLevel.NO:
            self.problem.show(self.manager.runPath+'thermal_optim.png')

    def TrainModel(self, dataset, load, trainMode='full', test=True):
        '''
        Trains building thermal system or loads from file
        Inputs:
            dataset: (dict) normalized dataset split into states (X), inputs (U), and disturbances (D)
            load: (bool) load model from file instead of training
        '''

        # # All states and input matrices are trainable
        # if trainMode.lower() == 'full':
        #     self.A.requires_grad = True
        #     self.B.requires_grad = True
        #     self.F.requires_grad = True
        # # Only states and disturbance matrices are trainable
        # elif trainMode.lower() == 'natural':
        #     self.A.requires_grad = True
        #     self.B.requires_grad = False
        #     self.F.requires_grad = True
        # # Only forced input matrices are trainable
        # elif trainMode.lower() == 'forced':
        #     self.A.requires_grad = False
        #     self.B.requires_grad = True
        #     self.F.requires_grad = False
        # elif trainMode.lower() == 'none':
        #     self.A.requires_grad = False
        #     self.B.requires_grad = False
        #     self.F.requires_grad = False
        # else:
        #     ValueError('Unexpected training mode for thermal model')

        # All states and input matrices are trainable
        exit = False
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
            test = True
            exit = True
        else:
            ValueError('Unexpected training mode for thermal model')

        if test:
            trainLoader, devLoader, testData = self.PrepareDataset(dataset)
            self.testData = testData

        # Exit without doing any training or loading
        if exit:
            print('Not training thermal model')
            os.makedirs(self.saveDir, exist_ok=True)
            return

        # If load is true, skip training and just load from state dict file
        if load:
            self.problem.load_state_dict(torch.load(self.saveDir+'/best_model_state_dict.pth'))
            self.problem.eval()
            return

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
                            logger=logger,
                            device=self.device,
                            callback=self.callback)

        print("----- Training thermal model -----")
        best_model = trainer.train()
        self.problem.load_state_dict(best_model)

        if self.callback.debugLevel > DebugLevel.NO:
            A_np = self.model.nodes[0].callable.A.detach().cpu().numpy()
            A_df = pd.DataFrame(A_np)
            A_df.to_csv(self.saveDir+f"/A_{trainMode}.csv", index=False, header=False)

            # B_np = self.model.nodes[0].callable.B.detach().cpu().numpy()
            # F_np = self.model.nodes[0].callable.F.detach().cpu().numpy()
            # B_np = np.concatenate((B_np, F_np), axis=1)
            # B_df = pd.DataFrame(B_np)
            # B_df.to_csv(self.saveDir+f"/B_{trainMode}.csv", index=False, header=False)


    def TestModel(self, label=''):
        '''Plots the testing of the building thermal model'''
        dynamics_model = self.model
        dynamics_model.nsteps = self.testData['yn'].shape[1]

        trajectories = dynamics_model(self.testData)

        pred_traj = trajectories['xn'][:,:-1,:].detach().cpu().numpy().reshape(-1,self.nx)
        pred_y = trajectories['y'].detach().cpu().numpy().reshape(-1,self.ny)
        true_y = self.testData['yn'].detach().cpu().numpy().reshape(-1,self.ny)
        input_traj = self.testData['u'].detach().cpu().numpy().reshape(-1,self.nu)
        dist_traj = self.testData['d'].detach().cpu().numpy().reshape(-1,self.nd)

        testMetrics = pd.DataFrame()
        test_mae = np.mean(np.abs(pred_y - true_y))
        testMetrics['mae'] = [test_mae]
        testMetrics.to_csv(self.saveDir+'/test_metrics.csv', index=False)

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
        # ax[2].set(xlim=(0,150))
        for i in range(self.nu):
            ax[1+self.ny+i].plot(input_traj[:,i], linewidth=lw)
            ax[1+self.ny+i].set_xlabel('$time$', fontsize=figsize)
            ax[1+self.ny+i].set_ylabel('$u$', rotation=0, labelpad=20, fontsize=figsize)
            ax[1+self.ny+i].tick_params(labelbottom=False, labelsize=figsize)
            # ax[3+i].set(xlim=(0,150))
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

        # Plot training and validation loss
        # Should only run if the model was trained on this run instead of loaded from a file
        if (self.callback.debugLevel >= DebugLevel.EPOCH_LOSS) and (len(self.callback.loss) > 0):
            loss_df = pd.DataFrame(self.callback.loss)
            loss_df = loss_df[['train_loss', 'dev_loss']]
            loss_df['train_loss'] = loss_df['train_loss'].apply(lambda x: x.detach().item())
            loss_df['dev_loss'] = loss_df['dev_loss'].apply(lambda x: x.detach().item())
            loss_df.to_csv(self.saveDir+'/loss.csv', index=False)
            fig = plt.figure()
            plt.plot(loss_df)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(loss_df.keys())
            plt.tight_layout()
            fig.savefig(self.saveDir+'/building_loss'+label, dpi=fig.dpi)
            plt.close(fig)

class ModeClassifier():
    def __init__(self, nm, nu,  manager, name, device, debugLevel, saveDir=None):
        '''
        Constructor function
        Inputs:
            nm: (int) number of hvac modes
            nu: (int) number of control signals
            manager (RunManager) object that keeps track of various model parameters
            name: (str) name of the model
            device: (torch.device) device to run training on
            debugLevel: (int or DebugLevel) sets the level of debug detail outputted from training
            saveDir: (str) relative path to where the model should be saved/loaded from, defaults to value of 'name'
        '''
        self.nm = nm
        self.nu = nu
        
        self.manager = manager

        self.hsizes = manager.models[name]['hsizes']

        train_params = manager.models[name]['train_params']
        self.nsteps = train_params['nsteps']
        self.batch_size = train_params['batch_size']
        self.max_epochs = train_params['max_epochs']
        self.patience = train_params['patience']
        self.warmup = train_params['warmup']
        self.lr = train_params['lr']
        self.device = device

        self.callback = Callback_NODE(debugLevel)

        if saveDir is None:
            self.saveDir = name
        else:
            self.saveDir = saveDir

        self.runName = manager.name

        self.model = None
        self.problem = None
        self.testData = None
        self.loss = None

    def PrepareDataset(self, dataset):
        n = len(dataset['M'])

        train = {}
        train['M'] = dataset['M'][:int(np.round(n*0.7)),:]
        train['U'] = dataset['U'][:int(np.round(n*0.7)),:]

        dev = {}
        dev['M'] = dataset['M'][int(np.round(n*0.7)):int(np.round(n*0.9)),:]
        dev['U'] = dataset['U'][int(np.round(n*0.7)):int(np.round(n*0.9)),:]

        test = {}
        test['M'] = dataset['M'][int(np.round(n*0.9)):,:]
        test['U'] = dataset['U'][int(np.round(n*0.9)):,:]

        nbatch = len(test['M']) // self.nsteps

        trainM = train['M'].reshape(nbatch*7, self.nsteps, self.nm)
        trainM = torch.tensor(trainM, dtype=torch.float32, device=self.device)
        trainU = train['U'].reshape(nbatch*7, self.nsteps, self.nu)
        trainU = torch.tensor(trainU, dtype=torch.float32, device=self.device)
        trainData = DictDataset({'m': trainM, 'u_dec': trainU}, name='train')
        trainLoader = DataLoader(trainData, batch_size=self.batch_size,
                                collate_fn=trainData.collate_fn, shuffle=True)
        
        devM = dev['M'].reshape(nbatch*2, self.nsteps, self.nm)
        devM = torch.tensor(devM, dtype=torch.float32, device=self.device)
        devU = dev['U'].reshape(nbatch*2, self.nsteps, self.nu)
        devU = torch.tensor(devU, dtype=torch.float32, device=self.device)
        devData = DictDataset({'m': devM, 'u_dec': devU}, name='dev')
        devLoader = DataLoader(devData, batch_size=self.batch_size,
                                collate_fn=devData.collate_fn, shuffle=True)
        
        testM = test['M'].reshape(nbatch, self.nsteps, self.nm)
        testM = torch.tensor(testM, dtype=torch.float32, device=self.device)
        testU = test['U'].reshape(nbatch, self.nsteps, self.nu)
        testU = torch.tensor(testU, dtype=torch.float32, device=self.device)
        testData = {'m': testM, 'u_dec': testU}

        return trainLoader, devLoader, testData

    def CreateModel(self):
        '''Defines HVAC mode classifier'''
        net = blocks.MLP_bounds(
            insize=self.nu,
            outsize=self.nm,
            hsizes=self.hsizes,
            nonlin=nn.Sigmoid
        )
        self.model = Node(net, ['u_dec'], ['u'], name='Classifier')

        system = System([self.model], name='System', nsteps=self.nsteps)
        system.to(device=self.device)

        m = variable("m")
        mhat = variable('u')

        referenceLoss = 5.*(mhat == m)^2
        referenceLoss.name = 'ref_loss'

        onestepLoss = 1.*(mhat[:,1,:] == m[:,1,:])^2
        onestepLoss.name = 'onestep_loss'

        objectives = [referenceLoss, onestepLoss]
        constraints = []

        loss = PenaltyLoss(objectives, constraints)

        self.problem = Problem([system], loss)

    def TrainModel(self, dataset, load, test=True):

        if test:
            trainLoader, devLoader, testData = self.PrepareDataset(dataset)
            self.testData = testData

        # If load is true, skip training and just load from state dict file
        if load:
            self.problem.load_state_dict(torch.load(self.saveDir+'/best_model_state_dict.pth'))
            self.problem.eval()
            return

        optimizer = torch.optim.Adam(self.problem.parameters(), lr = self.lr)
        logger = BasicLogger(args=None, savedir=self.saveDir, verbosity=1,
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
                          logger=logger,
                          callback=self.callback,
                          device=self.device
        )

        print("----- Training classifier -----")
        best_model = trainer.train()
        self.problem.load_state_dict(best_model)

    def TestModel(self):
        dynamics_model = self.problem.nodes[0]
        dynamics_model.nsteps = self.testData['m'].shape[1]

        testOutputs = dynamics_model(self.testData)

        pred_traj = testOutputs['u'].detach().cpu().numpy().reshape(-1,self.nm)
        true_traj = self.testData['m'].detach().cpu().numpy().reshape(-1,self.nm)
        input_traj = self.testData['u_dec'].detach().cpu().numpy().reshape(-1,self.nu)
        pred_traj, true_traj = pred_traj.transpose(1,0), true_traj.transpose(1,0)

        testMetrics = pd.DataFrame()
        test_mae = np.mean(np.abs(pred_traj - true_traj))
        testMetrics['mae'] = [test_mae]
        testMetrics.to_csv(self.saveDir+'/test_metrics.csv', index=False)

        figsize = 25
        lw = 4.0
        fig,ax = plt.subplots(self.nm+self.nu, figsize=(figsize, figsize))
        labels = [f'$y_{k}$' for k in range(len(true_traj))]
        for row, (t1, t2, label) in enumerate(zip(true_traj, pred_traj, labels)):
            axe = ax[row]
            axe.set_ylabel(label, rotation=0, labelpad=20, fontsize=figsize)
            axe.plot(t1, 'c', linewidth=lw, label="True")
            axe.plot(t2, 'm--', linewidth=lw, label='Pred')
            axe.tick_params(labelbottom=False, labelsize=figsize)
            axe.set_title("Class (Normalized)", fontsize=figsize)
            axe.set_xlim(0,1000)
        axe.tick_params(labelbottom=True, labelsize=figsize)
        axe.legend(fontsize=figsize)
        ax[-1].plot(input_traj, linewidth=lw)
        ax[-1].set_xlabel('$time$', fontsize=figsize)
        ax[-1].set_ylabel('$u$', rotation=0, labelpad=20, fontsize=figsize)
        ax[-1].tick_params(labelbottom=True, labelsize=figsize)
        ax[-1].set_title("HVAC Consumption (Normalized)", fontsize=figsize)
        ax[-1].set_xlim(0,1000)
        plt.tight_layout()
        fig.savefig(self.saveDir+'/rollout', dpi=fig.dpi)
        plt.close(fig)

        figsize = 6
        lw = 2.0
        fig,ax = plt.subplots(self.nm, figsize=(figsize, 4))
        labels = [f'$y_{k}$' for k in range(len(true_traj))]
        for row, (t1, t2, label) in enumerate(zip(true_traj, pred_traj, labels)):
            ax.set_ylabel(label, rotation=0, labelpad=20, fontsize=figsize)
            ax.plot(t1, 'c', linewidth=lw, label="True")
            ax.plot(t2, 'm--', linewidth=lw, label='Pred')
            ax.tick_params(labelbottom=False, labelsize=figsize)
            ax.set_title("Class (Normalized)", fontsize=figsize)
            ax.set_xlim(0,1000)
        ax.tick_params(labelbottom=True, labelsize=figsize)
        ax.legend(fontsize=figsize)
        plt.tight_layout()
        fig.savefig(self.saveDir+'/mode', dpi=fig.dpi)
        plt.close(fig)

        # Plot training and validation loss
        # Should only run if the model was trained on this run instead of loaded from a file
        if (self.callback.debugLevel >= DebugLevel.EPOCH_LOSS) and (len(self.callback.loss) > 0):
            loss_df = pd.DataFrame(self.callback.loss)
            loss_df = loss_df[['train_loss', 'dev_loss']]
            loss_df['train_loss'] = loss_df['train_loss'].apply(lambda x: x.detach().item())
            loss_df['dev_loss'] = loss_df['dev_loss'].apply(lambda x: x.detach().item())
            loss_df.to_csv(self.saveDir+'/loss.csv', index=False)
            fig = plt.figure()
            plt.plot(loss_df)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(loss_df.keys())
            plt.tight_layout()
            fig.savefig(self.saveDir+'/loss', dpi=fig.dpi)
            plt.close(fig)

class ControllerSystem():
    def __init__(self, nx, nu, nd, nd_obs, ny, y_idx, d_idx, manager, name,
                 thermalModel, classifier, device, debugLevel, saveDir=None):
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
            manager (RunManager) object that keeps track of various model parameters
            name: (str) name of the model
            norm: (Normalizer) object used to normalize and denormalize data for plotting purposes
            thermalModel: (neuromancer.system or Node) trained building thermal model
            device: (torch.device) device to run training on
            debugLevel: (int or DebugLevel) sets the level of debug detail outputted from training
            saveDir: (str) relative path to where the model should be saved/loaded from, defaults to value of 'name'
        '''
        self.nx = nx
        self.nu = nu
        self.nd = nd
        self.nd_obs = nd_obs
        self.ny = ny
        self.nref = ny
        self.y_idx = y_idx
        self.d_idx = d_idx

        self.manager = manager

        self.weights = manager.models[name]['weights']
        self.hsizes = manager.models[name]['hsizes']

        train_params = manager.models[name]['train_params']
        self.nsteps = train_params['nsteps']
        self.n_samples = train_params['n_samples']
        self.batch_size = train_params['batch_size']
        self.max_epochs = train_params['max_epochs']
        self.patience = train_params['patience']
        self.warmup = train_params['warmup']
        self.lr = train_params['lr']

        self.thermalModel = thermalModel
        self.classifier = classifier
        self.device = device

        if saveDir is None:
            self.saveDir = name
        else:
            self.saveDir = saveDir

        self.runName = manager.name

        self.callback = Callback_Controller(debugLevel, self.saveDir)

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
        for p in self.classifier.parameters():
            p.requires_grad=False

        # convert = lambda u: my_round_func.apply(u)
        # converter = Node(convert, ['u_dec'], ['u'], name='round')

        dist_model = lambda d: d[:, self.d_idx]
        dist_obs = Node(dist_model, ['d'], ['d_obs'], name='dist_obs')

        # closed loop system model
        self.system = System([dist_obs, policy, self.thermalModel.nodes[0], self.thermalModel.nodes[1]],
                        nsteps=self.nsteps,
                        name='cl_system')
        self.system.to(device=self.device)
        if self.callback.debugLevel > DebugLevel.NO:
            self.system.show(self.manager.runPath+"clSystemDigram.png")
        
        # DPC objectives and constraints
        # variables
        y = variable('y')
        u = variable('u')
        ymin = variable('ymin')
        ymax = variable('ymax')

        # objectives
        action_loss = u.minimize(weight=self.weights['action_loss'], name='action_loss')
        du_loss = self.weights['du_loss'] * (u[:,:-1,:] - u[:,1:,:] == torch.tensor(np.zeros((self.batch_size, self.nsteps-1, self.nu)), device=self.device)) # delta u minimization

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
        if self.callback.debugLevel > DebugLevel.NO:
            self.problem.show(self.manager.runPath+"MPC_optim.png")

    def TrainModel(self, dataset, tempMin, tempMax, load, test=True):
        '''
        Trains control system
        Inputs:
            dataset: (dict) normalized dataset split into states (X), information inputs (I), and disturbances (D)
            tempMin: (float)
            tempMax: (float)
        '''
        if test:
            xmin_range = torch.distributions.Uniform(tempMin, tempMax)

            trainLoader, devLoader = [
                self.PrepareDataset(dataset, xmin_range, 
                                name=name) for name in ("train", "dev")
            ]

            for key, value in trainLoader.dataset.datadict.items():
                print(key, value.shape)

        if load:
            self.problem.load_state_dict(torch.load(self.saveDir+'/best_model_state_dict.pth'))
            self.problem.eval()
            return

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
            warmup=self.warmup,
            device=self.device,
            callback=self.callback
        )

        print("----- Training controller -----")
        best_model = trainer.train()
        trainer.model.load_state_dict(best_model)
        
    def TestModel(self, dataset, tempMin, tempMax):
        '''Plots the testing of the controller system'''
        nsteps_test = 2000
        # nsteps_test = 60

        # generate reference
        np_refs = (tempMin.cpu() + 4.0) * np.ones(nsteps_test+1)
        # np_refs = psl.signals.step(nsteps_test+1, 1, min=tempMin, max=tempMax, randsteps=5, rng=self.rng)
        ymin_val = torch.tensor(np_refs, dtype=torch.float32, device=self.device).reshape(1, nsteps_test+1, 1)
        ymax_val = ymin_val + 2.0
        # get disturbance signal
        start_idx = self.rng.integers(0, len(dataset['D'])-nsteps_test)
        torch_dist = torch.tensor(self.Get_D(dataset['D'], nsteps_test+1, start_idx),
                                  dtype=torch.float32, device=self.device).unsqueeze(0)
        # initial data for closed loop sim
        x0 = torch.tensor(self.Get_X0(dataset['X']),
                          dtype=torch.float32, device=self.device).reshape(1,1,self.nx)
        data = {'xn': x0,
                'y': x0[:,:,self.y_idx],
                'ymin': ymin_val,
                'ymax': ymax_val,
                'd': torch_dist}
        print('Input dataset')
        for key, value in data.items():
            print(key, value.shape)
        self.system.nsteps = nsteps_test
        trajectories = self.system(data)
        print('Output trajectories')
        for key, value in trajectories.items():
            print(key, value.shape)
        print(trajectories.keys())

        traj_df = pd.DataFrame()
        for key, value in trajectories.items():
            print(key)
            if value.shape[2] == 1:
                traj_df[key] = value.detach().cpu().numpy().flatten()[:nsteps_test]
            else:
                for i in range(value.shape[2]):
                    traj_df[key+f"_{i}"] = value.detach().cpu().numpy()[:,:,i].flatten()[:nsteps_test]
        if not(os.path.exists('Saved Figures')):
            os.makedirs('Saved Figures')
        traj_df.to_csv('Saved Figures/trajectories.csv', index=False)

        # constraints bounds
        Ymin = trajectories['ymin'].detach().cpu().reshape(nsteps_test+1, self.nref)
        Ymax = trajectories['ymax'].detach().cpu().reshape(nsteps_test+1, self.nref)

        numPlots = 3
        fig, ax = plt.subplots(numPlots, figsize=(20,16))
        ax[0].plot(trajectories['y'].detach().cpu().reshape(nsteps_test+1, self.ny), linewidth=3)
        ax[0].plot(Ymin, '--', linewidth=3, c='k')
        ax[0].plot(Ymax, '--', linewidth=3, c='k')
        ax[0].set_ylabel('y', fontsize=26)
        # ax[0].set(ylim=[10,30])
        ax[1].plot(trajectories['u'].detach().cpu().reshape(nsteps_test, self.nu), linewidth=3)
        ax[1].set_ylabel('u', fontsize=26)
        ax[1].set(ylim=(-0.1,1.1))
        ax[2].plot(trajectories['d'].detach().cpu().reshape(nsteps_test+1, self.nd), linewidth=3)
        ax[2].set_ylabel('d', fontsize=26)
        ax[2].set_xlabel('Time [mins]', fontsize=26)
        for i in range(numPlots):
            ax[i].grid(True)
            ax[i].tick_params(axis='x', labelsize=26)
            ax[i].tick_params(axis='y', labelsize=26)
            ax[i].set_xlim(0, nsteps_test)
            # ax[i].set(xlim=[750, 1500])
        plt.figtext(0.01, 0.01, self.runName, fontsize=26)
        plt.tight_layout()
        plt.savefig(self.saveDir+'/controller_rollout', dpi=fig.dpi)
        plt.close(fig)

        numPlots = 2
        lw = 3
        fs = 26
        fig, ax = plt.subplots(numPlots, figsize=(10,8))
        ax[0].plot(trajectories['y'].detach().cpu().reshape(nsteps_test+1, self.ny), linewidth=lw)
        ax[0].plot(Ymin, '--', linewidth=lw, c='k')
        ax[0].plot(Ymax, '--', linewidth=lw, c='k')
        ax[0].set_ylabel('y', fontsize=fs)
        # ax[0].set(ylim=[10,30])
        ax[1].plot(trajectories['u'].detach().cpu().reshape(nsteps_test, self.nu), linewidth=lw)
        ax[1].set_ylabel('u', fontsize=fs)
        ax[1].set(ylim=(-0.1,1.1))
        for i in range(numPlots):
            ax[i].grid(True)
            ax[i].tick_params(axis='x', labelsize=26)
            ax[i].tick_params(axis='y', labelsize=26)
            ax[i].set_xlim(0, nsteps_test)
            # ax[i].set(xlim=[0, 1440])
        plt.tight_layout()
        plt.savefig(self.saveDir+'/simplified_rollout', dpi=fig.dpi)
        plt.close(fig)

        # Plot training and validation loss
        # Should only run if the model was trained on this run instead of loaded from a file
        if (self.callback.debugLevel >= DebugLevel.EPOCH_LOSS) and (len(self.callback.loss) > 0):
            loss_df = pd.DataFrame(self.callback.loss)
            loss_df = loss_df[['train_loss', 'dev_loss']]
            loss_df['train_loss'] = loss_df['train_loss'].apply(lambda x: x.detach().item())
            loss_df['dev_loss'] = loss_df['dev_loss'].apply(lambda x: x.detach().item())
            loss_df.to_csv(self.saveDir+'/loss.csv', index=False)
            fig = plt.figure()
            plt.plot(loss_df)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(loss_df.keys())
            plt.tight_layout()
            fig.savefig(self.saveDir+'/controller_loss', dpi=fig.dpi)
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
        batched_range = torch.tensor(self.rng.uniform(low=1.0, high = 8.0, size=(self.n_samples, 1, self.nref)), dtype=torch.float32, device=self.device).repeat(1, self.nsteps+1, 1)
        batched_xmax = batched_xmin + batched_range

        # sampled disturbance trajectories from simulation model
        temp_d = []
        for _ in range(self.n_samples):
            start_idx = self.rng.integers(0, len(dataset['D'])-1-self.nsteps)
            temp_d.append(torch.tensor(self.Get_D(dataset['D'], self.nsteps, start_idx),
                                      dtype=torch.float32, device=self.device))
        batched_dist = torch.stack(temp_d)

        # sampled initial conditions
        batched_x0 = torch.stack([torch.tensor(self.Get_X0(dataset['X']),
                                               dtype=torch.float32, device=self.device).unsqueeze(0)
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
    
class my_round_func(torch.autograd.function.InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input
    
class SSM(nn.Module):
    def __init__(self, A, B, F, nx, nu, nd, device):
        super().__init__()
        self.A = nn.Parameter(A, requires_grad=True)
        # self.B = nn.Parameter(B, requires_grad=True)
        self.B = B
        self.F = nn.Parameter(F, requires_grad=True)
        # self.A = A
        # self.B = B
        # self.F = F
        self.device = device
        self.in_features = nx + nu + nd
        self.out_features = nx

    def forward(self, x, u, d):
        assert len(x.shape) == 2, x.shape
        assert len(u.shape) == 2
        assert len(d.shape) == 2

        x = x @ self.A.T + self.B(u) + d @ self.F.T
        L, V = torch.linalg.eig(self.A)
        L = L.imag[np.newaxis, :]
        # L = L.to(device=self.device)

        return (x, L)
    
    # def forward(self, x, u, d):
    #     assert len(x.shape) == 2, x.shape
    #     assert len(u.shape) == 2
    #     assert len(d.shape) == 2

    #     x = x @ self.A.T + u @ self.B.T + d @ self.F.T

    #     return x
    
class Normalizer():
    '''Class that handles the normalizing and de-normalizing of a variety of data structures'''
    def __init__(self):
        self.dataInfo = {'leave': {'max': 1,
                                   'min': 0,
                                   'mean': 0,
                                   'std': 1}}

    def add_data(self, data, keys=None):
        '''
        Adds an entry to dataInfo to keep track of input data's descriptors
        Inputs:
            data: (dict, DictDataset, DataFrame, nparray) data to describe
            keys: (list[str]) names to use when storing data descriptors, defaults to None
        Outputs:
            (None)
        '''
        # Numpy array or tensor where data descriptors (calculated along axis 0) should be stored under a single key
        # Assumes only one key is provided
        if isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
            if keys is None:
                raise ValueError('List of keys is required for numpy arrays and torch tensors')
            names = enumerate(keys)

            self.dataInfo[keys[0]] = {'max': np.max(data, axis=0),
                                    'min': np.min(data, axis=0),
                                    'mean': np.mean(data, axis=0),
                                    'std': np.std(data, axis=0)}
            return
        # For dataframe and series, use column names if user keys are not provided
        elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            names = data.columns.tolist()
            if keys is None:
                keys = names
        # For dict and DictDataset, use keys if user keys are not provided
        elif isinstance(data, dict) or isinstance(data, DictDataset):
            names = list(data.keys())
            if keys is None:
                names = keys
        else:
            raise TypeError('Data structure used is not supported')
        
        for key, name in zip(keys,names):
                self.dataInfo[key] = {'max': np.max(data[name], axis=0),
                                    'min': np.min(data[name], axis=0),
                                    'mean': np.mean(data[name], axis=0),
                                    'std': np.std(data[name], axis=0)}
        
        if self.dataInfo[key]['max'] == self.dataInfo[key]['min']:
            self.dataInfo[key] = self.dataInfo['leave']
            
        return

    def norm(self, data, keys=None):
        '''
        Normalizes a provided dataset based on previously added descriptors
        Inputs:
            data: (dict, DictDataset, DataFrame, nparray) data to normalize
            keys: (list[str]) keys to use when normalizing, defaults to None
        Outputs:
            Normalized data
        '''
        # numpy arrays and tensors are normalized based on a multi-dimensonal descriptor stored under one key
        # Assumes only one key is provided
        if isinstance(data, (np.ndarray, torch.Tensor)):
            if keys is None:
                raise ValueError('List of keys is required for numpy arrays and torch tensors')
            stats = self.dataInfo[keys[0]]
            norm_data = (data - stats['min']) / (stats['max'] - stats['min'])
            return norm_data
        # For dataframe, use column names if user keys are not provided
        elif isinstance(data, pd.DataFrame):
            names = data.columns.tolist()
            if keys is None:
                keys = data.columns.tolist()
        # For dict and DictDataset, use keys if user keys are not provided
        elif isinstance(data, dict) or isinstance(data, DictDataset):
            names = list(data.keys())
            if keys is None:
                keys = list(data.keys())
        else:
            # Special case for a single, scalar value
            try:
                stats = self.dataInfo[keys[0]]
                norm_data = (data - stats['min']) / (stats['max'] - stats['min'])
                return norm_data
            except: 
                raise TypeError('Data structure used is not supported')

        # Min max normalization
        norm_data = data.copy()
        for key,name in zip(keys,names):
            stats = self.dataInfo[key]
            if isinstance(norm_data[name], torch.Tensor):
                dev = norm_data[name].get_device()
                if dev == -1:
                    device = torch.device("cpu")
                else:
                    device = torch.device("cuda:"+str(dev))
                norm_data[name] = (norm_data[name] - torch.tensor(stats['min'], device=device)) / torch.tensor((stats['max'] - stats['min']), device=device)
            else:
                norm_data[name] = (norm_data[name] - stats['min']) / (stats['max'] - stats['min'])

        return norm_data
    
    def normDelta(self, data, keys):
        '''
        Special case normalizer for delta values
        Inputs:
            data: (ndarray or scalar value) delta value(s) to normalize
            keys: list[str] list of length 1 with key to use
        Outputs:
            normalized data
        '''
        stats = self.dataInfo[keys[0]]
        norm_data = data / (stats['max'] - stats['min'])
        return norm_data

    def denorm(self, data, keys=None):
        '''
        De-normalize a provided dataset based on previously added descriptors
        Inputs:
            data: (dict, DictDataset, DataFrame, nparray) data to de-normalize
            keys: (list[str]) keys to use for de-normalizing, defaults to None
        Outputs:
            De-normalized data
        '''
        # numpy arrays and tensors are de-normalized based on a multi-dimensonal descriptor stored under one key
        # Assumes only one key is provided
        if isinstance(data, (np.ndarray, torch.Tensor)):
            if keys is None:
                raise ValueError('List of keys is required for numpy arrays and torch tensors')
            stats = self.dataInfo[keys[0]]
            denorm_data = data * (stats['max'] - stats['min']) + stats['min']
            return denorm_data
        # For dataframe, use column names if user keys are not provided
        if isinstance(data, pd.DataFrame):
            names = data.columns.tolist()
            if keys is None:
                keys = data.columns.tolist()
        # For dict and DictDataset, use keys if user keys are not provided
        elif isinstance(data, dict) or isinstance(data, DictDataset):
            names = list(data.keys())
            if keys is None:
                keys = list(data.keys())
        else:
            # Special case for a single, scalar value
            try:
                stats = self.dataInfo[keys[0]]
                denorm_data = data * (stats['max'] - stats['min']) + stats['min']
                return denorm_data
            except: 
                raise TypeError('Data structure used is not supported')

        denorm_data = data.copy()

        for key,name in zip(keys,names):
            stats = self.dataInfo[key]
            if isinstance(denorm_data[name], torch.Tensor):
                dev = denorm_data[name].get_device()
                if dev == -1:
                    device = torch.device("cpu")
                else:
                    device = torch.device("cuda:"+str(dev))
                denorm_data[name] = denorm_data[name] * torch.tensor((stats['max'] - stats['min']), device=device) + torch.tensor(stats['min'], device=device)
            else:
                denorm_data[name] = denorm_data[name] * (stats['max'] - stats['min']) + stats['min']

        return denorm_data
    
    def save(self, filePath=''):
        '''
        Saves the dataInfo dict as a json file
        '''
        dataInfoCopy = copy.deepcopy(self.dataInfo)
        for key,value in dataInfoCopy.items():
            if type(value['max']) == np.ndarray:
                for k,v in value.items():
                    value[k] = v.tolist()
        Path(filePath).mkdir(parents=True, exist_ok=True)
        with open(filePath+'norm_dataInfo.json', 'w') as fp:
            json.dump(dataInfoCopy, fp)

    def load(self, filePath=''):
        '''
        Loads a previously saved dataInfo dict from a json file
        '''
        with open(filePath+'norm_dataInfo.json') as fp:
            dataInfoCopy = json.load(fp)
        for key, value in dataInfoCopy.items():
            if type(value['max']) == list:
                for k,v in value.items():
                    value[k] = np.array(v)
        self.dataInfo = dataInfoCopy

# Custom debug and callback code
class DebugLevel(Enum):
    NO = 0
    MODEL_DIAGRAMS = 1
    EPOCH_LOSS = 2
    EPOCH_VALUES = 3

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        if isinstance(other, numbers.Number):
            return self.value < other
        return NotImplemented
    
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        if isinstance(other, numbers.Number):
            return self.value <= other
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        if isinstance(other, numbers.Number):
            return self.value > other
        return NotImplemented
    
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        if isinstance(other, numbers.Number):
            return self.value >= other
        return NotImplemented

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.value == other.value
        if isinstance(other, numbers.Number):
            return self.value == other
        return NotImplemented
    
    def __ne__(self, other):
        if self.__class__ is other.__class__:
            return self.value != other.value
        if isinstance(other, numbers.Number):
            return self.value != other
        return NotImplemented


class Callback_NODE(Callback):
    def __init__(self, debugLevel):
        self.debugLevel = debugLevel

        # Monitoring loss by epoch
        self.loss = []

    def end_eval(self, trainer, output):
        if self.debugLevel < DebugLevel.EPOCH_LOSS:
            return
        self.loss.append(output)

class Callback_Controller(Callback):
    def __init__(self, debugLevel, savePath):
        self.debugLevel = debugLevel

        # Monitoring loss by epoch
        self.loss = []

        # Monitoring values by epoch
        self.savePath = savePath
        self.ResetTrainValues()

    def ResetTrainValues(self):
        self.trainValues = {'train_u': [], 'train_y': [], 'train_ymin': [], 'train_ymax': []}

    def end_batch(self, trainer, output):
        if self.debugLevel < DebugLevel.EPOCH_VALUES:
            return
        # print(output.keys())
        # print('train_xn', output['train_xn'].shape)
        # print('train_y', output['train_y'].shape)
        # print('train_ymin', output['train_ymin'].shape)
        # input('paused')
        self.trainValues['train_u'].append(output['train_u'].detach().cpu().flatten().numpy())
        self.trainValues['train_y'].append(output['train_y'][:,:-1,:].detach().cpu().flatten().numpy())
        self.trainValues['train_ymin'].append(output['train_ymin'][:,:-1,:].detach().cpu().flatten().numpy())
        self.trainValues['train_ymax'].append(output['train_ymax'][:,:-1,:].detach().cpu().flatten().numpy())

    def begin_epoch(self, trainer, output):
        if self.debugLevel < DebugLevel.EPOCH_VALUES:
            return
        train_u = np.array(self.trainValues['train_u']).flatten()
        train_y = np.array(self.trainValues['train_y']).flatten()
        train_ymin = np.array(self.trainValues['train_ymin']).flatten()
        train_ymax = np.array(self.trainValues['train_ymax']).flatten()
        fig, ax = plt.subplots(2, figsize=(10,8))
        ax[0].plot(train_u)
        ax[0].set(ylim=[0,1.1], title='u', xlim=[0,2000])
        ax[1].plot(train_y)
        ax[1].plot(train_ymin, '--', c='k', zorder=-1)
        ax[1].plot(train_ymax, '--', c='k', zorder=-1)
        ax[1].set(ylim=[10,35], title='y', xlim=[0,2000])
        plt.figtext(0.01, 0.01, f"Epoch: {output['train_epoch']}", fontsize=14)
        plt.tight_layout()
        Path(f'{self.savePath}/debug').mkdir(exist_ok=True)
        fig.savefig(f"{self.savePath}/debug/epoch{output['train_epoch']}")
        plt.close(fig)
        self.ResetTrainValues()

    def end_eval(self, trainer, output):
        if self.debugLevel < DebugLevel.EPOCH_LOSS:
            return
        self.loss.append(output)

    def end_train(self, trainer, output):
        if self.debugLevel < DebugLevel.EPOCH_LOSS:
            return
        ims = [imageio.v3.imread(f"{self.savePath}/debug/epoch{i}.png") for i in range(len(os.listdir(self.savePath+'/debug/')))]
        imageio.mimwrite(self.savePath+'/train.gif', ims, fps=10)
        shutil.rmtree(self.savePath+'/debug/')
        