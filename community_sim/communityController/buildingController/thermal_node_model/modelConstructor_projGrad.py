import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from pathlib import Path
import imageio
import os
import shutil
import datetime as dt
import typing
from abc import ABC, abstractmethod
from enum import Enum
import numbers
import copy
import json

import neuromancer.psl as psl
from neuromancer.system import Node, System
from neuromancer.modules import blocks, solvers
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable, Variable
from neuromancer.loss import PenaltyLoss, BarrierLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.dynamics import integrators
from neuromancer.loggers import BasicLogger
from neuromancer.callbacks import Callback
import neuromancer.slim as slim

class NeuromancerModel(ABC):
    '''
    Base class for neuromancer models
    Attributes:
        manager: (RunManager) object that keeps track of various parameters about the model
        nsteps: (int) number of timesteps the model is expected to forecast
        batch_size: (int) number of samples per batch
        max_epochs: (int) maximum epochs for training
        patience: (int) minimum number of epochs since improvement to end training early
        warmup: (int) minimum number of epochs for training
        lr: (float) learning rate for training
        device: (torch.device) device the training will be run on
        callback: (neuromancer.Callback) callback function used to debug mid-training
        saveDir: (str) directory where the model and test outputs will be saved
        runName: (str) name of the overall training run
        model: (neuromancer.Node) object containing the model
        problem: (neuromancer.Problem) optimization problem for training the model
        testData: (dict) dictionary containing the section of the dataset used for testing
    '''
    def __init__(self, manager, name:str, device:torch.device, debugLevel, saveDir:str):
        '''
        Constructor

        :param manager: (RunManager) object that keeps track of various model parameters
        :param name: (str) name of the model
        :param device: (torch.device) device to run training on
        :param debugLevel: (int or DebugLevel) sets the level of debug detail outputted from training
        :param saveDir: (str) relative path to where the model should be saved/loaded from, defaults to value of 'name'
        '''
        self.manager = manager

        train_params = manager.models[name]['train_params']
        self.nsteps = train_params['nsteps']
        self.batch_size = train_params['batch_size']
        self.max_epochs = train_params['max_epochs']
        self.patience = train_params['patience']
        self.warmup = train_params['warmup']
        self.lr = train_params['lr']
        self.device = device

        if saveDir is None:
            self.saveDir = name
        else:
            self.saveDir = saveDir

        self.callback = Callback_Basic(debugLevel)

        self.runName = manager.name

        self.model = None
        self.problem = None
        self.testData = None

    @abstractmethod
    def PrepareDataset(self, dataset):
        '''
        Set up and batch dataset into desired training split
        Empty method designed to be overridden by a child class
        '''
        pass

    def TrainModel(self, dataset, load, test=True):
        '''
        Trains building thermal system or loads from file

        :param dataset: (dict) normalized dataset split into states (X), inputs (U), and disturbances (D)
        :param load: (bool) load model from file instead of training
        '''
        # 
        if test:
            trainLoader, devLoader, testData = self.PrepareDataset(dataset)
            self.testData = testData

        # If load is true, skip training and just load from state dict file
        if load:
            self.LoadModel()
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

        best_model = trainer.train()
        self.problem.load_state_dict(best_model)

    def LoadModel(self):
        '''
        Loads the model from previously saved weights
        Call this directly when deploying
        '''
        self.problem.load_state_dict(torch.load(self.saveDir+'/best_model_state_dict.pth', weights_only=True))
        self.problem.eval()

    def PlotLoss(self):
        '''
        Plots the training and validation loss of the epochs
        '''
        # Should only run if the model was trained on this run instead of loaded from a file
        if (self.callback.debugLevel < DebugLevel.EPOCH_LOSS) or (len(self.callback.loss) == 0):
            print("Model was not trained at sufficiently high debug level or was loaded from a file. Loss plots will not be created.")
            return
        
        # Plotting
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
        fig.savefig(self.saveDir+'/building_loss', dpi=fig.dpi)
        plt.close(fig)

class BuildingNode(NeuromancerModel):
    def __init__(self, nx:int, nu:int, nd:int, manager, name:str, device:torch.device, debugLevel, saveDir:typing.Optional[str]=None):
        '''
        Constructor function

        :param nx: (int) number of states
        :param nu: (int) number of control signals
        :param nd: (int) number of disturbances
        :param manager (RunManager) object that keeps track of various model parameters
        :param name: (str) name of the model
        :param device: (torch.device) device to run training on
        :param debugLevel: (int or DebugLevel) sets the level of debug detail outputted from training
        :param saveDir: (str) relative path to where the model should be saved/loaded from, defaults to value of 'name'
        '''
        super().__init__(manager, name, device, debugLevel, saveDir)

        self.nx = nx
        self.nu = nu
        self.nd = nd

        self.hsizes = self.manager.models[name]['hsizes']

        self.callback = Callback_NODE(debugLevel, self.saveDir)

    def PrepareDataset(self, dataset):
        '''
        Prepare data for training and testing of building thermal model

        :param dataset: (dict) normalized dataset split into states (X), inputs (U), and disturbances (D)

        :return trainLoader: (DataLoader) DataLoader object for training
        :return devLoader: (DataLoader) DataLoader object for validation
        :return testData: (dict) Batched dataset designated for testing
        '''
        n = len(dataset['X'])

        train = {}
        train['X'] = dataset['X'][:int(np.round(n*0.7)),:]
        train['U'] = dataset['U'][:int(np.round(n*0.7)),:]
        train['D'] = dataset['D'][:int(np.round(n*0.7)),:]

        dev = {}
        dev['X'] = dataset['X'][int(np.round(n*0.7)):int(np.round(n*0.9)),:]
        dev['U'] = dataset['U'][int(np.round(n*0.7)):int(np.round(n*0.9)),:]
        dev['D'] = dataset['D'][int(np.round(n*0.7)):int(np.round(n*0.9)),:]

        test = {}
        test['X'] = dataset['X'][int(np.round(n*0.9)):,:]
        test['U'] = dataset['U'][int(np.round(n*0.9)):,:]
        test['D'] = dataset['D'][int(np.round(n*0.9)):,:]

        nbatch = len(test['X']) // self.nsteps

        trainX = train['X'].reshape(nbatch*7, self.nsteps, self.nx)
        trainX = torch.tensor(trainX, dtype=torch.float32, device=self.device)
        trainU = train['U'].reshape(nbatch*7, self.nsteps, self.nu)
        trainU = torch.tensor(trainU, dtype=torch.float32, device=self.device)
        trainD = train['D'].reshape(nbatch*7, self.nsteps, self.nd)
        trainD = torch.tensor(trainD, dtype=torch.float32, device=self.device)
        trainData = DictDataset({'x': trainX, 'xn': trainX[:,0:1,:],
                                'u': trainU, 'd': trainD}, name='train')
        # for key, value in trainData.datadict.items():
        #     print(key, value.shape)
        trainLoader = DataLoader(trainData, batch_size=self.batch_size,
                                collate_fn=trainData.collate_fn, shuffle=False)
        
        devX = dev['X'].reshape(nbatch*2, self.nsteps, self.nx)
        devX = torch.tensor(devX, dtype=torch.float32, device=self.device)
        devU = dev['U'].reshape(nbatch*2, self.nsteps, self.nu)
        devU = torch.tensor(devU, dtype=torch.float32, device=self.device)
        devD = dev['D'].reshape(nbatch*2, self.nsteps, self.nd)
        devD = torch.tensor(devD, dtype=torch.float32, device=self.device)
        devData = DictDataset({'x': devX, 'xn': devX[:,0:1,:],
                                'u': devU, 'd': devD}, name='dev')
        # for key, value in devData.datadict.items():
        #     print(key, value.shape)
        devLoader = DataLoader(devData, batch_size=self.batch_size,
                                collate_fn=devData.collate_fn, shuffle=True)
        
        testX = test['X'].reshape(nbatch, self.nsteps, self.nx)
        testX = torch.tensor(testX, dtype=torch.float32, device=self.device)
        testU = test['U'].reshape(nbatch, self.nsteps, self.nu)
        # testU[:,:,1] = np.zeros_like(testU[:,:,1])
        # testU[:,:,0] = np.zeros_like(testU[:,:,0])
        testU = torch.tensor(testU, dtype=torch.float32, device=self.device)
        testD = test['D'].reshape(nbatch, self.nsteps, self.nd)
        testD = torch.tensor(testD, dtype=torch.float32, device=self.device)
        testData = {'x': testX, 'xn': testX[:,0:1,:], 'u': testU, 'd':testD}
        # for key, value in testData.items():
        #     print(key, value.shape)

        return trainLoader, devLoader, testData

    def CreateModel(self):
        '''Defines building thermal system'''
        fx = blocks.MLP(self.nx+self.nu+self.nd, self.nx, bias=True, linear_map=torch.nn.Linear,
                        nonlin=torch.nn.Sigmoid, hsizes=self.hsizes)
    
        fxRK4 = integrators.RK4(fx)

        model = Node(fxRK4, ['xn', 'u', 'd'], ['xn'], name="buildingNODE")

        C = torch.tensor([[1.0]], device=self.device)
        ynext = lambda x: x @ C.T
        output_model = Node(ynext, ['xn'], ['y'], name='out_obs')

        dynamics_model = System([model, output_model], name='system', nsteps=self.nsteps)
        self.model = dynamics_model
        dynamics_model.to(device=self.device)
        if self.callback.debugLevel > DebugLevel.NO:
            dynamics_model.show(self.manager.runPath+'thermalModelDiagram.png')

        x = variable("x")
        xhat = variable('xn')[:, :-1, :]

        referenceLoss = 5.*(xhat == x)^2
        referenceLoss.name = 'ref_loss'

        onestepLoss = 1.*(xhat[:,1,:] == x[:,1,:])^2
        onestepLoss.name = 'onestep_loss'

        objectives = [referenceLoss, onestepLoss]
        constraints = []

        loss = PenaltyLoss(objectives, constraints)

        self.problem = Problem([dynamics_model], loss)
        if self.callback.debugLevel > DebugLevel.NO:
            self.problem.show(self.manager.runPath+'NODE_optim.png')

    def TestModel(self):
        '''Plots the testing of the building thermal model'''
        dynamics_model = self.problem.nodes[0]
        dynamics_model.nsteps = self.testData['x'].shape[1]

        testOutputs = dynamics_model(self.testData)

        pred_traj = testOutputs['xn'][:,:-1,:].detach().cpu().numpy().reshape(-1,self.nx)
        true_traj = self.testData['x'].detach().cpu().numpy().reshape(-1,self.nx)
        input_traj = self.testData['u'].detach().cpu().numpy().reshape(-1,self.nu)
        dist_traj = self.testData['d'].detach().cpu().numpy().reshape(-1,self.nd)
        pred_traj, true_traj = pred_traj.transpose(1,0), true_traj.transpose(1,0)

        testMetrics = pd.DataFrame()
        test_mae = np.mean(np.abs(pred_traj - true_traj))
        testMetrics['mae'] = [test_mae]
        testMetrics.to_csv(self.saveDir+'/test_metrics.csv', index=False)

        figsize = 25
        lw = 4.0
        fig,ax = plt.subplots(self.nx+self.nu+self.nd, figsize=(figsize, figsize))
        labels = [f'$y_{k}$' for k in range(len(true_traj))]
        for row, (t1, t2, label) in enumerate(zip(true_traj, pred_traj, labels)):
            axe = ax[row]
            axe.set_ylabel(label, rotation=0, labelpad=20, fontsize=figsize)
            axe.plot(t1, 'c', linewidth=lw, label="True")
            axe.plot(t2, 'm--', linewidth=lw, label='Pred')
            axe.tick_params(labelbottom=False, labelsize=figsize)
            axe.set_title("Indoor Air Temperature (Normalized)", fontsize=figsize)
            # axe.set_xlim([0,60])
            # axe.vlines(np.linspace(30, 270, 9), 0, 1)
        axe.tick_params(labelbottom=True, labelsize=figsize)
        axe.legend(fontsize=figsize)
        ax[-2].plot(dist_traj, 'c', linewidth=lw, label='Outdoor Air Temp')
        ax[-2].legend(fontsize=figsize)
        ax[-2].set_title("Outdoor Air Temperature (Normalized)", fontsize=figsize)
        ax[-2].tick_params(labelbottom=True, labelsize=figsize)
        # ax[-2].set_xlim([0,60])
        ax[-1].plot(input_traj, linewidth=lw, label='HVAC Consumption')
        ax[-1].legend(fontsize=figsize)
        ax[-1].set_xlabel('$time$', fontsize=figsize)
        ax[-1].set_ylabel('$u$', rotation=0, labelpad=20, fontsize=figsize)
        ax[-1].tick_params(labelbottom=True, labelsize=figsize)
        ax[-1].set_title("HVAC Consumption (Normalized)", fontsize=figsize)
        # ax[-1].set_xlim([0,60])
        plt.figtext(0.01, 0.01, self.runName, fontsize=figsize)
        plt.tight_layout()
        fig.savefig(self.saveDir+'/building_rollout', dpi=fig.dpi)
        plt.close(fig)

        lw = 2.0
        fs = 12
        fig,ax = plt.subplots(self.nx, figsize=(7, 5))
        labels = [f'$y_{k}$' for k in range(len(true_traj))]
        for row, (t1, t2) in enumerate(zip(true_traj, pred_traj)):
            ax.plot(t1, 'c', linewidth=lw, label="True")
            ax.plot(t2, 'm--', linewidth=lw, label='Pred')
        ax.tick_params(labelbottom=False, labelsize=fs)
        ax.set_ylabel('Indoor Air Temperature (Normalized)', rotation=0, labelpad=20, fontsize=fs)
        ax.tick_params(labelbottom=True, labelsize=fs)
        # ax.legend(fontsize=fs, loc='lower left', bbox_to_anchor=(0.78,1), ncol=2)
        ax.legend(fontsize=fs)
        plt.tight_layout()
        fig.savefig(self.saveDir+'/justTemp_rollout', dpi=fig.dpi)
        plt.close(fig)

        # No updated conditions
        # dynamics_model.nsteps = 60
        # testOutputs = dynamics_model(self.testData[0:1,:,:])

        # pred_traj = testOutputs['xn'][:,:-1,:].detach().cpu().numpy().reshape(-1,self.nx)
        # true_traj = self.testData['x'].detach().cpu().numpy().reshape(-1,self.nx)
        # input_traj = self.testData['u'].detach().cpu().numpy().reshape(-1,self.nu)
        # dist_traj = self.testData['d'].detach().cpu().numpy().reshape(-1,self.nd)
        # pred_traj, true_traj = pred_traj.transpose(1,0), true_traj.transpose(1,0)
        
        # figsize = 25
        # lw = 4.0
        # fig,ax = plt.subplots(self.nx+self.nu+self.nd, figsize=(figsize, figsize))
        # labels = [f'$y_{k}$' for k in range(len(true_traj))]
        # for row, (t1, t2, label) in enumerate(zip(true_traj, pred_traj, labels)):
        #     axe = ax[row]
        #     axe.set_ylabel(label, rotation=0, labelpad=20, fontsize=figsize)
        #     axe.plot(t1, 'c', linewidth=lw, label="True")
        #     axe.plot(t2, 'm--', linewidth=lw, label='Pred')
        #     axe.tick_params(labelbottom=False, labelsize=figsize)
        #     axe.set_title("Indoor Air Temperature (Normalized)", fontsize=figsize)
        #     axe.set_xlim([0,60])
        # axe.tick_params(labelbottom=True, labelsize=figsize)
        # axe.legend(fontsize=figsize)
        # ax[-2].plot(dist_traj, 'c', linewidth=lw, label='Outdoor Air Temp')
        # ax[-2].legend(fontsize=figsize)
        # ax[-2].set_title("Outdoor Air Temperature (Normalized)", fontsize=figsize)
        # ax[-2].tick_params(labelbottom=True, labelsize=figsize)
        # ax[-2].set_xlim([0,60])
        # ax[-1].plot(input_traj, linewidth=lw, label='HVAC Consumption')
        # ax[-1].legend(fontsize=figsize)
        # ax[-1].set_xlabel('$time$', fontsize=figsize)
        # ax[-1].set_ylabel('$u$', rotation=0, labelpad=20, fontsize=figsize)
        # ax[-1].tick_params(labelbottom=True, labelsize=figsize)
        # ax[-1].set_title("HVAC Consumption (Normalized)", fontsize=figsize)
        # ax[-1].set_xlim([0,60])
        # plt.figtext(0.01, 0.01, self.runName, fontsize=figsize)
        # plt.tight_layout()
        # fig.savefig(self.saveDir+'/building_rollout_short', dpi=fig.dpi)
        # plt.close(fig)

        # Plot training and validation loss
        self.PlotLoss()

class ExperimentalThermalModel(NeuromancerModel):
    def __init__(self, nx:int, nu:int, nd:int, manager, name:str, device:torch.device, debugLevel, saveDir:typing.Optional[str]=None):
        '''
        Constructor function

        :param nx: (int) number of states
        :param nu: (int) number of control signals
        :param nd: (int) number of disturbances
        :param manager (RunManager) object that keeps track of various model parameters
        :param name: (str) name of the model
        :param device: (torch.device) device to run training on
        :param debugLevel: (int or DebugLevel) sets the level of debug detail outputted from training
        :param saveDir: (str) relative path to where the model should be saved/loaded from, defaults to value of 'name'
        '''
        super().__init__(manager, name, device, debugLevel, saveDir)

        self.nx = nx
        self.nu = nu
        self.nd = nd

        self.hsizes = self.manager.models[name]['hsizes']

        self.callback = Callback_NODE(debugLevel, self.saveDir)

    def PrepareDataset(self, dataset):
        '''
        Prepare data for training and testing of building thermal model

        :param dataset: (dict) normalized dataset split into states (X), inputs (U), and disturbances (D)

        :return trainLoader: (DataLoader) DataLoader object for training
        :return devLoader: (DataLoader) DataLoader object for validation
        :return testData: (dict) Batched dataset designated for testing
        '''
        n = len(dataset['X'])

        train = {}
        train['X'] = dataset['X'][:int(np.round(n*0.7)),:]
        train['U'] = dataset['U'][:int(np.round(n*0.7)),:]
        train['D'] = dataset['D'][:int(np.round(n*0.7)),:]

        dev = {}
        dev['X'] = dataset['X'][int(np.round(n*0.7)):int(np.round(n*0.9)),:]
        dev['U'] = dataset['U'][int(np.round(n*0.7)):int(np.round(n*0.9)),:]
        dev['D'] = dataset['D'][int(np.round(n*0.7)):int(np.round(n*0.9)),:]

        test = {}
        test['X'] = dataset['X'][int(np.round(n*0.9)):,:]
        test['U'] = dataset['U'][int(np.round(n*0.9)):,:]
        test['D'] = dataset['D'][int(np.round(n*0.9)):,:]

        nbatch = len(test['X']) // self.nsteps

        trainX = train['X'].reshape(nbatch*7, self.nsteps, self.nx)
        trainX = torch.tensor(trainX, dtype=torch.float32, device=self.device)
        trainU = train['U'].reshape(nbatch*7, self.nsteps, self.nu)
        trainU = torch.tensor(trainU, dtype=torch.float32, device=self.device)
        trainD = train['D'].reshape(nbatch*7, self.nsteps, self.nd)
        trainD = torch.tensor(trainD, dtype=torch.float32, device=self.device)
        trainData = DictDataset({'x': trainX, 'xn': trainX[:,0:1,:],
                                'u': trainU, 'd': trainD}, name='train')
        # for key, value in trainData.datadict.items():
        #     print(key, value.shape)
        trainLoader = DataLoader(trainData, batch_size=self.batch_size,
                                collate_fn=trainData.collate_fn, shuffle=False)
        
        devX = dev['X'].reshape(nbatch*2, self.nsteps, self.nx)
        devX = torch.tensor(devX, dtype=torch.float32, device=self.device)
        devU = dev['U'].reshape(nbatch*2, self.nsteps, self.nu)
        devU = torch.tensor(devU, dtype=torch.float32, device=self.device)
        devD = dev['D'].reshape(nbatch*2, self.nsteps, self.nd)
        devD = torch.tensor(devD, dtype=torch.float32, device=self.device)
        devData = DictDataset({'x': devX, 'xn': devX[:,0:1,:],
                                'u': devU, 'd': devD}, name='dev')
        # for key, value in devData.datadict.items():
        #     print(key, value.shape)
        devLoader = DataLoader(devData, batch_size=self.batch_size,
                                collate_fn=devData.collate_fn, shuffle=True)
        
        testX = test['X'].reshape(nbatch, self.nsteps, self.nx)
        testX = torch.tensor(testX, dtype=torch.float32, device=self.device)
        testU = test['U'].reshape(nbatch, self.nsteps, self.nu)
        # testU[:,:,1] = np.zeros_like(testU[:,:,1])
        # testU[:,:,0] = np.zeros_like(testU[:,:,0])
        testU = torch.tensor(testU, dtype=torch.float32, device=self.device)
        testD = test['D'].reshape(nbatch, self.nsteps, self.nd)
        testD = torch.tensor(testD, dtype=torch.float32, device=self.device)
        testData = {'x': testX, 'xn': testX[:,0:1,:], 'u': testU, 'd':testD}
        # for key, value in testData.items():
        #     print(key, value.shape)

        return trainLoader, devLoader, testData

    def CreateModel(self):
        '''Defines building thermal system'''
        fx = blocks.MLP(self.nx+self.nu+self.nd, self.nx, bias=True, linear_map=torch.nn.Linear,
                        nonlin=torch.nn.Sigmoid, hsizes=self.hsizes)
    
        fxRK4 = integrators.RK4(fx)

        model = Node(fxRK4, ['xn', 'u', 'd'], ['xn'], name="buildingNODE")

        C = torch.tensor([[1.0, 0.0]], device=self.device)
        ynext = lambda x: x @ C.T
        output_model = Node(ynext, ['xn'], ['y'], name='out_obs')

        dynamics_model = System([model, output_model], name='system', nsteps=self.nsteps)
        self.model = dynamics_model
        dynamics_model.to(device=self.device)
        if self.callback.debugLevel > DebugLevel.NO:
            dynamics_model.show(self.manager.runPath+'thermalModelDiagram.png')

        x = variable("x")
        xhat = variable('xn')[:, :-1, :]

        referenceLoss = 5.*(xhat == x)^2
        referenceLoss.name = 'ref_loss'

        onestepLoss = 1.*(xhat[:,1,:] == x[:,1,:])^2
        onestepLoss.name = 'onestep_loss'

        objectives = [referenceLoss, onestepLoss]
        constraints = []

        loss = PenaltyLoss(objectives, constraints)

        self.problem = Problem([dynamics_model], loss)
        if self.callback.debugLevel > DebugLevel.NO:
            self.problem.show(self.manager.runPath+'NODE_optim.png')

    def TestModel(self):
        '''Plots the testing of the building thermal model'''
        dynamics_model = self.problem.nodes[0]
        dynamics_model.nsteps = self.testData['x'].shape[1]

        testOutputs = dynamics_model(self.testData)

        pred_traj = testOutputs['xn'][:,:-1,:].detach().cpu().numpy().reshape(-1,self.nx)
        true_traj = self.testData['x'].detach().cpu().numpy().reshape(-1,self.nx)
        input_traj = self.testData['u'].detach().cpu().numpy().reshape(-1,self.nu)
        dist_traj = self.testData['d'].detach().cpu().numpy().reshape(-1,self.nd)
        pred_traj, true_traj = pred_traj.transpose(1,0), true_traj.transpose(1,0)

        testMetrics = pd.DataFrame()
        test_mae = np.mean(np.abs(pred_traj - true_traj))
        testMetrics['mae'] = [test_mae]
        testMetrics.to_csv(self.saveDir+'/test_metrics.csv', index=False)

        figsize = 25
        lw = 4.0
        fig,ax = plt.subplots(self.nx+self.nu+self.nd, figsize=(figsize, figsize))
        labels = [f'$y_{k}$' for k in range(len(true_traj))]
        for row, (t1, t2, label) in enumerate(zip(true_traj, pred_traj, labels)):
            axe = ax[row]
            axe.set_ylabel(label, rotation=0, labelpad=20, fontsize=figsize)
            axe.plot(t1, 'c', linewidth=lw, label="True")
            axe.plot(t2, 'm--', linewidth=lw, label='Pred')
            axe.tick_params(labelbottom=False, labelsize=figsize)
            axe.set_title("Indoor Air Temperature (Normalized)", fontsize=figsize)
            axe.set_xlim([0,60])
            # axe.vlines(np.linspace(30, 270, 9), 0, 1)
        axe.tick_params(labelbottom=True, labelsize=figsize)
        axe.legend(fontsize=figsize)
        ax[-2].plot(dist_traj, 'c', linewidth=lw, label='Outdoor Air Temp')
        ax[-2].legend(fontsize=figsize)
        ax[-2].set_title("Outdoor Air Temperature (Normalized)", fontsize=figsize)
        ax[-2].tick_params(labelbottom=True, labelsize=figsize)
        ax[-2].set_xlim([0,60])
        ax[-1].plot(input_traj, linewidth=lw, label='HVAC Consumption')
        ax[-1].legend(fontsize=figsize)
        ax[-1].set_xlabel('$time$', fontsize=figsize)
        ax[-1].set_ylabel('$u$', rotation=0, labelpad=20, fontsize=figsize)
        ax[-1].tick_params(labelbottom=True, labelsize=figsize)
        ax[-1].set_title("HVAC Consumption (Normalized)", fontsize=figsize)
        ax[-1].set_xlim([0,60])
        plt.figtext(0.01, 0.01, self.runName, fontsize=figsize)
        plt.tight_layout()
        fig.savefig(self.saveDir+'/building_rollout', dpi=fig.dpi)
        plt.close(fig)

        lw = 2.0
        fs = 12
        fig,ax = plt.subplots(self.nx, figsize=(7, 5))
        labels = [f'$y_{k}$' for k in range(len(true_traj))]
        for row, (t1, t2, label) in enumerate(zip(true_traj, pred_traj, labels)):
            ax[row].plot(t1, 'c', linewidth=lw, label="True")
            ax[row].plot(t2, 'm--', linewidth=lw, label='Pred')
            ax[row].tick_params(labelbottom=False, labelsize=fs)
            ax[row].set_ylabel(label, rotation=0, labelpad=20, fontsize=fs)
            ax[row].tick_params(labelbottom=True, labelsize=fs)
        # ax.legend(fontsize=fs, loc='lower left', bbox_to_anchor=(0.78,1), ncol=2)
            ax[row].legend(fontsize=fs)
        plt.tight_layout()
        fig.savefig(self.saveDir+'/justTemp_rollout', dpi=fig.dpi)
        plt.close(fig)

        # Plot training and validation loss
        self.PlotLoss()

class ModeClassifier(NeuromancerModel):
    def __init__(self, nm:int, nu:int,  manager, name:str, device:torch.device, debugLevel, saveDir:typing.Optional[str]=None):
        '''
        Constructor function

        :param nm: (int) number of hvac modes
        :param nu: (int) number of control signals
        :param manager (RunManager) object that keeps track of various model parameters
        :param name: (str) name of the model
        :param device: (torch.device) device to run training on
        :param debugLevel: (int or DebugLevel) sets the level of debug detail outputted from training
        :param saveDir: (str) relative path to where the model should be saved/loaded from, defaults to value of 'name'
        '''
        super().__init__(manager, name, device, debugLevel, saveDir)

        self.nm = nm
        self.nu = nu

        self.hsizes = self.manager.models[name]['hsizes']

    def PrepareDataset(self, dataset):
        '''
        Prepare dataset for training
        '''
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
        trainData = DictDataset({'m': trainM, 'u_cool': trainU}, name='train')
        trainLoader = DataLoader(trainData, batch_size=self.batch_size,
                                collate_fn=trainData.collate_fn, shuffle=True)
        
        devM = dev['M'].reshape(nbatch*2, self.nsteps, self.nm)
        devM = torch.tensor(devM, dtype=torch.float32, device=self.device)
        devU = dev['U'].reshape(nbatch*2, self.nsteps, self.nu)
        devU = torch.tensor(devU, dtype=torch.float32, device=self.device)
        devData = DictDataset({'m': devM, 'u_cool': devU}, name='dev')
        devLoader = DataLoader(devData, batch_size=self.batch_size,
                                collate_fn=devData.collate_fn, shuffle=True)
        
        testM = test['M'].reshape(nbatch, self.nsteps, self.nm)
        testM = torch.tensor(testM, dtype=torch.float32, device=self.device)
        testU = test['U'].reshape(nbatch, self.nsteps, self.nu)
        testU = torch.tensor(testU, dtype=torch.float32, device=self.device)
        testData = {'m': testM, 'u_cool': testU}

        return trainLoader, devLoader, testData

    def CreateModel(self):
        '''Defines HVAC mode classifier'''
        net = blocks.MLP_bounds(
            insize=self.nu,
            outsize=self.nm,
            hsizes=self.hsizes,
            nonlin=nn.Sigmoid
        )
        self.model = Node(net, ['u_cool'], ['u'], name='Classifier')

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

    def TestModel(self):
        '''
        Create plots from test data
        '''
        dynamics_model = self.problem.nodes[0]
        dynamics_model.nsteps = self.testData['m'].shape[1]

        testOutputs = dynamics_model(self.testData)

        pred_traj = testOutputs['u'].detach().cpu().numpy().reshape(-1,self.nm)
        true_traj = self.testData['m'].detach().cpu().numpy().reshape(-1,self.nm)
        input_traj = self.testData['u_cool'].detach().cpu().numpy().reshape(-1,self.nu)
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
        # labels = [f'$y_{k}$' for k in range(len(true_traj))]
        labels = ['HVAC Mode']
        for row, (t1, t2, label) in enumerate(zip(true_traj, pred_traj, labels)):
            ax.set_ylabel(label, rotation=0, labelpad=20, fontsize=figsize)
            ax.plot(t1, 'c', linewidth=lw, label="True")
            ax.plot(t2, 'm--', linewidth=lw, label='Pred')
            ax.tick_params(labelbottom=False, labelsize=figsize)
            ax.set_xlim(0,1000)
        ax.tick_params(labelbottom=True, labelsize=figsize)
        ax.legend(fontsize=figsize)
        plt.tight_layout()
        fig.savefig(self.saveDir+'/mode', dpi=fig.dpi)
        plt.close(fig)

        # Plot training and validation loss
        self.PlotLoss()

class ControllerSystem(NeuromancerModel):
    '''
    Building energy management system controller
    Attributes:
        nx: (int) number of states
        nu: (int) number of control signals
        nd: (int) number of disturbances
        nd_obs: (int) number of observable disturbances
        ni: (int) number of information inputs
        ny: (int) number of controlled outputs
        nref: (int) number of reference signals
        y_idx: (list[int]) indices of observed states
        d_idx: (list[int]) indices of observable disturbances
        norm: (Normalizer) object used to normalize and denormalize data
        weights: (dict) weights to be used for each objective and constraint
        hsizes: (list[int]) size of hidden layers
        n_samples: (int) number of samples to create from dataset
        thermalModel: (Node) trained thermal model
        classifier: (Node) trained classifier model
        callback: (Callback_Controller) custom callback class
        rng: (np.random) random number generator
        system: (neuromancer.System) whole control system
    '''
    def __init__(self, nx:int, nu:int, nd:int, nd_obs:int, ni:int, ny:int, y_idx:int, d_idx:list[int], manager, name:str, norm,
                 thermalModel:Node, classifier:Node, device:torch.device, debugLevel, saveDir:typing.Optional[str]=None):
        '''
        Constructor

        :param nx: (int) number of states
        :param nu: (int) number of control signals
        :param nd: (int) number of disturbances
        :param nd_obs: (int) number of observable disturbances
        :param ni: (int) number of information inputs
        :param ny: (int) number of controlled outputs
        :param y_idx: (list[int]) indices of observed states
        :param d_idx: (list[int]) indices of observable disturbances
        :param manager (RunManager) object that keeps track of various model parameters
        :param name: (str) name of the model
        :param norm: (Normalizer) object used to normalize and denormalize data for plotting purposes
        :param thermalModel: (neuromancer.system or Node) trained building thermal model
        :param device: (torch.device) device to run training on
        :param debugLevel: (int or DebugLevel) sets the level of debug detail outputted from training
        :param saveDir: (str) relative path to where the model should be saved/loaded from, defaults to value of 'name'
        '''
        super().__init__(manager, name, device, debugLevel, saveDir)

        self.nx = nx
        self.nu = nu
        self.nd = nd
        self.nd_obs = nd_obs
        self.ni = ni
        self.ny = ny
        self.nref = ny
        self.y_idx = y_idx
        self.d_idx = d_idx

        self.norm = norm

        self.weights = manager.models[name]['weights']
        self.hsizes = manager.models[name]['hsizes']
        self.n_samples = manager.models[name]['train_params']['n_samples']

        self.trainParams = manager.models[name]['train_params']

        self.batSize = 16.4
        self.ratedKW = 9.6

        self.thermalModel = thermalModel
        self.classifier = classifier

        self.callback = Callback_Controller(debugLevel, self.saveDir)

        self.rng = np.random.default_rng(seed=60)

        self.system = None

    def CreateModel(self):
        '''Defines control system'''
        coolingNet = blocks.MLP_bounds(
            insize=self.ny + 2*self.nref + self.nd_obs + self.ni + 1,
            outsize=self.nu,
            hsizes=self.hsizes,
            nonlin=nn.GELU,
            min=0,
            max=1
        )

        coolPolicy = Node(coolingNet, ['y', 'ymin', 'ymax', 'd_obs', 'cost', 'dr', 'hvacPower'], ['u'], name='coolPolicy', groupName='HVAC')

        heatingNet = blocks.MLP_bounds(
            insize=self.ny + 2*self.nref + self.nd_obs + self.ni + 1,
            outsize=self.nu,
            hsizes=self.hsizes,
            nonlin=nn.GELU,
            min=0,
            max=1
        )

        heatPolicy = Node(heatingNet, ['y', 'ymin', 'ymax', 'd_obs', 'cost', 'dr', 'hvacPower'], ['u_heat'], name='heatPolicy', groupName='HVAC')

        hvacDenorm = lambda x: x * (self.norm.dataInfo['u']['max'] - self.norm.dataInfo['u']['min']) + self.norm.dataInfo['u']['min']
        hvacTruePower = Node(hvacDenorm, ['u'], ['hvacPower'], name='hvacTruePower', groupName='HVAC')

        batNet = blocks.MLP_bounds(
        insize=self.ni+4,
        outsize=self.nu,
        hsizes=self.hsizes,
        nonlin=nn.GELU,
        min=-self.ratedKW,
        max=self.ratedKW)
        batPolicy = Node(batNet, ['hvacPower', 'batPower', 'stored', 'cost', 'dr', 'batRef'], ['u_bat'], name='batPolicy', groupName='Battery')

        batModel = Node(BatteryModel(eff=0.95, capacity=self.batSize, nx=1, nu=1), ['stored', 'u_bat'], ['stored'], name='batModel', groupName='Battery')

        batTruePower = Node(batCorrect(self.batSize), ['stored', 'u_bat'], ['batPower'], name='batTruePower', groupName='Battery')
        # batTruePower = Node(lambda x: torch.gradient(x), ['stored'], ['batPower'], name='batTruePower', groupName='Battery')

        # Freeze thermal model
        thermal = self.thermalModel.nodes[0]
        out_obs = self.thermalModel.nodes[1]
        for p in thermal.parameters():
            p.requires_grad=False
        for p in self.classifier.parameters():
            p.requires_grad=False

        for item in self.thermalModel.nodes:
            item.groupName = 'HVAC'
        self.classifier.groupName = 'HVAC'

        dist_model = lambda d: d[:, self.d_idx]
        dist_obs = Node(dist_model, ['d'], ['d_obs'], name='dist_obs', groupName='HVAC')

        # closed loop system model
        self.system = System([dist_obs, coolPolicy, hvacTruePower, thermal, out_obs,
                              batPolicy, batModel, batTruePower
                              ],
                        nsteps=self.nsteps,
                        name='cl_system',
                        drop_init_cond=True)
        self.system.to(device=self.device)
        if self.callback.debugLevel > DebugLevel.NO:
            self.system.show(self.manager.runPath+"clSystemDigram.png", outputLines=False)
            with open(self.manager.runPath+'systemDiagram.txt', 'w') as file:
                file.write(self.system.graph().to_string())
        
        # DPC objectives and constraints
        # variables
        y = variable('y')
        # xn = variable('xn')
        # d = variable('d')
        u = variable('u')
        ymin = variable('ymin')
        ymax = variable('ymax')
        dr = variable('dr')         # Varies between -1 and 1
        cost = variable('cost')
        u_bat = variable('u_bat')
        stored = variable('stored')
        batRef = variable('batRef')
        hvacPower = variable('hvacPower')
        batPower = variable('batPower')
        # temp_model_var = Variable(input_variables=[xn, u, d],
        #                           func=thermal.callable,
        #                           display_name=thermal.name+'_var')

        # objectives
        # u_tot = hvacPower
        u_tot = hvacPower + batPower
        u_tot.name = 'u_tot'

        dr_loss = (dr * u_tot).minimize(weight=self.weights['dr_loss'], name='dr_loss')
        cost_loss = (cost * u_tot).minimize(weight=self.weights['cost_loss'], name='cost_loss')
        delta_loss = (u_tot[1:]-u_tot[:-1]).minimize(weight=self.weights['delta_loss'], name='delta_loss')

        hvac_loss = u == self.weights['hvac_loss'] * torch.zeros([self.batch_size, self.nsteps, 1])
        bat_loss = u_bat == self.weights['bat_loss'] * torch.zeros([self.batch_size, self.nsteps, 1])
        bat_max_loss = torch.abs(u_bat).minimize(weight=self.weights['bat_max_loss'], metric=torch.max, name='bat_max_loss')
        u_tot_loss = torch.abs(u_tot).minimize(weight=1.0, metric=torch.max, name='u_tot_max')

        # thermal comfort constraints
        state_lower_bound_penalty = self.weights['x_min'] * (y > ymin)
        state_upper_bound_penalty = self.weights['x_max'] * (y < ymax)
        hvac_lower = self.weights['x_min'] * (u > 0.0)

        # temp_model = self.weights['x_min'] * (y == temp_model_var)^2
        # temp_model.name = 'temp_model_eq'

        bat_life_lower_bound_penalty = self.weights['bat_min'] * (stored > batRef)

        # objectives and constraints names for nicer plot
        state_lower_bound_penalty.name = 'x_min'
        state_upper_bound_penalty.name = 'x_max'
        bat_life_lower_bound_penalty.name = 'bat_life_min'
        hvac_lower.name = 'hvac_lower'

        # list of constraints and objectives
        # objectives = [hvac_cost_loss, hvac_loss, bat_cost_loss, bat_loss]
        objectives = [cost_loss]
        constraints = [state_lower_bound_penalty, state_upper_bound_penalty, bat_life_lower_bound_penalty, hvac_lower]
        # constraints = [state_lower_bound_penalty, state_upper_bound_penalty, bat_life_lower_bound_penalty, bat_lower, bat_upper]

        proj_hvac = solvers.GradientProjSystemDyn(system=System([thermal, out_obs], nsteps=self.nsteps, name='thermal dynamics'),
                                     constraints=[state_lower_bound_penalty, state_upper_bound_penalty, hvac_lower],
                                     input_keys=["u"],
                                     system_keys=['y'],
                                     ic_keys=['xn'],
                                     num_steps=self.trainParams['projHVAC_steps'],
                                     step_size=self.trainParams['projHVAC_size'],
                                     decay=0.1,
                                     name='proj_hvac')
        
        proj_bat = solvers.GradientProjSystemDyn(system=System([batModel], nsteps=self.nsteps, name='battery dynamics', drop_init_cond=True),
                                         constraints = [bat_life_lower_bound_penalty],
                                         input_keys=['u_bat'],
                                         system_keys=['stored'],
                                         ic_keys=['stored'],
                                         num_steps=self.trainParams['projBat_steps'],
                                         step_size=self.trainParams['projBat_size'],
                                         decay=0.1,
                                         name='proj_bat')

        # Problem construction
        # nodes = [self.system]
        nodes = [self.system, proj_hvac, proj_bat]
        loss = PenaltyLoss(objectives, constraints)
        self.problem = Problem(nodes, loss, grad_inference=True)
        if self.callback.debugLevel > DebugLevel.NO:
            self.problem.show(self.manager.runPath+"MPC_optim.png")

    def TrainModel(self, dataset, tempMin, tempMax, load, test=True):
        '''
        Trains control system

        :param dataset: (dict) normalized dataset split into states (X), information inputs (I), and disturbances (D)
        :param tempMin: (float)
        :param tempMax: (float)
        :param load: (bool) load previous run from file
        '''
        if test:
            xmin_range = torch.distributions.Uniform(tempMin, tempMax)

            trainLoader, devLoader = [
                self.PrepareDataset(dataset, xmin_range, 
                                name=name) for name in ("train", "dev")
            ]

        # for key, value in trainLoader.dataset.datadict.items():
        #     print(key, value.shape)

        if load:
            self.LoadModel()
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
        # nsteps_test = 1440
        nsteps_test = 60

        # generate reference
        np_refs = (tempMin.cpu() + self.norm.normDelta(4.0, keys=['y'])) * np.ones(nsteps_test+1)
        np_refs = psl.signals.step(nsteps_test+1, 1, min=tempMin, max=tempMax, randsteps=5, rng=self.rng)
        ymin_val = torch.tensor(np_refs, dtype=torch.float32, device=self.device).reshape(1, nsteps_test+1, 1)
        ymax_val = ymin_val + self.norm.normDelta(2.0, keys=['y'])
        # get disturbance signal
        start_idx = self.rng.integers(0, len(dataset['D'])-nsteps_test)
        torch_dist = torch.tensor(self.Get_D(dataset['D'], nsteps_test+1, start_idx),
                                  dtype=torch.float32, device=self.device).unsqueeze(0)
        torch_price = torch.tensor(self.Get_D(dataset['I'], nsteps_test+1, start_idx),
                                  dtype=torch.float32, device=self.device).unsqueeze(0)
        # initial data for closed loop sim
        x0 = torch.tensor(self.Get_X0(dataset['X']),
                          dtype=torch.float32, device=self.device).reshape(1,1,self.nx)
        dr = torch.tensor(psl.signals.step(nsteps_test+1, 1, min=0, max=1, randsteps=5, rng=self.rng), dtype=torch.float32).reshape(1, nsteps_test+1, 1)
        # dr = torch.tensor(np.zeros((1, nsteps_test+1, 1)), dtype=torch.float32)
        stored0 = torch.tensor(self.rng.uniform(0,self.batSize,[1,1,1]), dtype=torch.float32, device=self.device)
        batRefYear = np.tile(pd.read_csv('socSchedule.csv', header=None).to_numpy(), (len(dataset['D']) // 1440,1)) * self.batSize
        batRef = torch.tensor(self.Get_D(batRefYear, nsteps_test, start_idx), dtype=torch.float32, device=self.device).unsqueeze(0)
        batMax = torch.tensor(np.ones((1,nsteps_test,1))*0.9*self.batSize, dtype=torch.float32, device=self.device)

        data = {'xn': x0,
                'y': x0[:,:,self.y_idx],
                'ymin': ymin_val,
                'ymax': ymax_val,
                'd': torch_dist,
                'cost': torch_price,
                'dr': dr,
                'stored': stored0,
                'batRef': batRef,
                'batMax': batMax,
                'hvacPower': torch.zeros_like(x0),
                'batPower': torch.zeros_like(x0)}
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
        Ymin = trajectories['ymin'].detach().cpu().reshape(nsteps_test, self.nref)
        Ymax = trajectories['ymax'].detach().cpu().reshape(nsteps_test, self.nref)

        numPlots = 4
        fig, ax = plt.subplots(numPlots, figsize=(20,16))
        ax[0].plot(trajectories['y'].detach().cpu().reshape(nsteps_test, self.ny), linewidth=3)
        ax[0].plot(Ymin, '--', linewidth=3, c='k')
        ax[0].plot(Ymax, '--', linewidth=3, c='k')
        ax[0].set_ylabel('y', fontsize=26)
        # ax[0].set(ylim=[10,30])
        ax[1].plot(trajectories['u'].detach().cpu().reshape(nsteps_test, self.nu), linewidth=3)
        ax[1].set_ylabel('u', fontsize=26)
        ax[1].set(ylim=(-0.1,1.1))
        ax[2].plot(trajectories['d'].detach().cpu().reshape(nsteps_test, self.nd), linewidth=3)
        ax[2].set_ylabel('d', fontsize=26)
        # ax[2].set_xlabel('Time [mins]', fontsize=26)
        ax[3].plot(trajectories['dr'].detach().cpu().reshape(nsteps_test, 1), linewidth=3, label='dr')
        ax[3].plot(trajectories['cost'].detach().cpu().reshape(nsteps_test, 1), linewidth=3, label='cost')
        ax[3].set_ylabel('dr', fontsize=26)
        ax[3].set_xlabel('Time [mins]', fontsize=26)
        ax[3].legend(fontsize=26)
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

        numPlots = 3
        fig, ax = plt.subplots(numPlots, figsize=(16,16))
        ax[0].plot(trajectories['stored'].detach().cpu().reshape(nsteps_test, self.ny), linewidth=3)
        ax[0].plot(trajectories['batRef'].detach().cpu().reshape(nsteps_test, 1), linewidth=3)
        ax[0].plot(trajectories['batMax'].detach().cpu().reshape(nsteps_test, 1), linewidth=3)
        ax[0].set_ylabel('stored', fontsize=26)
        # ax[0].set(ylim=[10,30])
        ax[1].plot(trajectories['u_bat'].detach().cpu().reshape(nsteps_test, 1), linewidth=3)
        ax[1].plot(trajectories['batPower'].detach().cpu().reshape(nsteps_test,1), linewidth=3)
        ax[1].set_ylabel('u_bat', fontsize=26)
        ax[2].plot(trajectories['dr'].detach().cpu().reshape(nsteps_test, 1), linewidth=3, label='dr')
        ax[2].plot(trajectories['cost'].detach().cpu().reshape(nsteps_test, 1), linewidth=3, label='cost')
        ax[2].set_ylabel('dr', fontsize=26)
        ax[2].set_xlabel('Time [mins]', fontsize=26)
        ax[2].legend(fontsize=26)
        for i in range(numPlots):
            ax[i].grid(True)
            ax[i].tick_params(axis='x', labelsize=26)
            ax[i].tick_params(axis='y', labelsize=26)
            ax[i].set_xlim(0, nsteps_test)
            # ax[i].set(xlim=[750, 1500])
        plt.figtext(0.01, 0.01, self.runName, fontsize=26)
        plt.tight_layout()
        plt.savefig(self.saveDir+'/bat_controller_rollout', dpi=fig.dpi)
        plt.close(fig)

        numPlots = 3
        fig, ax = plt.subplots(numPlots, figsize=(16,16))
        ax[0].plot(trajectories['batPower'].detach().cpu().reshape(nsteps_test, self.ny) + 
                   trajectories['hvacPower'].detach().cpu().reshape(nsteps_test, self.ny), linewidth=3)
        ax[0].set_ylabel('power [kW]', fontsize=26)
        # ax[0].set(ylim=[10,30])
        ax[1].plot(trajectories['d'].detach().cpu().reshape(nsteps_test, 1), linewidth=3)
        ax[1].set_ylabel('d', fontsize=26)
        ax[2].plot(trajectories['dr'].detach().cpu().reshape(nsteps_test, 1), linewidth=3, label='dr')
        ax[2].plot(trajectories['cost'].detach().cpu().reshape(nsteps_test, 1), linewidth=3, label='cost')
        ax[2].set_ylabel('dr', fontsize=26)
        ax[2].set_xlabel('Time [mins]', fontsize=26)
        ax[2].legend(fontsize=26)
        for i in range(numPlots):
            ax[i].grid(True)
            ax[i].tick_params(axis='x', labelsize=26)
            ax[i].tick_params(axis='y', labelsize=26)
            ax[i].set_xlim(0, nsteps_test)
            # ax[i].set(xlim=[750, 1500])
        plt.figtext(0.01, 0.01, self.runName, fontsize=26)
        plt.tight_layout()
        plt.savefig(self.saveDir+'/net_load', dpi=fig.dpi)
        plt.close(fig)

        numPlots = 2
        lw = 3
        fs = 26
        fig, ax = plt.subplots(numPlots, figsize=(10,8))
        ax[0].plot(trajectories['y'].detach().cpu().reshape(nsteps_test, self.ny), linewidth=lw)
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

        if len(self.problem.nodes) > 1:
            # Solution to pNLP via Neuromancer
            data['name'] = 'test'
            model_out = self.problem(data)
            x_nm = model_out['test_' + "y"].detach().numpy()
            y_nm = model_out['test_' + "u"].detach().numpy()

            # intermediate projection steps
            sol_map = self.problem.nodes[0]
            proj = self.problem.nodes[1]
            num_steps = proj.num_steps
            x_proj = sol_map(data)
            proj.num_steps = 1    # set projections steps to 1 for visualisation
            X_projected = [np.concatenate([x_proj['y'].detach().numpy(), x_proj['u'].detach().numpy()],axis=2)]
            for steps in range(num_steps):
                proj_inputs = {**data, **x_proj}
                x_proj = proj(proj_inputs)
                X_projected.append(np.concatenate([x_proj['y'].detach().numpy(), x_proj['u'].detach().numpy()], axis=2))
            projected_steps = np.concatenate(X_projected, axis=0)
            proj.num_steps = num_steps

            # plot optimal solutions CasADi vs Neuromancer
            fig, ax = plt.subplots(2)
            ax[0].plot(x_nm[0,:,0], linewidth=lw, label='problem')
            ax[0].plot(trajectories['y'].detach().numpy().reshape(-1,1), linewidth=lw, label='traj')
            ax[0].plot(Ymax, color='black')
            ax[0].plot(Ymin, color='black')
            # plot projected steps
            ax[1].plot(y_nm[0,:,0], linewidth=lw, label='problem')
            ax[1].plot(trajectories['u'].detach().numpy().reshape(-1,1), linewidth=lw, label='traj')
            # ax[1].legend()
            for i in range(projected_steps.shape[0]):
            # for i in range(5):
            # i = 15
                ax[0].plot(projected_steps[i,:, 0], label='projection steps', zorder=-1)
                ax[1].plot(projected_steps[i,:, 1], label='projection steps', zorder=-1)
            plt.savefig(self.saveDir+'/grad_proj_hvac', dpi = fig.dpi)


            x_nm = model_out['test_' + "stored"].detach().numpy()
            y_nm = model_out['test_' + "u_bat"].detach().numpy()
            # intermediate projection steps
            sol_map = self.problem.nodes[0]
            proj = self.problem.nodes[2]
            num_steps = proj.num_steps
            x_proj = sol_map(data)
            proj.num_steps = 1    # set projections steps to 1 for visualisation
            X_projected = [np.concatenate([x_proj['stored'].detach().numpy(), x_proj['u_bat'].detach().numpy()],axis=2)]
            for steps in range(num_steps):
                proj_inputs = {**data, **x_proj}
                x_proj = proj(proj_inputs)
                X_projected.append(np.concatenate([x_proj['stored'].detach().numpy(), x_proj['u_bat'].detach().numpy()], axis=2))
            projected_steps = np.concatenate(X_projected, axis=0)
            proj.num_steps = num_steps

            # plot optimal solutions CasADi vs Neuromancer
            fig, ax = plt.subplots(2)
            ax[0].plot(x_nm[0,:,0], linewidth=lw, label='problem')
            ax[0].plot(trajectories['stored'].detach().numpy().reshape(-1,1), linewidth=lw, label='traj')
            ax[0].plot(trajectories['batRef'].detach().cpu().reshape(nsteps_test, 1), color='black')
            # plot projected steps
            ax[1].plot(y_nm[0,:,0], linewidth=lw, label='problem')
            ax[1].plot(trajectories['u_bat'].detach().numpy().reshape(-1,1), linewidth=lw, label='traj')
            # ax[1].legend()
            for i in range(projected_steps.shape[0]):
            # for i in range(5):
            # i = 15
                ax[0].plot(projected_steps[i,:, 0], label='projection steps', zorder=-1)
                ax[1].plot(projected_steps[i,:, 1], label='projection steps', zorder=-1)
            plt.savefig(self.saveDir+'/grad_proj_bat', dpi = fig.dpi)

        # Plot training and validation loss
        self.PlotLoss()

    def PrepareDataset(self, dataset, xmin_range, name='train'):
        '''
        Prepare data for training and testing of controller model

        :param dataset: (dict) normalized dataset split into states (X), information inputs (I), and disturbances (D)

        :return: (DataLoader) DataLoader object with batched dataset
        '''
        # sampled references for training the policy
        batched_xmin = xmin_range.sample((self.n_samples, 1, self.nref)).repeat(1, self.nsteps, 1)
        batched_range = torch.tensor(self.rng.uniform(low=1.0, high = 8.0, size=(self.n_samples, 1, self.nref)), dtype=torch.float32, device=self.device).repeat(1, self.nsteps, 1)
        batched_range = self.norm.normDelta(batched_range, keys=['y'])
        batched_xmax = batched_xmin + batched_range

        bat_range = torch.distributions.Uniform(0.2 * self.batSize, 0.8 * self.batSize)
        batched_batRef = bat_range.sample((self.n_samples, 1, self.nref)).repeat(1, self.nsteps, 1)
        # batched_batMax = torch.tensor(np.ones((self.n_samples,self.nsteps+1,1))*0.9*self.batSize, dtype=torch.float32, device=self.device)
        batched_batMax = torch.clamp(batched_batRef + torch.tensor(self.rng.uniform(low=0.1*self.batSize, high=0.8*self.batSize, size=(self.n_samples, 1, self.nref)), dtype=torch.float32, device=self.device), min=None, max=self.batSize)

        batched_dr = torch.tensor(self.rng.uniform(low=-1, high=1, size=(self.n_samples, 1, 1)), dtype=torch.float32, device=self.device).repeat(1, self.nsteps, 1)
        
        # sampled disturbance trajectories from simulation model
        temp_d = []
        temp_i = []
        for _ in range(self.n_samples):
            start_idx = self.rng.integers(0, len(dataset['D'])-1-self.nsteps)
            temp_d.append(torch.tensor(self.Get_D(dataset['D'], self.nsteps, start_idx),
                                      dtype=torch.float32, device=self.device))
            temp_i.append(torch.tensor(self.Get_D(dataset['I'], self.nsteps, start_idx),
                                        dtype=torch.float32, device=self.device))
        batched_dist = torch.stack(temp_d)
        batched_price = torch.stack(temp_i)

        # sampled initial conditions
        batched_x0 = torch.stack([torch.tensor(self.Get_X0(dataset['X']),
                                               dtype=torch.float32, device=self.device).unsqueeze(0)
                                               for _ in range(self.n_samples)])
        batched_stored0 = torch.tensor(self.rng.uniform(0,self.batSize,[self.n_samples,1,1]), dtype=torch.float32, device=self.device)
        batched_hvacPower = torch.zeros_like(batched_x0)
        batched_batPower = torch.zeros_like(batched_x0)

        data = DictDataset(
            {"xn": batched_x0,
            "y": batched_x0[:,:,self.y_idx],
            "ymin": batched_xmin,
            "ymax": batched_xmax,
            "d": batched_dist,
            "cost": batched_price,
            'dr': batched_dr,
            "stored": batched_stored0,
            "batRef": batched_batRef,
            "batMax": batched_batMax,
            "hvacPower": batched_hvacPower,
            "batPower": batched_batPower},
            name=name,
        )
        print(f'-----{name}-----')
        for key, value in data.datadict.items():
            print(key, value.shape)

        return DataLoader(data, batch_size=self.batch_size, collate_fn=data.collate_fn, shuffle=False)

    def Get_X0(self, data):
        '''
        Randomly samples state data to create a series of initial states

        :param data: (ndarray/tensor) state data

        :return: (ndarray) array of initial states
        '''
        # brackets because there is only one state currently
        return np.array(self.rng.uniform(low=np.min(data, axis=0), high=np.max(data, axis=0)))

    def Get_D(self, data, nsim, start_idx):
        '''
        Samples a slice of disturbance data

        :param data: (ndarray/tensor) disturbance data
        :param nsim: (int) length of slice to sample
        :param start_idx: (int) index to start slice at

        :return: Selected slice of disturbance data
        '''
        return data[start_idx:start_idx+nsim, :]
    
class BatteryModel(nn.Module):
    def __init__(self, eff, capacity, nx, nu):
        super().__init__()
        self.nx = nx
        self.nu = nu
        self.in_features = nx+nu
        self.out_features = nx
        self.dt = 60

        self.eff = eff
        self.capacity = capacity

    def forward(self, x, u):
        assert len(x.shape) == 2
        assert len(u.shape) == 2

        x = x + self.eff * u / self.dt
        x = torch.clamp(x, self.capacity*0.2, self.capacity)

        return x
    
class batCorrect(nn.Module):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity

    def forward(self, stored, u_bat):
        assert len(stored.shape) == 2
        assert len(u_bat.shape) == 2

        return torch.where((stored >= self.capacity) | (stored <= self.capacity*0.2), torch.zeros_like(u_bat), u_bat)

        # if x.shape[0] == 1:
        #     return torch.ones_like(x)
        # else:   
        #     return torch.gradient(x, dim=0)

        # if x.shape[0] == 1:
        #     deltaX = (x - self.xPrev) * 60 * 1/0.95
        #     self.xPrev = x[-1].item()
        # else:
        #     deltaX = torch.zeros_like(x)
        #     deltaX[1:] = (x[1:] - x[:-1]) * 60 * 1/0.95
        #     self.xPrev = x[-1].item()
        # return torch.clamp(deltaX, min=-9.6, max=9.6)

class iMODE(blocks.Block):
    def __init__(self, insize, outsize, bias=True, linear_map=slim.Linear, nonlin=nn.Softplus, hsizes=[64], linargs=dict()):
        '''
        Constructor
        Define model topology
        '''
        super().__init__()
        self.in_features, self.out_features = insize, outsize
        self.nhidden = len(hsizes)
        sizes = [insize] + hsizes + [outsize]
        self.nonlin = nn.ModuleList(
            [nonlin() for k in range(self.nhidden)] + [nn.Identity()]
        )
        self.linear = nn.ModuleList(
            [linear_map(sizes[k], sizes[k+1], baias=bias, **linargs) for k in range(self.nhidden + 1)]
        )

    def block_eval(self, x):
        '''
        Apply inputs through model layers
        '''
        first = True
        for lin, nlin in zip(self.linear, self.nonlin):
            if first:
                x = nlin(lin(x))
                first = False
                continue
            xin = x
            x = nlin(lin(x))
            x = torch.cat([xin,x],1)
        return x

# Custom debug and callback code
class Callback_Basic(Callback):
    '''
    Basic callback class that plots loss over training epochs

    Attributes:
        debugLevel: (DebugLevel or int) level of debug used in training
        loss: (list) various loss values over training epochs
    '''
    def __init__(self, debugLevel):
        '''
        Constructor

        :param debugLevel: (DebugLevel or int) level of debug used in training
        '''
        self.debugLevel = debugLevel

        # Monitoring loss by epoch
        self.loss = []

    def end_eval(self, trainer, output):
        if self.debugLevel < DebugLevel.EPOCH_LOSS:
            return
        self.loss.append(output)

class DebugLevel(Enum):
    '''
    Enum used for setting debug level of training

    0: No debug files are created
    1: Diagrams of the models and optimizations are created
    2: Loss over training epochs is plotted
    3: Values over training epochs is plotted
    '''
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
    '''
    Custom callback class to monitor NODE training
    '''
    def __init__(self, debugLevel, savePath):
        self.debugLevel = debugLevel

        # Monitoring loss by epoch
        self.loss = []

        # Monitoring values by epoch
        self.savePath = savePath
        self.ResetTrainValues()

    def ResetTrainValues(self):
        self.trainValues = {'train_u': [], 'train_y': [], 'train_yhat': []}

    def end_batch(self, trainer, output):
        if self.debugLevel < DebugLevel.EPOCH_VALUES:
            return
        
        self.trainValues['train_u'].append(output['train_u'].detach().cpu().flatten().numpy())
        self.trainValues['train_y'].append(output['train_y'].detach().cpu().flatten().numpy())
        self.trainValues['train_yhat'].append(output['train_x'].detach().cpu().flatten().numpy())

    def begin_epoch(self, trainer, output):
        if self.debugLevel < DebugLevel.EPOCH_VALUES:
            return
        train_u = np.array(self.trainValues['train_u'][:-1]).flatten()
        train_y = np.array(self.trainValues['train_y'][:-1]).flatten()
        train_yhat = np.array(self.trainValues['train_yhat'][:-1]).flatten()
        fig, ax = plt.subplots(2, figsize=(10,8))
        ax[0].plot(train_u)
        ax[0].set(ylim=[0,1.1], title='u', xlim=[0,2000])
        ax[1].plot(train_y)
        ax[1].plot(train_yhat)
        ax[1].set(ylim=[0,1.1], title='y', xlim=[0,2000])
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

class Callback_Controller(Callback):
    '''
    Custom callback class to monitor controller training
    '''
    def __init__(self, debugLevel, savePath):
        self.debugLevel = debugLevel

        # Monitoring loss by epoch
        self.loss = []

        # Monitoring values by epoch
        self.savePath = savePath
        self.ResetTrainValues()

    def ResetTrainValues(self):
        self.trainValues = {'train_u': [], 'train_y': [], 'train_ymin': [], 'train_ymax': [],
                            'train_ubat': [], 'train_stored': [], 'train_batRef': [], 'train_batMax': []}
        self.lossValues = {'bat_max': [], 'bat_max_value': [], 'bat_max_violation': []}

    def end_batch(self, trainer, output):
        if self.debugLevel < DebugLevel.EPOCH_VALUES:
            return

        # for key, value in output.items():
        #     try:
        #         print(key, value.shape)
        #     except:
        #         print(key)
        # input('paused')

        self.trainValues['train_u'].append(output['train_u'].detach().cpu().flatten().numpy())
        self.trainValues['train_y'].append(output['train_y'][:,:-1,:].detach().cpu().flatten().numpy())
        self.trainValues['train_ymin'].append(output['train_ymin'][:,:-1,:].detach().cpu().flatten().numpy())
        self.trainValues['train_ymax'].append(output['train_ymax'][:,:-1,:].detach().cpu().flatten().numpy())

        self.trainValues['train_ubat'].append(output['train_u_bat'][:,:,:].detach().cpu().flatten().numpy())
        self.trainValues['train_stored'].append(output['train_stored'][:,:-1,:].detach().cpu().flatten().numpy())
        self.trainValues['train_batRef'].append(output['train_batRef'][:,:-1,:].detach().cpu().flatten().numpy())
        self.trainValues['train_batMax'].append(output['train_batMax'][:,:-1,:].detach().cpu().flatten().numpy())

        # self.lossValues['bat_max'].append(output['train_stored_lt_batMax'].detach().cpu().flatten().numpy())
        # self.lossValues['bat_max_value'].append(output['train_stored_lt_batMax_value'].detach().cpu().flatten().numpy())
        # self.lossValues['bat_max_violation'].append(output['train_stored_lt_batMax_violation'].detach().cpu().flatten().numpy())

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
        ax[1].set(ylim=[0,1.1], title='y', xlim=[0,2000])
        plt.figtext(0.01, 0.01, f"Epoch: {output['train_epoch']}", fontsize=14)
        plt.tight_layout()
        Path(f'{self.savePath}/debug/HVAC').mkdir(exist_ok=True, parents=True)
        fig.savefig(f"{self.savePath}/debug/HVAC/epoch{output['train_epoch']}")
        plt.close(fig)

        train_ubat = np.array(self.trainValues['train_ubat']).flatten()
        train_stored = np.array(self.trainValues['train_stored']).flatten()
        train_batRef = np.array(self.trainValues['train_batRef']).flatten()
        train_batMax = np.array(self.trainValues['train_batMax']).flatten()
        fig, ax = plt.subplots(2, figsize=(10,8))
        ax[0].plot(train_ubat)
        ax[0].set(title='u', xlim=[0,2000])
        ax[1].plot(train_stored)
        ax[1].plot(train_batRef, '--', c='k', zorder=-1)
        ax[1].plot(train_batMax, '--', c='k', zorder=-1)
        ax[1].set(title='y', xlim=[0,2000])
        plt.figtext(0.01, 0.01, f"Epoch: {output['train_epoch']}", fontsize=14)
        plt.tight_layout()
        Path(f'{self.savePath}/debug/Battery').mkdir(exist_ok=True, parents=True)
        fig.savefig(f"{self.savePath}/debug/Battery/epoch{output['train_epoch']}")
        plt.close(fig)

        self.ResetTrainValues()

    def end_eval(self, trainer, output):
        if self.debugLevel < DebugLevel.EPOCH_LOSS:
            return
        lossDict= {key: output[key] for key in ['train_loss', 'dev_loss']}
        self.loss.append(lossDict)

    def end_train(self, trainer, output):
        if self.debugLevel < DebugLevel.EPOCH_VALUES:
            return
        ims = [imageio.v3.imread(f"{self.savePath}/debug/HVAC/epoch{i}.png") for i in range(len(os.listdir(self.savePath+'/debug/HVAC/')))]
        imageio.mimwrite(self.savePath+'/HVAC_train.gif', ims, fps=10)
        ims = [imageio.v3.imread(f"{self.savePath}/debug/Battery/epoch{i}.png") for i in range(len(os.listdir(self.savePath+'/debug/Battery/')))]
        imageio.mimwrite(self.savePath+'/battery_train.gif', ims, fps=10)
        # ims = [imageio.v3.imread(f"{self.savePath}/debug/Loss/epoch{i}.png") for i in range(len(os.listdir(self.savePath+'/debug/Loss/')))]
        # imageio.mimwrite(self.savePath+'/loss.gif', ims, fps=10)
        shutil.rmtree(self.savePath+'/debug/')

class Normalizer():
    '''
    Class that handles the normalizing and de-normalizing of a variety of data structures
    
    Attributes:
        dataInfo: (dict{dict}) stored the values needed for norm and de-norm processes for data keys
    '''
    def __init__(self):
        self.dataInfo = {'leave': {'max': 1,
                                   'min': 0,
                                   'mean': 0,
                                   'std': 1}}

    def add_data(self, data, keys: typing.Optional[typing.Iterable[str]] = None):
        '''
        Adds an entry to dataInfo to keep track of input data's descriptors

        :param data: (dict, DictDataset, DataFrame, nparray) data to describe
        :param keys: (list[str]) names to use when storing data descriptors, defaults to None
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

    def norm(self, data, keys: typing.Optional[typing.Iterable[str]] = None):
        '''
        Normalizes a provided dataset based on previously added descriptors

        :param data: (dict, DictDataset, DataFrame, nparray) data to normalize
        :param keys: (list[str]) keys to use when normalizing, defaults to None
        :return: Normalized data with data structure matching input
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
    
    def normDelta(self, data, keys: typing.Iterable[str]):
        '''
        Special case normalizer for delta values
        
        :param data: (ndarray or scalar value) delta value(s) to normalize
        :param keys: list[str] list of length 1 with key to use

        :return: normalized data matching data structure of input
        '''
        stats = self.dataInfo[keys[0]]
        norm_data = data / (stats['max'] - stats['min'])
        return norm_data

    def denorm(self, data, keys: typing.Optional[typing.Iterable[str]] = None):
        '''
        De-normalize a provided dataset based on previously added descriptors

        :param data: (dict, DictDataset, DataFrame, nparray) data to de-normalize
        :param keys: (list[str]) keys to use for de-normalizing, defaults to None

        :return: De-normalized data matching data structure of input
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

        :param filePath: (str) path to the directory the normalizer info should be saved in
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

        :param filePath: (str) path to the directory containing previously saved normalizer info
        '''
        with open(filePath+'norm_dataInfo.json') as fp:
            dataInfoCopy = json.load(fp)
        for key, value in dataInfoCopy.items():
            if type(value['max']) == list:
                for k,v in value.items():
                    value[k] = np.array(v)
        self.dataInfo = dataInfoCopy

def TOUPricing(date, timeSteps=96):
    """
    Gets the time of use pricing depending on the date
    Parameters:
        date: datetime of the desired day
        timeSteps: int of the number of timesteps in a single day (96 for 15 min intervals)
    Returns:
        list of ints of the hourly cost of energy in cents/kWh
    """
    # Prices in cents/kWh
    # Summer: May 1st to September 30th
    summer = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 19, 19, 28, 28, 28, 28, 10, 10, 10, 10, 10]
    
    # Winter: October 1st to April 30th
    winter = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 14, 14, 17, 17, 17, 17, 10, 10, 10, 10, 10]
    
    limit1 = dt.datetime(date.year, 5, 1)
    limit2 = dt.datetime(date.year, 10, 1)
    
    if (date >= limit1) & (date < limit2):
        price = summer
    else:
        price = winter

    # tempList = []
    # for item in price:
    #     for i in range(0,timeSteps//24):
    #         tempList.append(item)
    # return tempList

    return price[date.hour]