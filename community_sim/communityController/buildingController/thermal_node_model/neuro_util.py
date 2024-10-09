from enum import Enum
import numbers
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import copy
from pathlib import Path
import typing
from abc import ABC, abstractmethod

from neuromancer.callbacks import Callback
from neuromancer.trainer import Trainer
from neuromancer.loggers import BasicLogger
from neuromancer.dataset import DictDataset

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
    def __init__(self, manager, name, device, debugLevel, saveDir):
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

        self.callback = Callback_Basic(debugLevel, self.saveDir)

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

    def TrainModel(self, dataset, load):
        '''
        Trains building thermal system or loads from file

        :param dataset: (dict) normalized dataset split into states (X), inputs (U), and disturbances (D)
        :param load: (bool) load model from file instead of training
        '''
        # 
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
            print("Model was trained at sufficiently high debug level or loaded from a file. Loss plots will not be created.")
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