import json
import os
import shutil

class RunManager():
    def __init__(self, name, saveDir='savedRuns'):
        '''Constructor'''

        self.name = name
        dirName = os.path.dirname(__file__)
        self.saveDir = os.path.join(dirName, saveDir)
        self.models = {}
        self.dataset = {"path": '', "sliceBool": False, "slice_idx": [0,0]}

        self.tempBounds = []

        
        self.runPath = self.saveDir+'/latestRun/'

        if not(os.path.exists(self.saveDir)):
            print("SaveDir does not exist, creating it")
            os.mkdir(self.saveDir)

    def WriteRunJson(self):
        '''
        Creates the run json file based on the current state of the models
        '''
        # Create dict
        outDict = {}
        outDict['name'] = self.name
        outDict['models'] = self.models
        outDict['dataset'] = self.dataset
        outDict['tempBounds'] = self.tempBounds

        # Write dict to json
        with open(self.runPath+'run.json', '+w') as fp:
            json.dump(outDict, fp)

    def PrepNewRun(self):
        '''
        Prepares the file structure for a new run by moving the most recent run to a new folder if the user wants to save it
        '''
        # Check if previous run needs to be moved
        if os.path.isdir(self.runPath):
            userInput = input("Would you like to save the most recent run? (y/n) ")
            if userInput.lower() == 'y':
                RunManager.SavePreviousRun(self.saveDir)
            else:
                shutil.rmtree(self.runPath)
        
        os.mkdir(self.runPath)

    def LoadRunJson(self, runName='latestRun'):
        '''
        Loads the parameters of a previous run into instance of class object
        Inputs:
            runName: (str) name of the run to load, defaults to "latestRun"
        '''
        runJson = RunManager.ReadRunJson(self.saveDir, runName)

        self.name = runJson['name']
        self.models = runJson['models']
        self.dataset = runJson['dataset']
        self.tempBounds = runJson['tempBounds']

        if runName != 'latestRun':
            self.runPath = self.saveDir+'/'+self.name+'/'

    def LoadModel(self, runName, modelName):
        '''
        Updates model dict with values from a previously trained model
        Inputs:
            runName: (str) name of the run containing the model
            modelName: (str) name of the model to be loaded
        '''
        runJson = RunManager.ReadRunJson(self.saveDir, runName)
        self.models[modelName] = runJson['models'][modelName]

    @staticmethod
    def ReadRunJson(saveDir, runName='latestRun'):
        '''
        Reads a run json file to retrieve the needed weights and models
        Inputs:
            saveDir: (str) name of directory where runs are saved
            runName: (str) name of run to load, defaults to "latestRun"
        Outputs:
            (dict) json file loaded as a dict
        '''
        with open(saveDir+'/'+runName+'/run.json') as fp:
            runJson = json.load(fp)

        return runJson

    @staticmethod
    def SavePreviousRun(saveDir):
        '''
        Reads the json file of the latest run and then renames run folder based on name
        Inputs:
            saveDir: (str) name of directory where runs are saved
        '''
        runJson = RunManager.ReadRunJson(saveDir)
        oldPath = saveDir+'/latestRun'
        newPath = saveDir+'/'+runJson['name']
        if os.path.isdir(newPath):
            userInput = input("There is already a run with this name. Please enter a new name, or leave blank to overwrite existing run. ")
            looping = True
            while looping:
                if userInput.lower() == '':
                    shutil.rmtree(newPath)
                    looping = False
                else:
                    runJson['name'] = userInput
                    newPath = saveDir+'/'+runJson['name']
                    if os.path.isdir(newPath):
                        print('WARNING: There is also already a run with this name.')
                        userInput = input('Please enter a different name, or leave blank to overwrite existing run. ')
                    else:
                        looping = False

        os.rename(oldPath, newPath)

