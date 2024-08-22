import sys

from buildingController import BuildingController

class CommunityController:
    '''
    '''
    def __init__(self, controlAliasList, mode):
        '''
        '''

        # Import modules
        if mode == 'sim':
            sys.path.insert(1, "../flexibility_metrics")
            sys.path.insert(1, "../coordinator")
        from flexibilityMetrics import FlexibilityPredictor     # type:ignore
        from community_optim import Coordinator     # type:ignore

        self.controlAliasList = controlAliasList
        self.mode = mode

        self.controllerList = []
        self.flexibilityList = []

    def CoordinatorInit(self):
        '''
        '''
        pass

    def ControllerInit(self):
        '''
        '''
        if self.mode == 'deploy':
            

        # Create controller object
        controllerList = []
        for alias in self.controlAliasList:
            controllerList.append(BuildingController(alias, devices, self.mode, 'MPC'))

    def FlexibilityInit(self):
        '''
        '''
        pass

    def Step(self):
        '''
        Run every time step
        Returns a list of dicts containing the control events for each house. Matches controls API format except for device IDs
        '''
        pass