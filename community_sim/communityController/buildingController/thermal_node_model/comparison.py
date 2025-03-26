from modelConstructor import BuildingNode, Normalizer
from runManager import RunManager
import torch
import casadi
import l4casadi as l4c
import pandas as pd
from neuromancer.system import System

def Main():
    buildingID = 4
    device = torch.device('cpu')

    run = 'bat_SB_power_1'

    manager = RunManager(run)
    manager.LoadRunJson(run)
    norm = Normalizer()
    norm.load(f"{manager.runPath}norm/{buildingID}/")

    for key in manager.models.keys():
        if key.find('buildingThermal') != -1:
            thermalModelName = key
        elif key.find('classifier') != -1:
            classifierModelName = key
        elif key.find('controller') != -1:
            controllerModelName = key
        else:
            raise ValueError(f"Model name '{key}' does not meet expected naming conventions.")
        
    nsteps = manager.models[controllerModelName]['train_params']['nsteps']

    # Thermal model definition
    # Building thermal model
    initParams = manager.models[thermalModelName]['init_params']
    buildingThermal = BuildingNode(nx=initParams['nx'],
                                    nu=initParams['nu'],
                                    nd=initParams['nd'],
                                    manager=manager,
                                    name=thermalModelName,
                                    device=device,
                                    debugLevel = 0,
                                    saveDir=f"{manager.runPath+thermalModelName}/{buildingID}")
    buildingThermal.CreateModel()

    buildingThermal.TrainModel(dataset=None, load=True, test=False)
    buildingThermal.nsteps = 1440
    buildingThermal.model.nsteps = 1440

    testTrajectories = pd.read_csv('Saved Figures/trajectories.csv')

    pytorchModel = NeuromancerNodeWrapper(buildingThermal.model)

    l4c_model = l4c.L4CasADi(pytorchModel)

    # construct casadi problem
    opti, u, y = NLP_param(l4c_model,
                           dr=testTrajectories['dr'],
                           cost=testTrajectories['cost'],
                           ymin=testTrajectories['ymin'],
                           ymax=testTrajectories['ymax'])
    # solve NLP via casadi
    sol = opti.solve()

# instantiate casadi optimizaiton problem class
def NLP_param(l4c_model, dr, cost, ymin, ymax, opti_silent=False):
    opti = casadi.Opti()
    # define variables
    u = opti.variable(1440,1)
    y = opti.variable(1440,1)
    dr_opti = opti.parameter(1440,1)
    cost_opti = opti.parameter(1440,1)
    ymin_opti = opti.parameter(1440,1)
    ymax_opti = opti.parameter(1440,1)
    d = opti.parameter(1440,1)

    x = casadi.horzcat(y,u,d)
    print(x.shape)

    # define objective and constraints
    opti.minimize(6 * (dr_opti.T @ u) + 5 * (cost_opti.T @ u))
    opti.subject_to(ymin_opti <= y)
    opti.subject_to(ymax_opti >= y)
    opti.subject_to(l4c_model(x) == y)

    # select IPOPT solver and solve the NLP
    if opti_silent:
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    else:
        opts = {}
    opti.solver('ipopt', opts)
    # set parametric values
    opti.set_value(dr_opti, dr)
    opti.set_value(cost_opti, cost)
    opti.set_value(ymin_opti, ymin)
    opti.set_value(ymax_opti, ymax)
    return opti, u, y

class NeuromancerNodeWrapper(System):
    def __init__(self, model):
        super().__init__(nodes=model.nodes,
                       name=model.name,
                       nstep_key=model.nstep_key,
                       init_func=model.init,
                       nsteps=model.nsteps)

    def forward(self, x):
        input_dict = {'xn': x[:,0:1].unsqueeze(0), 'u': x[:,1:2].unsqueeze(0), 'd': x[:,2:3].unsqueeze(0)}
        data = super().forward(input_dict)
        return data['y'][0,:,:]

if __name__ == '__main__':
    Main()