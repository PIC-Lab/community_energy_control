import helics as h
import json
import logging
import os
import datetime as dt
import pandas as pd

from communityController import CommunityController

with open('simParams.json') as fp:
    simParams = json.load(fp)

# Controller name maps
# 'controller name': 'model name'
controlInMap = { 'indoorAirTemp': 'living space Air Temperature'}
controlOutMap = {'heatingSetpoint': 'heating setpoint', 'coolingSetpoint': 'cooling setpoint'}

# ----- HELICS federate setup -----
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

# Register federate from json
fed = h.helicsCreateCombinationFederateFromConfig(
    os.path.join(os.path.dirname(__file__), "controlFederate.json")
)
federate_name = h.helicsFederateGetName(fed)
logger.info(f"Created federate {federate_name}")

sub_count = h.helicsFederateGetInputCount(fed)
logger.debug(f"\tNumber of subscriptions: {sub_count}")
pub_count = h.helicsFederateGetPublicationCount(fed)
logger.debug(f"\tNumber of publications: {pub_count}")

# Diagnostics to confirm JSON config correctly added the required
# publications, and subscriptions.
subid = {}
for i in range(0, sub_count):
    ipt = h.helicsFederateGetInputByIndex(fed, i)
    sub_name = h.helicsInputGetName(ipt)
    subid[sub_name] = i
    logger.debug(f"\tRegistered subscription---> {sub_name}")

pubid = {}
for i in range(0, pub_count):
    pub = h.helicsFederateGetPublicationByIndex(fed, i)
    pub_name = h.helicsPublicationGetName(pub)
    pubid[pub_name] = i
    logger.debug(f"\tRegistered publication---> {pub_name}")

# Define time parameters of simulation
start_time = dt.datetime.strptime(simParams['start'], "%m/%d/%y")
stepsize = pd.Timedelta(simParams['step'])
duration = pd.Timedelta(simParams['duration'])
end_time = start_time + duration
logger.debug(f"Run period: {start_time} to {end_time}")

# ----- Control setup -----
controller = CommunityController(simParams['controlledAliases'])

# ----- Primary co-simulation loop -----
# Define lists for data collection

# Execute federate and start co-sim
h.helicsFederateEnterExecutingMode(fed)
times = pd.date_range(start_time, freq=stepsize, end=end_time)
for step, current_time in enumerate(times):
    # Update controllers
    for key,value in controlInMap.items():
        controllerList[i].sensorValues[key] = alf_outs[value]

    if simParams['testCase'] == 'base':
        controllerList[i].HVAC_Control(forceMode=-1)
    elif simParams['testCase'] == 'MPC':
        trajectories = controllerList[i].PredictiveControl(step)

    for key,value in controlOutMap.items():
        input_dicts[alias][value] = controllerList[i].actuatorValues[key]

    if simParams['testCase'] == 'MPC':
        alf_outs['Control Effort'] = trajectories['u'][0,0,0].detach().item()
        alf_outs['Predicted Temperature'] = trajectories['y'][0,0,0].detach().item()
        alf_outs['Ymax'] = trajectories['ymax'][0,0,0].detach().item()
        alf_outs['Ymin'] = trajectories['ymin'][0,0,0].detach().item()

# finalize and close the federate
h.helicsFederateDestroy(fed)
h.helicsCloseLibrary()