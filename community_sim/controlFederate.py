import json
import logging
import os
import datetime as dt
import pandas as pd
import numpy as np

from communityController.communityController import CommunityController

import helics as h          # Importing helics before torch will cause a segfault for some reason

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

with open('simParams.json') as fp:
    simParams = json.load(fp)

with open('indexMapping.json') as fp:
    sensorIdxMapping = json.load(fp)        # Map sensor indices to simulation indices
simIdxMapping = {v: k for k, v in sensorIdxMapping.items()}     # Map simulation indices to sensor indices

# Controller name maps
# 'controller name': 'model name'
controlInMap = {'indoorAirTemp': 'living space Air Temperature'}
controlOutMap = {'heatingSetpoint': 'heating setpoint', 'coolingSetpoint': 'cooling setpoint',
                 'battery': 'Battery'}
sensorNames = ['indoorAirTemp', 'batterySOC']

# ----- HELICS federate setup -----
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
    sub_name = sub_name[sub_name.find('/')+1:]
    subid[sub_name] = ipt
    logger.debug(f"\tRegistered subscription---> {sub_name}")

pubid = {}
for i in range(0, pub_count):
    pub = h.helicsFederateGetPublicationByIndex(fed, i)
    pub_name = h.helicsPublicationGetName(pub)
    pubid[pub_name] = pub
    logger.debug(f"\tRegistered publication---> {pub_name}")

# Define time parameters of simulation
start_time = dt.datetime.strptime(simParams['start'], "%m/%d/%y")
stepsize = pd.Timedelta(simParams['step'])
duration = pd.Timedelta(simParams['duration'])
end_time = start_time + duration
logger.debug(f"Run period: {start_time} to {end_time}")

# ----- Control setup -----
aliasesSensorIdx = [simIdxMapping[alias] for alias in simParams['controlledAliases']]       # Convert list of controlled buildings from sim idx to sensor idx
controller = CommunityController(aliasesSensorIdx)

# ----- Primary co-simulation loop -----
# Define lists for data collection
outputs = {alias: [] for alias in aliasesSensorIdx}

# Execute federate and start co-sim
h.helicsFederateEnterExecutingMode(fed)
times = pd.date_range(start_time, freq=stepsize, end=end_time)
for step, current_time in enumerate(times):
    # Update time in co-simulation
    present_step = (current_time - start_time).total_seconds()
    present_step += 1
    h.helicsFederateRequestTime(fed, present_step)

    # get signals from other federate
    logger.debug(f"Current time: {current_time}, step: {step}")
    isupdated = h.helicsInputIsUpdated(subid['battery_soc'])
    if isupdated == 1:
        batterySOC = h.helicsInputGetString(subid['battery_soc'])
        batterySOC = json.loads(batterySOC)
        logger.debug("Recieved updated value for battery_soc")
        logger.debug(batterySOC)
    else:
        batterySOC = {}

    isupdated = h.helicsInputIsUpdated(subid['indoor_temp'])
    if isupdated == 1:
        indoorTemp = h.helicsInputGetString(subid['indoor_temp'])
        indoorTemp = json.loads(indoorTemp)
        logger.debug("Recieved updated value for indoor_temp")
        logger.debug(indoorTemp)
    else:
        indoorTemp = {}
    
    sensorValues = {alias: {} for alias in aliasesSensorIdx}
    for key, value in batterySOC.items():
        sensorValues[simIdxMapping[key]]['batterySOC'] = value
    for key, value in indoorTemp.items():
        if key in simParams['controlledAliases']:
            sensorValues[simIdxMapping[key]]['indoorTemp'] = value

    logger.debug(sensorValues)

    controlEvents = controller.Step(sensorValues, current_time)

    # Map actuator values to control event format
    input_dicts = []
    for event in controlEvents:
        tempDict = {}
        tempDict['location'] = sensorIdxMapping[event['location']]
        tempDict['devices'] = {}
        for key,value in controlOutMap.items():
            if key == 'batteryState':
                tempDict['devices'][value+tempDict['location']] = event['devices'][key]
            else:
                tempDict['devices'][value] = event['devices'][key]
        input_dicts.append(tempDict)

    for alias in aliasesSensorIdx:
        controlTraj = {}
        controlTraj['Time'] = current_time
        controlTraj['Control Effort'] = controller.trajectories['u'][0,0,0].detach().item()
        controlTraj['Predicted Temperature'] = controller.trajectories['y'][0,0,0].detach().item()
        controlTraj['Ymax'] = controller.trajectories['ymax'][0,0,0].detach().item()
        controlTraj['Ymin'] = controller.trajectories['ymin'][0,0,0].detach().item()
        controlTraj['dr'] = controller.trajectories['dr'][0,0,0].detach().item()
        outputs[alias].append(controlTraj)

    logger.debug("Publishing values to other federates")
    h.helicsPublicationPublishString(pubid['control_events'], json.dumps(input_dicts))
    logger.debug(input_dicts)

# Put output lists in dataframes
outputs_df = []
for key, building in outputs.items():
    df = pd.DataFrame(building)
    col = df.pop('Time')
    df.insert(0, col.name, col)
    outputs_df.append(df)

# Save data to csv
aggregateLoad = np.zeros(len(outputs_df[0]))
for i, building in enumerate(outputs_df):
    building.to_csv('results/'+simParams['controlledAliases'][i]+'_control.csv', index=False)

# finalize and close the federate
h.helicsFederateDestroy(fed)
h.helicsCloseLibrary()