import json
import logging
import os
import datetime as dt
import pandas as pd
import numpy as np

from communityController.communityController import CommunityController

import helics as h          # Importing helics before torch will cause a segfault for some reason

initTime = dt.datetime.now()

with open('simParams.json') as fp:
    simParams = json.load(fp)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(simParams['logLevel'])

logger.info("----- Control Federate Logs -----")
logger.info(f"Started at {initTime}")
logger.info(f"Simulation results will be saved to {simParams['resultsDir']}")

with open('indexMapping.json') as fp:
    sensorIdxMapping = json.load(fp)        # Map sensor indices to simulation indices
simIdxMapping = {v: k for k, v in sensorIdxMapping.items()}     # Map simulation indices to sensor indices

# Controller name maps
# 'controller name': 'model name'
controlInMap = {'indoorAirTemp': 'living space Air Temperature'}
controlOutMap = {'heatingSetpoint': 'heating setpoint', 'coolingSetpoint': 'cooling setpoint',
                 'battery': 'Battery'}

# ----- HELICS federate setup -----
# Register federate from json
fed = h.helicsCreateCombinationFederateFromConfig(
    os.path.join(os.path.dirname(__file__), "controlFederate.json")
)
federate_name = h.helicsFederateGetName(fed)
logger.info(f"Created federate {federate_name}")

sub_count = h.helicsFederateGetInputCount(fed)
logger.info(f"\tNumber of subscriptions: {sub_count}")
pub_count = h.helicsFederateGetPublicationCount(fed)
logger.info(f"\tNumber of publications: {pub_count}")

# Diagnostics to confirm JSON config correctly added the required
# publications, and subscriptions.
subid = {}
for i in range(0, sub_count):
    ipt = h.helicsFederateGetInputByIndex(fed, i)
    sub_name = h.helicsInputGetName(ipt)
    sub_name = sub_name[sub_name.find('/')+1:]
    subid[sub_name] = ipt
    logger.info(f"\tRegistered subscription---> {sub_name}")

pubid = {}
for i in range(0, pub_count):
    pub = h.helicsFederateGetPublicationByIndex(fed, i)
    pub_name = h.helicsPublicationGetName(pub)
    pubid[pub_name] = pub
    logger.info(f"\tRegistered publication---> {pub_name}")

# Define time parameters of simulation
start_time = dt.datetime.strptime(simParams['start'], "%m/%d/%y %H:%M")
stepsize = pd.Timedelta(simParams['step'])
duration = pd.Timedelta(simParams['duration'])
end_time = start_time + duration
logger.info(f"Run period: {start_time} to {end_time}")

# ----- Control setup -----
aliasesSensorIdx = [simIdxMapping[alias] for alias in simParams['controlledAliases']]       # Convert list of controlled buildings from sim idx to sensor idx
controller = CommunityController(aliasesSensorIdx, simParams['controllerRun'])

# ----- Primary co-simulation loop -----
# Define lists for data collection
outputs = {alias: [] for alias in aliasesSensorIdx}
coord_out = {alias: [] for alias in aliasesSensorIdx}

# Execute federate and start co-sim
first = True
h.helicsFederateEnterExecutingMode(fed)
times = pd.date_range(start_time, freq=stepsize, end=end_time)
try:
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
            try:
                sensorValues[simIdxMapping[key[key.index('y')+1:]]]['batterySOC'] = value
            except KeyError:
                logger.debug(f"No matching key found for {key}")
        for key, value in indoorTemp.items():
            if key in simParams['controlledAliases']:
                sensorValues[simIdxMapping[key]]['indoorAirTemp'] = value

        logger.debug(sensorValues)

        controlEvents = controller.Step(sensorValues, current_time)

        # Map actuator values to control event format
        input_dicts = []
        for event in controlEvents:
            tempDict = {}
            tempDict['location'] = sensorIdxMapping[event['location']]
            tempDict['devices'] = {}
            for key,value in controlOutMap.items():
                tempDict['devices'][value] = event['devices'][key]
            input_dicts.append(tempDict)

        for alias in aliasesSensorIdx:
            if first:
                predictedTraj = {}
                predictedTraj[alias] = controller.trajectoryList[alias]['horizon_stored'][0,:,0].detach().numpy()
                first = False

            controlTraj = {}
            controlTraj['Time'] = current_time
            controlTraj['Control Effort'] = controller.trajectoryList[alias]['horizon_u'][0,0,0].detach().item()
            controlTraj['Predicted Temperature'] = controller.trajectoryList[alias]['horizon_y'][0,0,0].detach().item()
            controlTraj['Ymax'] = controller.trajectoryList[alias]['horizon_ymax'][0,0,0].detach().item()
            controlTraj['Ymin'] = controller.trajectoryList[alias]['horizon_ymin'][0,0,0].detach().item()
            controlTraj['dr'] = controller.trajectoryList[alias]['horizon_dr'][0,0,0].detach().item()
            controlTraj['cost'] = controller.trajectoryList[alias]['horizon_cost'][0,0,0].detach().item()
            controlTraj['stored'] = controller.trajectoryList[alias]['horizon_stored'][0,0,0].detach().item()
            controlTraj['u_bat'] = controller.trajectoryList[alias]['horizon_u_bat'][0,0,0].detach().item()
            controlTraj['bat_ref'] = controller.trajectoryList[alias]['horizon_batRef'][0,0,0].detach().item()
            controlTraj['hvacPower'] = controller.trajectoryList[alias]['horizon_hvacPower'][0,0,0].detach().item()
            controlTraj['batPower'] = controller.trajectoryList[alias]['horizon_batPower'][0,0,0].detach().item()
            outputs[alias].append(controlTraj)

            coordDict = {}
            coordDict['Time'] = current_time
            coordDict['usagePenalty'] = controller.coordDebug[alias]['usagePenalty']
            coordDict['flexLoad'] = controller.coordDebug[alias]['flexLoad']
            coord_out[alias].append(coordDict)

        logger.debug("Publishing values to other federates")
        h.helicsPublicationPublishString(pubid['control_events'], json.dumps(input_dicts))
        logger.debug(input_dicts)
except KeyboardInterrupt:
    print('Keyboard interrupt received. Stopping simulation and saving current data.')

# Put output lists in dataframes
outputs_df = []
for key, building in outputs.items():
    df = pd.DataFrame(building)
    col = df.pop('Time')
    df.insert(0, col.name, col)
    # print(len(df), len(predictedTraj[key]))
    # df['PredStored'] = np.pad(predictedTraj[key], (0,len(df) - len(predictedTraj[key])))
    outputs_df.append(df)

coord_df = []
for key, building in coord_out.items():
    df = pd.DataFrame(building)
    col = df.pop('Time')
    df.insert(0, col.name, col)
    coord_df.append(df)

# Save data to csv
aggregateLoad = np.zeros(len(outputs_df[0]))
for i, building in enumerate(outputs_df):
    building.to_csv('results/'+simParams['controlledAliases'][i]+'_control.csv', index=False)
for i, building in enumerate(coord_df):
    building.to_csv('results/'+simParams['controlledAliases'][i]+'_coord.csv', index=False)

# finalize and close the federate
h.helicsFederateDestroy(fed)
h.helicsCloseLibrary()