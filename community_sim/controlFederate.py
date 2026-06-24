import json
import logging
import os
import datetime as dt
import pandas as pd
import numpy as np

from communityController.communityController import CommunityController

import helics as h          # Importing helics before torch will cause a segfault for some reason

initTime = dt.datetime.now()

with open('configs/simParams.json') as fp:
    simParams = json.load(fp)

with open('communityController/coordinator/transInfo.json') as fp:
    transInfo = json.load(fp)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)
logger.setLevel(simParams['logLevel'])

logger.info("----- Control Federate Logs -----")
logger.info(f"Started at {initTime}")
logger.info(f"Simulation results will be saved to {simParams['resultsDir']}")

MainDir = os.path.abspath(os.path.dirname(__file__))
ModelDir = os.path.join(MainDir, 'network_model')
BuildingDir = os.path.join(MainDir, 'building_models')
ResultsDir = os.path.join(MainDir, simParams['resultsDir'])
os.makedirs(ResultsDir, exist_ok=True)

if simParams['resultLevel']== 'ROLLOUT':
    ExtrasDir = os.path.join(MainDir, simParams['resultsDir']+'/extra/')
    os.makedirs(ExtrasDir, exist_ok=True)

with open('configs/indexMapping.json') as fp:
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
    os.path.join(os.path.dirname(__file__), "configs/controlFederate.json")
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

# base_load_data = pd.read_csv('sim_schedules/base_load.csv')
base_load_data = pd.read_csv('sim_schedules/base_load_ws.csv')
base_load = base_load_data.loc[:,simParams['controlledAliases']].rename(simIdxMapping, axis=1)
logger.debug(f"base_load: {base_load.columns}")


# trans_base_load = np.zeros((len(transInfo), int(base_load_data.shape[0] / simParams['stepSizeCoord'])))
# i = 0
# for key, value in transInfo.items():
#     trans_idx = [x for x in value['SimBuildings'] if x not in simParams['controlledAliases']]
#     logger.debug(f"{key}: {trans_idx}")
#     temp = base_load_data.loc[:,trans_idx].sum(axis=1)
#     trans_base_load[i,:] = temp.values.reshape(int(temp.shape[0] / simParams['stepSizeCoord']),simParams['stepSizeCoord']).mean(axis=1)
#     i += 1
trans_base_load = pd.read_csv('sim_schedules/transBase.csv').values
trans_base_load = trans_base_load.reshape(int(trans_base_load.shape[0] / simParams['stepSizeCoord']),simParams['stepSizeCoord'],trans_base_load.shape[1]).mean(axis=1).T
logger.debug(f"trans_base_load shape: {trans_base_load.shape}")

# ----- Control setup -----
aliasesSensorIdx = [simIdxMapping[alias] for alias in simParams['controlledAliases']]       # Convert list of controlled buildings from sim idx to sensor idx
controller = CommunityController(controlAliasList=aliasesSensorIdx,
                                 runName=simParams['controllerRun'],
                                 logger=logger,
                                 baseLoad=base_load,
                                 testCase=simParams['testCase'],
                                 deploy=False,
                                 nstepsCoord=simParams['nstepsCoord'],
                                 stepSizeCoord=simParams['stepSizeCoord'],
                                 nstepsBuild=simParams['nstepsBuild'],
                                 stepSizeBuild=simParams['stepSizeBuild'])
controller.baseTransLoad = trans_base_load

# ----- Primary co-simulation loop -----
# Define lists for data collection
outputs = {alias: [] for alias in aliasesSensorIdx}
coord_out = {alias: [] for alias in aliasesSensorIdx}
coord_out['gen'] = []

if simParams['resultLevel'] == 'ROLLOUT':
    tempRollout = {key: [] for key in aliasesSensorIdx}
    hvacRollout = {key: [] for key in aliasesSensorIdx}
    batPowerRollout = {key: [] for key in aliasesSensorIdx}
    batSOCRollout = {key: [] for key in aliasesSensorIdx}
    flexLoadRollout = {key: [] for key in aliasesSensorIdx}
    predLoadRollout = {key: [] for key in aliasesSensorIdx}
    usageRollout = {key: [] for key in aliasesSensorIdx}
    predTransRollout = {str(i+1): [] for i in range(0,len(transInfo.keys()))}

# Execute federate and start co-sim
logger.debug("Before federate execute")
first = True
h.helicsFederateEnterExecutingMode(fed)
times = pd.date_range(start_time, freq=stepsize, end=end_time)
stepTime = []
try:
    for step, current_time in enumerate(times):
        # Update time in co-simulation
        present_step = (current_time - start_time).total_seconds()
        present_step += 1
        h.helicsFederateRequestTime(fed, present_step)
        stepStart = dt.datetime.now()

        # get signals from other federate
        logger.info(f"Current time: {current_time}, step: {step}")
        isupdated = h.helicsInputIsUpdated(subid['battery_soc'])
        if isupdated == 1:
            batterySOC = h.helicsInputGetString(subid['battery_soc'])
            batterySOC = json.loads(batterySOC)
            logger.debug("Received updated value for battery_soc")
            logger.debug(batterySOC)
        else:
            batterySOC = {}

        isupdated = h.helicsInputIsUpdated(subid['indoor_temp'])
        if isupdated == 1:
            indoorTemp = h.helicsInputGetString(subid['indoor_temp'])
            indoorTemp = json.loads(indoorTemp)
            logger.debug("Received updated value for indoor_temp")
            logger.debug(indoorTemp)
        else:
            indoorTemp = {}
        
        logger.debug("Updating sensor values")
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

        logger.debug("Controller step")
        controlEvents = controller.Step(sensorValues, current_time)

        # Map actuator values to control event format
        logger.debug("Updating input_dict")
        input_dicts = []
        for event in controlEvents:
            tempDict = {}
            tempDict['location'] = sensorIdxMapping[event['location']]
            tempDict['devices'] = {}
            for key,value in controlOutMap.items():
                tempDict['devices'][value] = event['devices'][key]
            input_dicts.append(tempDict)

        logger.debug("Performing data collection")
        for alias in aliasesSensorIdx:
            # if first:
            #     predictedTraj = {}
            #     predictedTraj[alias] = controller.trajectoryList[alias]['horizon_stored'][0,:,0].detach().numpy()
            #     first = False

            controlTraj = {}
            controlTraj['Time'] = current_time
            if simParams['testCase'] == 'DPC':
                names = ['u_hvac', 'y', 'ymax', 'ymin', 'powerRef', 'cost', 'stored', 'u_bat', 'bat_ref', 'u_tot']
                for key in names:
                    controlTraj[key] = controller.trajectoryList[alias][f"horizon_{key}"][0,0,0].detach().item()
            elif simParams['testCase'] == 'MPC':
                # names = ['u_heat', 'u_cool', 'u_bat', 'u_tot', 'y', 'y_max', 'y_min', 'd', 'bat_max', 'bat_min', 'stored', 'power_ref', 'cost', 'bat_ref']
                # names = ['u_hvac', 'y', 'y_max', 'y_min', 'd', 'cost']
                # for key in names:
                #     if not(key in controller.trajectoryList[alias].keys()):
                #         continue
                #     controlTraj[key] = controller.trajectoryList[alias][key][0,0]
                for key, value in controller.trajectoryList[alias]:
                    controlTraj[key] = value[0,0]
                controlTraj['mpc_feas'] = controller.controllerList[aliasesSensorIdx.index(alias)].feasible
                controlTraj['mpc_obj'] = controller.controllerList[aliasesSensorIdx.index(alias)].prob.objective.value
                controlTraj['hvac_mode'] = controller.controllerList[aliasesSensorIdx.index(alias)].HVAC_mode
                controlTraj['hvac_lock'] = controller.controllerList[aliasesSensorIdx.index(alias)].HVAC_lock
            elif simParams['testCase'] == 'MPC_alt':
                # names = ['u_hvac', 'u_hvac_shift', 'u_bat', 'u_bat_hvac', 'u_load', 'u_tot', 'y', 'y_ref', 'd', 'bat_max', 'bat_min', 'stored', 'power_ref', 'cost', 'bat_ref', 'charge_incen', 'base_load']
                # for key in names:
                #     if not(key in controller.trajectoryList[alias].keys()):
                #         continue
                #     controlTraj[key] = controller.trajectoryList[alias][key][0,0]
                for key, value in controller.trajectoryList[alias].items():
                    if len(value.shape) == 2:
                        controlTraj[key] = value[0,0]
                    elif len(value.shape) == 1:
                        controlTraj[key] = value[0]
                    else:
                        pass
                controlTraj['mpc_feas'] = controller.controllerList[aliasesSensorIdx.index(alias)].feasible
                controlTraj['mpc_obj'] = controller.controllerList[aliasesSensorIdx.index(alias)].prob.objective.value
                controlTraj['hvac_mode'] = controller.controllerList[aliasesSensorIdx.index(alias)].HVAC_mode
            outputs[alias].append(controlTraj)

            coordDict = {}
            coordDict['Time'] = current_time
            for key, value in controller.coordDebug[alias].items():
                coordDict[key] = value[0]
            coord_out[alias].append(coordDict)

            if simParams['resultLevel'] == 'ROLLOUT':
                # logger.debug("Performing rollout data collection")
                if controller.trajectoryList[alias]['y'][:,0].shape[0] == 1:
                    ic = controller.trajectoryList[alias]['y'][0,0]
                    tempRollout[alias].append(np.ones_like(controller.trajectoryList[alias]['u_hvac'][:,0]) * ic)
                else:
                    tempRollout[alias].append(controller.trajectoryList[alias]['y'][:,0])
                hvacRollout[alias].append(controller.trajectoryList[alias]['u_hvac'][:,0])
                batPowerRollout[alias].append(controller.trajectoryList[alias]['u_bat'][:,0])
                if controller.trajectoryList[alias]['stored'][:,0].shape[0] == 1:
                    ic = controller.trajectoryList[alias]['stored'][0,0]
                    batSOCRollout[alias].append(np.ones_like(controller.trajectoryList[alias]['u_bat'][:,0]) * ic)
                else:
                    batSOCRollout[alias].append(controller.trajectoryList[alias]['stored'][:,0])
                flexLoadRollout[alias].append(controller.coordDebug[alias]['flexLoad'])
                usageRollout[alias].append(controller.coordDebug[alias]['usagePenalty'])

        genDict = {}
        genDict['Time'] = current_time
        for key, value in controller.coordDebug['gen'].items():
            genDict[key] = value[0]
        coord_out['gen'].append(genDict)

        if simParams['resultLevel'] == 'ROLLOUT':
            for i in range(0, len(transInfo.keys())):
                predTransRollout[str(i+1)].append(controller.coordDebug['gen'][f"trans{i+1} base load"] + controller.coordDebug['gen'][f"trans{i+1} pred load"])

        logger.debug("Publishing values to other federates")
        logger.debug(input_dicts)
        h.helicsPublicationPublishString(pubid['control_events'], json.dumps(input_dicts))
        stepTime.append(dt.datetime.now() - stepStart)
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
    if key == 'gen':
        df.to_csv(ResultsDir+'gen_coord.csv', index=False)
    else:
        coord_df.append(df)

# Save data to csv
aggregateLoad = np.zeros(len(outputs_df[0]))
for i, building in enumerate(outputs_df):
    building.to_csv(ResultsDir+simParams['controlledAliases'][i]+'_control.csv', index=False)
for i, building in enumerate(coord_df):
    building.to_csv(ResultsDir+simParams['controlledAliases'][i]+'_coord.csv', index=False)

if simParams['resultLevel'] == 'ROLLOUT':
    extras = [tempRollout, hvacRollout, batPowerRollout, batSOCRollout, flexLoadRollout, predLoadRollout, usageRollout]
    extraNames = ['temp', 'hvac', 'batPow', 'batSOC', 'flexLoad', 'predLoad', 'usage']
    i = 0
    for rollout in extras:
        for key, value in rollout.items():
            # print(f"{key}: {len(value)}, {len(value[0])}, {len(value[1])}, {len(value[-2])}, {len(value[-1])}")
            np.savetxt(ExtrasDir+sensorIdxMapping[key]+'_'+extraNames[i]+'.csv', np.array(rollout[key][:-1]), delimiter=',')
        i += 1
    for key, value in predTransRollout.items():
        np.savetxt(ExtrasDir+f'trans{key}_predLoad.csv', np.array(value), delimiter=',')

# finalize and close the federate
h.helicsFederateDestroy(fed)
h.helicsCloseLibrary()

elapsedTime = dt.datetime.now() - initTime

with open(ResultsDir+'simTime.txt', 'w') as fp:
    fp.write(f"Finished at {dt.datetime.now()}. Simulation took {elapsedTime}.\nAverage sim step time: {np.mean(stepTime)}")

logger.info(f"Finished at {dt.datetime.now()}. Simulation took {elapsedTime}.")
logger.info(f"Average sim step time: {np.mean(stepTime)}. Median: {np.median(stepTime)}")