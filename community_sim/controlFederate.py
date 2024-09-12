import helics as h
import json
import logging
import os
import datetime as dt
import pandas as pd
import sys

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

from communityController.communityController import CommunityController

with open('simParams.json') as fp:
    simParams = json.load(fp)

with open('indexMapping.json') as fp:
    sensorIdxMapping = json.load(fp)        # Map sensor indices to simulation indices
simIdxMapping = {v: k for k, v in sensorIdxMapping.items()}     # Map simulation indices to sensor indices

# Controller name maps
# 'controller name': 'model name'
controlInMap = {'indoorAirTemp': 'living space Air Temperature'}
controlOutMap = {'heatingSetpoint': 'heating setpoint', 'coolingSetpoint': 'cooling setpoint'}
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

# Execute federate and start co-sim
h.helicsFederateEnterExecutingMode(fed)
times = pd.date_range(start_time, freq=stepsize, end=end_time)
for step, current_time in enumerate(times):
    # Update time in co-simulation
    present_step = (current_time - start_time).total_seconds()
    h.helicsFederateRequestTime(fed, present_step)

    # get signals from other federate
    logger.debug(f"Current time: {current_time}, step: {step}")
    isupdated = h.helicsInputIsUpdated(subid['battery_soc'])
    if isupdated == 1:
        batterySOC = h.helicsInputGetString(subid['battery_soc'])
        batterySOC = json.loads(batterySOC)
        logger.debug("Recieved updated value for battery_soc")
    else:
        batterySOC = {}

    isupdated = h.helicsInputIsUpdated(subid['indoor_temp'])
    if isupdated == 1:
        indoorTemp = h.helicsInputGetString(subid['indoor_temp'])
        indoorTemp = json.loads(indoorTemp)
        logger.debug("Recieved updated value for indoor_temp")
    else:
        indoorTemp = {}
    
    sensorValues = {k:{name: v for name, v in zip(sensorNames, batterySOC, indoorTemp)}
                    for k in batterySOC.keys()}

    controlEvents = controller.Step(sensorValues, current_time)

    # Map actuator values to control event format
    input_dicts = {}
    for key,value in controlOutMap.items():
        input_dicts[controller.controlAliasList[i]][value] = controller.controllerList[i].actuatorValues[key]

# finalize and close the federate
h.helicsFederateDestroy(fed)
h.helicsCloseLibrary()