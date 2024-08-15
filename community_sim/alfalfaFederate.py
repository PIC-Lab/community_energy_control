import os
import datetime as dt
import pandas as pd
import helics as h
import json
import logging
from alfalfa_client.alfalfa_client import AlfalfaClient
from pathlib import Path
import numpy as np

from buildingController import BuildingController

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

# testCase = 'base'
testCase = 'MPC'

# Controller name maps
# 'controller name': 'model name'
controlInMap = { 'indoorAirTemp': 'living space Air Temperature'}
controlOutMap = {'heatingSetpoint': 'heating setpoint', 'coolingSetpoint': 'cooling setpoint'}

# -----Register federate from json-----
fed = h.helicsCreateCombinationFederateFromConfig(
    os.path.join(os.path.dirname(__file__), "alfalfaFederate.json")
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
    subid[i] = h.helicsFederateGetInputByIndex(fed, i)
    sub_name = h.helicsInputGetName(subid[i])
    logger.debug(f"\tRegistered subscription---> {sub_name}")

pubid = {}
for i in range(0, pub_count):
    pubid[i] = h.helicsFederateGetPublicationByIndex(fed, i)
    pub_name = h.helicsPublicationGetName(pubid[i])
    logger.debug(f"\tRegistered publication---> {pub_name}")

# Create results folder if it doesn't exist
resultsDir = Path('results')
if not(resultsDir.exists()):    
    resultsDir.mkdir()

# -----Create and setup alfalfa simulation-----
# Create new alfalfa client object
ac = AlfalfaClient(host='http://localhost')

# Define paths to models to by uploaded
model_paths = list(Path('./building_models').iterdir())

# Upload sites to alfalfa
site_ids = ac.submit(model_paths)
logger.debug(site_ids)

aliases = [x.name for x in model_paths]

# Alias sites to keep track of which buses they are at
for i in range(0, len(model_paths)):
    ac.set_alias(aliases[i], site_ids[i])
logger.debug(aliases)

# Define parameters to run simulation
# If you are using historian, you will need to search for this time period in Grafana dashboard to view results.
# start_time = dt.datetime(2023, 1, 15)
start_time = dt.datetime(2023, 7, 1)
stepsize = dt.timedelta(minutes=1)
duration = dt.timedelta(days=1)
warmup = dt.timedelta(hours=5)
start_warmup = start_time - warmup
end_time = start_time + duration
logger.debug(f"Run period: {start_warmup} to {end_time}")

# For external_clock == true, API calls are used to advance the model.  
# If external_clock == false, Alfalfa will handle advancing the model according to a specified timescale (timescale 1 => realtime)
params = {
    "external_clock": True,
    "start_datetime": start_warmup,
    "end_datetime": end_time
}

# Start simulations
for site in site_ids:
    logger.debug(f"Starting site: {site}")
    ac.start(site, **params)

# Run warmup
logger.debug("Running warmup period")
warmup_times = pd.date_range(start_warmup, freq=stepsize, end=start_time)
for step, current_time in enumerate(warmup_times):
    if step < warmup / stepsize:        # Don't advancing alfalfa sim at the end of the loop to avoid simulations getting out of sync
        ac.advance(site_ids)
logger.debug(f'Finished warmup, current sim time is {current_time}')

# Get model's input points
for site in site_ids:
    logger.debug(f"{site} inputs:")
    logger.debug(ac.get_inputs(site))

# Create input dictionary and set initial inputs
input_dicts = {}
for alias in aliases:
    input_dicts[alias] = {'heating setpoint': 21, 'cooling setpoint': 24}
    ac.set_inputs(ac.get_alias(alias), input_dicts[alias])

# Create controller object
controllerList = []
for i, alias in enumerate(aliases):
    controllerList.append(BuildingController(alias, testCase))

# -----Execute Federate, set up and start co-simulation-----
h.helicsFederateEnterExecutingMode(fed)
outputs = {alias: [] for alias in aliases}
voltage_results = []
times = pd.date_range(start_time, freq=stepsize, end=end_time)
for step, current_time in enumerate(times):
    # Update time in co-simulation
    present_step = (current_time - start_time).total_seconds()
    h.helicsFederateRequestTime(fed, present_step)

    # get signals from other federate
    isupdated = h.helicsInputIsUpdated(subid[0])
    if isupdated == 1:
        bus_voltages = h.helicsInputGetString(subid[0])
        bus_voltages = json.loads(bus_voltages)
    else:
        bus_voltages = {}

    logger.debug(f"Current time: {current_time}, step: {step}. Received value: bus_voltages = {bus_voltages}")

    # Get building electricity consumption
    for i,alias in enumerate(aliases):
        site = ac.get_alias(alias)
        alf_outs = ac.get_outputs(site)
        # logger.debug(alf_outs)
        alf_outs['Time'] = ac.get_sim_time(site)

        # Update controllers
        for key,value in controlInMap.items():
            controllerList[i].sensorValues[key] = alf_outs[value]

        if testCase == 'base':
            controllerList[i].HVAC_Control(forceMode=-1)
        elif testCase == 'MPC':
            trajectories = controllerList[i].PredictiveControl(step)

        for key,value in controlOutMap.items():
            input_dicts[alias][value] = controllerList[i].actuatorValues[key]
        # logger.debug(controllerList[i].actuatorValues)

        # Push updates to inputs to alfalfa
        ac.set_inputs(site, input_dicts[alias])

        # Should probably move this to the measures
        alf_outs['Whole Building Electricity'] *= 1e-3      # convert to kW
        alf_outs['Heating:Electricity'] *= 1e-3 / 60
        # alf_outs['Cooling:Electricity'] *= 1e-3 / 60
        alf_outs['WaterSystems:Electricity'] *= 1e-3 / 60
        alf_outs['Electricity:HVAC'] *= 1e-3 / 60
        if testCase == 'MPC':
            alf_outs['Control Effort'] = trajectories['u'][0,0,0].detach().item()
            alf_outs['Predicted Temperature'] = trajectories['y'][0,0,0].detach().item()
            alf_outs['Ymax'] = trajectories['ymax'][0,0,0].detach().item()
            alf_outs['Ymin'] = trajectories['ymin'][0,0,0].detach().item()
        outputs[alias].append(alf_outs)

    # if 'ERROR' in ac.status(site_ids):
    #     errorLog = ac.get_error_log(site_ids)
    #     raise RuntimeError(errorLog)

    # Advance the model
    if step < duration / stepsize:          # Don't advance alfalfa on the last iteration of the loop
        ac.advance(site_ids)
        logger.debug(f"Model advanced to time: {ac.get_sim_time(site_ids[0])}")

# Stop the simulation
ac.stop(site_ids)

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
    building.to_csv('results/'+aliases[i]+'_out.csv', index=False)
    aggregateLoad += building['Whole Building Electricity'].values

aggregate_df = pd.DataFrame(aggregateLoad)
aggregate_df.to_csv('results/aggregate_out.csv', index=False)

# finalize and close the federate
h.helicsFederateDestroy(fed)
h.helicsCloseLibrary()