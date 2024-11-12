import os
import datetime as dt
import pandas as pd
import helics as h
import json
import logging
from alfalfa_client.alfalfa_client import AlfalfaClient
from pathlib import Path
import numpy as np

# from eventParser import ParseControlEvent

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

with open('simParams.json') as fp:
    simParams = json.load(fp)

# ----- HELICS federate setup -----
# Register federate from json
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

# Define parameters to run simulation
# If you are using historian, you will need to search for this time period in Grafana dashboard to view results.
start_time = dt.datetime.strptime(simParams['start'], "%m/%d/%y")
stepsize = pd.Timedelta(simParams['step'])
duration = pd.Timedelta(simParams['duration'])
warmup = pd.Timedelta(simParams['warmup'])
start_warmup = start_time - warmup
end_time = start_time + duration
logger.debug(f"Run period: {start_warmup} to {end_time}")

# ----- Create and setup alfalfa simulation -----
# Create new alfalfa client object
ac = AlfalfaClient(host='http://localhost')

# Define paths to models to by uploaded
model_paths = list(Path('./building_models').iterdir())
# model_paths = [Path('./building_models/4')]

# Upload sites to alfalfa
site_ids = ac.submit(model_paths)
logger.debug(site_ids)

aliases = [x.name for x in model_paths]

# Alias sites to keep track of which buses they are at
for i in range(0, len(model_paths)):
    ac.set_alias(aliases[i], site_ids[i])
logger.debug(aliases)

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

# Create results folder if it doesn't exist
resultsDir = Path('results')
if not(resultsDir.exists()):    
    resultsDir.mkdir()

# ----- Primary co-simulation loop -----
# Define lists for data collection
outputs = {alias: [] for alias in aliases}

# Execute federate and start co-sim
h.helicsFederateEnterExecutingMode(fed)
times = pd.date_range(start_time, freq=stepsize, end=end_time)
for step, current_time in enumerate(times):
    # Update time in co-simulation
    present_step = (current_time - start_time).total_seconds()
    h.helicsFederateRequestTime(fed, present_step)

    # get signals from other federate
    logger.debug(f"Current time: {current_time}, step: {step}")
    isupdated = h.helicsInputIsUpdated(subid['control_events'])
    if isupdated == 1:
        controlEvents = h.helicsInputGetString(subid['control_events'])
        controlEvents = json.loads(controlEvents)
        logger.debug("Recieved updated value for control_events")
        logger.debug(controlEvents)
    else:
        controlEvents = {}
            
    # Get building electricity consumption
    loadPowers = {}
    indoorTemp = {}
    for i,alias in enumerate(aliases):
        site = ac.get_alias(alias)
        alf_outs = ac.get_outputs(site)
        alf_outs['Time'] = ac.get_sim_time(site)

        # Push to control federates
        # for key,value in controlInMap.items():
        #     controllerList[i].sensorValues[key] = alf_outs[value]

        # if simParams['testCase'] == 'base':
        #     controllerList[i].HVAC_Control(forceMode=-1)
        # elif simParams['testCase'] == 'MPC':
        #     trajectories = controllerList[i].PredictiveControl(step)

        # Parse control events
        for controlSet in controlEvents:
            location = controlSet['location']
            input_dicts[location] = {}
            for key, value in controlSet['devices'].items():
                if 'Battery' in key:
                    continue
                input_dicts[location][key] = value
        # for alias, controlSet in controlEvents.items():
        #     location = controlSet['location']
        #     for key, value in controlSet['devices']:
        #         if key == 'Battery':
        #             continue
        #         input_dicts[alias][value] = controlSet['status']

        # Push updates to inputs to alfalfa
        ac.set_inputs(site, input_dicts[alias])

        # Should probably move this to the measures
        alf_outs['Whole Building Electricity'] *= 1e-3      # convert to kW
        alf_outs['Heating:Electricity'] *= 1e-3 / 60
        # alf_outs['Cooling:Electricity'] *= 1e-3 / 60
        alf_outs['WaterSystems:Electricity'] *= 1e-3 / 60
        alf_outs['Electricity:HVAC'] *= 1e-3 / 60
        outputs[alias].append(alf_outs)

        loadPowers[alias] = alf_outs['Whole Building Electricity']
        indoorTemp[alias] = alf_outs['living space Air Temperature']

    # Publish values
    logger.debug("Publishing values to other federates")
    h.helicsPublicationPublishString(pubid['load_powers'], json.dumps(loadPowers))
    h.helicsPublicationPublishString(pubid['indoor_temp'], json.dumps(indoorTemp))

    logger.debug(loadPowers)
    logger.debug(indoorTemp)

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