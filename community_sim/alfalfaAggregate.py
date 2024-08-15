import os
import datetime as dt
import pandas as pd
import helics as h
import json
import logging
from alfalfa_client.alfalfa_client import AlfalfaClient
from pathlib import Path
import numpy as np
from slackNotifier import SlackNotifier

import buildingController

def Main():
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    # -----Create and setup alfalfa simulation-----
    # Create new alfalfa client object
    ac = AlfalfaClient(host='http://localhost')

    # Define paths to models to by uploaded
    model_paths = list(Path('./building_models').iterdir())

    # Upload sites to alfalfa
    site_ids = ac.submit(model_paths)
    logger.debug(site_ids)

    # Alias sites to keep track of which buses they are at
    aliases = []
    for i in range(0, len(model_paths)):
        alias = model_paths[i].parts[1]
        ac.set_alias(alias, site_ids[i])
        aliases.append(alias)

    # Define parameters to run simulation
    # If you are using historian, you will need to search for this time period in Grafana dashboard to view results.
    start_time = dt.datetime(2023, 1, 1)
    stepsize = dt.timedelta(minutes=1)
    duration = dt.timedelta(days=365)
    warmup = dt.timedelta(hours=1)
    start_warmup = start_time - warmup
    end_time = start_time + duration

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
    print("Running warmup period")
    warmup_times = pd.date_range(start_warmup, freq=stepsize, end=start_time)
    for step, current_time in enumerate(warmup_times):
        ac.advance(site_ids)

    # Get model's input points
    for site in site_ids:
        logger.debug(f"{site} inputs:")
        logger.debug(ac.get_inputs(site))

    # Set initial inputs
    # input_dicts = {
    #     'S22': {'Cafeteria_ZN_1_FLR_1_ZN_PSZ_AC_2_7_Outside_Air_Damper_CMD': 0.7},
    #     'S27': {'Auditorium_ZN_1_FLR_1_ZN_PSZ_AC_3_7_Outside_Air_Damper_CMD': 0.8},
    #     'S35': {'Core_ZN_ZN_PSZ_AC_1_Outside_Air_Damper_CMD': 0.6}
    # }
    # for i in range(0,len(aliases)):
    #     ac.set_inputs(ac.get_alias(aliases[i]), input_dicts[aliases[i]])

    # -----Execute Federate, set up and start co-simulation-----
    outputs = [[] for _ in range(len(aliases))]
    voltage_results = []
    times = pd.date_range(start_time, freq=stepsize, end=end_time)
    for step, current_time in enumerate(times):
        # Update time in co-simulation
        present_step = (current_time - start_time).total_seconds()

        # Controller code goes here

        # Advance the model
        ac.advance(site_ids)
        logger.debug(f"Model advanced to time: {ac.get_sim_time(site_ids[0])}")

        # Get building electricity consumption
        load_powers = {}
        for i, alias in enumerate(aliases):
            site = ac.get_alias(alias)
            alf_outs = ac.get_outputs(site)
            alf_outs['Time'] = ac.get_sim_time(site)
            outputs[i].append(alf_outs)
            load_powers.update({alias:alf_outs['Whole Building Electricity']*0.1e-3})
        # Probably going to want something like this in the future
        # load_powers = {k:alf_outs[k] for k in ('Whole Building Electricity') if k in alf_outs}

    # Stop the simulation
    ac.stop(site_ids)

    # Put output lists in dataframes
    outputs_df = []
    for building in outputs:
        outputs_df.append(pd.DataFrame(building))

    # Save data to csv
    aggregateLoad = np.zeros(len(outputs_df[0]))
    for i, building in enumerate(outputs_df):
        building['Whole Building Electricity'] = building['Whole Building Electricity'] * 2.7778e-7 * 60
        building.to_csv('results/'+aliases[i]+'_out.csv')
        aggregateLoad += building['Whole Building Electricity'].values

    aggregate_df = pd.DataFrame(aggregateLoad)
    aggregate_df.to_csv('results/aggregate_out.csv')

if __name__ == '__main__':
    notifier = SlackNotifier('Alfalfa Aggregate', ['U04J6NQG084'])
    try:
        
        notifier.Start()
        Main()
        notifier.Stop()
    except:
        notifier.Error()