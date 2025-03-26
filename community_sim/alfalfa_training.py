import os
import datetime as dt
import pandas as pd
from alfalfa_client.alfalfa_client import AlfalfaClient
from pathlib import Path
import numpy as np
import traceback

from communityController.buildingController.buildingController import BuildingController

# Controller name maps
controlInMap = { 'indoorAirTemp': 'living space Air Temperature'}
controlOutMap = {'heatingSetpoint': 'heating setpoint', 'coolingSetpoint': 'cooling setpoint'}

resultsDir = Path('results/summerExp')

def Main():
    # Create results folder if it doesn't exist
    if not(resultsDir.exists()):    
        resultsDir.mkdir(parents=True)

    start_time = dt.datetime(2023, 7, 1)
    duration = dt.timedelta(days=40)
    # first = True
    # for i in range(0,6):
    #     if i == 5:
    #         duration = dt.timedelta(days=65)
    #     current_time = RunAlfalfa(start_time, duration, first)
    #     if first:
    #         first = False
    #     current_time += dt.timedelta(minutes=1)
    #     start_time = current_time
    RunAlfalfa(start_time, duration, first=True)

def RunAlfalfa(start_time, duration, first):

    # Create new alfalfa client object
    ac = AlfalfaClient(host='http://localhost')

    # Define paths to models to by uploaded
    model_paths = list(Path('./building_models').iterdir())

    # Upload sites to alfalfa
    site_ids = ac.submit(model_paths)

    # Alias sites to keep track of which buses they are at
    aliases = []
    for i in range(0, len(model_paths)):
        alias = model_paths[i].parts[1]
        ac.set_alias(alias, site_ids[i])
        aliases.append(alias)

    # Load setpoint schedules
    dayNum = start_time.timetuple().tm_yday - 1
    temp_df = pd.read_csv('results/randSch/hvacSch.csv', header=None, names=['heatSP', 'coolSP', 'db'])
    randSP = temp_df[dayNum*24*60:(dayNum+duration.days)*24*60+1]
    randSP.reset_index(inplace=True, drop=True)

    temp_df = pd.read_csv('results/randSch/hvacAvail.csv', header=None, names=['avail'])
    randAvail = temp_df[dayNum*24*60:(dayNum+duration.days)*24*60+1]
    randAvail.reset_index(inplace=True, drop=True)

    # Define parameters to run simulation
    # If you are using historian, you will need to search for this time period in Grafana dashboard to view results.
    stepsize = dt.timedelta(minutes=1)
    warmup = dt.timedelta(minutes=30)
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
        ac.start(site, **params)

    # Run warmup
    print("Running warmup period")
    warmup_times = pd.date_range(start_warmup, freq=stepsize, end=start_time)
    for step, current_time in enumerate(warmup_times):
        if step < warmup / stepsize:        # Don't advancing alfalfa sim at the end of the loop to avoid simulations getting out of sync
            ac.advance(site_ids)

    # Create input dictionary and set initial inputs
    input_dicts = {}
    for alias in aliases:
        input_dicts[alias] = {'heating setpoint': 21, 'cooling setpoint': 24}
        ac.set_inputs(ac.get_alias(alias), input_dicts[alias])

    # Create controller object
    controllerList = []
    for alias in aliases:
        controllerList.append(BuildingController(alias, None, train=True))

    # -----Execute Federate, set up and start co-simulation-----
    outputs = {alias: [] for alias in aliases}
    times = pd.date_range(start_time, freq=stepsize, end=end_time)
    for step, current_time in enumerate(times):
        # Update time in co-simulation
        present_step = (current_time - start_time).total_seconds()

        # Get building electricity consumption
        for i,alias in enumerate(aliases):
            site = ac.get_alias(alias)
            alf_outs = ac.get_outputs(site)
            alf_outs['Time'] = ac.get_sim_time(site)

            # Update controllers
            for key,value in controlInMap.items():
                controllerList[i].sensorValues[key] = alf_outs[value]

            controllerList[i].setpointInfo['heatSP'] = randSP['heatSP'].loc[step]
            controllerList[i].setpointInfo['coolSP'] = randSP['coolSP'].loc[step]
            controllerList[i].setpointInfo['deadband'] = randSP['db'].loc[step]

            # controllerList[i].HVAC_Control(forceMode=-1, op_mode=randAvail['avail'].loc[step])
            controllerList[i].RandomControl()

            for key,value in controlOutMap.items():
                input_dicts[alias][value] = controllerList[i].actuatorValues[key]

            # Push updates to inputs to alfalfa
            ac.set_inputs(site, input_dicts[alias])

            # Should probably move this to the measures
            alf_outs['Whole Building Electricity'] *= 1e-3      # convert to kW
            alf_outs['Heating:Electricity'] *= 1e-3 / 60
            # alf_outs['Cooling:Electricity'] *= 1e-3 / 60
            alf_outs['WaterSystems:Electricity'] *= 1e-3 / 60
            alf_outs['Electricity:HVAC'] *= 1e-3 / 60
            outputs[alias].append(alf_outs)

        # Advance the model
        if step < duration / stepsize:          # Don't advance alfalfa on the last iteration of the loop
            ac.advance(site_ids)

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
    if first:
        for i, building in enumerate(outputs_df):
            fileName = aliases[i]+'_out.csv'
            building.to_csv(resultsDir/fileName, index=False)
    else:
        for i, building in enumerate(outputs_df):
            fileName = aliases[i]+'_out.csv'
            building.to_csv(resultsDir/fileName, index=False, header=False, mode='a')

    return current_time

if __name__ == '__main__':
    slack = True
    try:
        import sys
        sys.path.insert(1, '../../../slackNotifier')
        from slackNotifier import SlackNotifier         # type:ignore
    except:
        print('Slack notifier not found. Running without it')
        slack = False

    if slack:
        notifier = SlackNotifier('Leadville Alfalfa Training Data', ['U04J6NQG084'])
        try:
            
            notifier.Start()
            Main()
            notifier.Stop()
        except Exception as e:
            notifier.Error(e)
            print(traceback.format_exc())
    else:
        Main()
