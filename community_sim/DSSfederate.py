import os
import datetime as dt
import pandas as pd
import helics as h
import json
import logging
from opendss_wrapper import OpenDSS

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

with open('simParams.json') as fp:
    simParams = json.load(fp)

# Folder and File locations
MainDir = os.path.abspath(os.path.dirname(__file__))
ModelDir = os.path.join(MainDir, 'network_model')
ResultsDir = os.path.join(MainDir, 'results')
os.makedirs(ResultsDir, exist_ok=True)

# Output files
load_info_file = os.path.join(ResultsDir, 'load_info.csv')
main_results_file = os.path.join(ResultsDir, 'main_results.csv')
voltage_file = os.path.join(ResultsDir, 'voltage_results.csv')
elements_file = os.path.join(ResultsDir, 'element_results.csv')
load_powers_results_file = os.path.join(ResultsDir, 'load_powers_results.csv')
battery_results_file = os.path.join(ResultsDir, 'battery_results.csv')
battery_dispatch_file = os.path.join(ResultsDir, 'battery_dispatch.csv')
battery_power_file = os.path.join(ResultsDir, 'battery_power.csv')
battery_state_file = os.path.join(ResultsDir, 'battery_state.csv')

# ----- HELICS federate setup -----
# Register federate from json
fed = h.helicsCreateCombinationFederateFromConfig(
    os.path.join(os.path.dirname(__file__), "DSSfederate.json")
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

# ----- Create DSS Network -----
MasterFile = os.path.join(ModelDir, 'Master.dss')
start_time = dt.datetime.strptime(simParams['start'], "%m/%d/%y")
stepsize = pd.Timedelta(simParams['step'])
duration = pd.Timedelta(simParams['duration'])
end_time = start_time + duration
dss = OpenDSS([MasterFile], stepsize, start_time)

# Run additional OpenDSS commands
dss.run_command('set controlmode=time')

# Get info on all properties of a class
df = dss.get_all_elements('Load')
df.to_csv(load_info_file)

storageInfo = dss.get_all_elements('Storage')

# ----- Primary co-simulation loop -----
# Define lists for data collection
main_results = []
voltage_results = []
element_results = []
load_powers_results = []
battery_results = []
battery_dispatch = []
battery_power = []
battery_state = []

# Execute federate and start co-sim
h.helicsFederateEnterExecutingMode(fed)
times = pd.date_range(start_time, freq=stepsize, end=end_time)
for step, current_time in enumerate(times):

    # Update time in co-simulation
    present_step = (current_time - start_time).total_seconds()
    present_step += 1  # Ensures other federates update before DSS federate
    h.helicsFederateRequestTime(fed, present_step)

    # get signals from other federate
    logger.debug(f"Current time: {current_time}, step: {step}")
    isupdated = h.helicsInputIsUpdated(subid['load_powers'])
    if isupdated == 1:
        loadPowers = h.helicsInputGetString(subid['load_powers'])
        loadPowers = json.loads(loadPowers)
        logger.debug("Recieved updated value for load_powers")
        logger.debug(loadPowers)
    else:
        loadPowers = {}

    isupdated = h.helicsInputIsUpdated(subid['control_events'])
    if isupdated == 1:
        controlEvents = h.helicsInputGetString(subid['control_events'])
        controlEvents = json.loads(controlEvents)
        logger.debug("Recieved updated value for control_events")
        logger.debug(controlEvents)
    else:
        controlEvents = {}

    # load
    for loadName, set_point in loadPowers.items():
        dss.set_power(loadName, element='Load', p=set_point)

    # Battery control
    for controlSet in controlEvents:
            location = controlSet['location']
            for key, value in controlSet['devices'].items():
                if 'Battery' in key:
                    max_kW = storageInfo.loc['Storage.battery'+location, 'kWrated']
                    # value = round(min(max_kW, max(-1*max_kW, value)), 3)         # Clamp value to be no larger than rated kW 
                    # print(key, value)
                    # print(dss.get_property(key+location, 'kW', element='Storage'))
                    # print(dss.get_property(key+location, 'State', element='Storage'))
                    # try:
                    #     dss.set_property(key+location, 'kW', value, element='Storage')
                    # except AssertionError:
                    #     logger.debug(f'Ignored value of {value} for {key}')
                    # print(dss.get_property(key+location, 'kW', element='Storage'))
                    # print(dss.get_property(key+location, 'State', element='Storage'))
                    
                    dss.set_power(key+location, element='Storage', p=value)
                else:
                    continue

    # solve OpenDSS network model
    dss.run_dss()

    # Publish battery results
    logger.debug("Publishing values to other federates")
    battery_data = dss.get_all_elements('Storage')
    logger.debug(battery_data["%stored"].to_dict())
    h.helicsPublicationPublishString(pubid['battery_soc'], json.dumps(battery_data['%stored'].to_dict()))
      
    # Get outputs for the feeder, all voltages, and individual element voltage and power
    main_results.append(dss.get_circuit_info())

    voltage_results.append(dss.get_all_bus_voltages(average=True))

    battery_results.append(battery_data['%stored'].to_dict())
    battery_dispatch.append(battery_data['kW'].to_dict())
    battery_power.append(dss.get_power('Battery4', element='Storage', total=True)[0])
    battery_state.append(battery_data['State'].to_dict())
    
    # Get load data
    load_powers_data = {}
    for loadName in loadPowers:
        load_powers_data.update({loadName: dss.get_power(loadName, element='Load', total=True)[0]})
        logger.debug(f"{loadName}, {dss.get_power(loadName, element='Load')}") 
        logger.debug(f"{loadName}, {dss.get_power(loadName, element='Load', total=True)}")
    load_powers_results.append(load_powers_data)

# Save results files
pd.DataFrame(main_results).to_csv(main_results_file)
pd.DataFrame(voltage_results).to_csv(voltage_file)
pd.DataFrame(load_powers_results).to_csv(load_powers_results_file)
pd.DataFrame(battery_results).to_csv(battery_results_file)
pd.DataFrame(battery_dispatch).to_csv(battery_dispatch_file)
pd.DataFrame(battery_power).to_csv(battery_power_file)
pd.DataFrame(battery_state).to_csv(battery_state_file)

# finalize and close the federate
h.helicsFederateDestroy(fed)
h.helicsCloseLibrary()