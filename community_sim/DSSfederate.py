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


""" Register federate from json """
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
    subid[i] = h.helicsFederateGetInputByIndex(fed, i)
    sub_name = h.helicsInputGetName(subid[i])
    logger.debug(f"\tRegistered subscription---> {sub_name}")

pubid = {}
for i in range(0, pub_count):
    pubid[i] = h.helicsFederateGetPublicationByIndex(fed, i)
    pub_name = h.helicsPublicationGetName(pubid[i])
    logger.debug(f"\tRegistered publication---> {pub_name}")

""" Create DSS Network """
MasterFile = os.path.join(ModelDir, 'Master.dss')
start_time = dt.datetime(2023, 7, 1)
stepsize = dt.timedelta(minutes=1)
duration = dt.timedelta(days=1)
dss = OpenDSS([MasterFile], stepsize, start_time)

# Run additional OpenDSS commands
dss.run_command('set controlmode=time')

# Get info on all properties of a class
df = dss.get_all_elements('Load')
df.to_csv(load_info_file)

""" Execute Federate, set up and start co-simulation """
h.helicsFederateEnterExecutingMode(fed)
main_results = []
voltage_results = []
element_results = []
load_powers_results = []
times = pd.date_range(start_time, freq=stepsize, end=start_time + duration)
for step, current_time in enumerate(times):

    # Update time in co-simulation
    present_step = (current_time - start_time).total_seconds()
    present_step += 1  # Ensures other federates update before DSS federate
    h.helicsFederateRequestTime(fed, present_step)

    # get signals from other federate
    isupdated = h.helicsInputIsUpdated(subid[0])
    if isupdated == 1:
        load_powers = h.helicsInputGetString(subid[0])
        load_powers = json.loads(load_powers)
    else:
        load_powers = {}

    logger.debug(f"Current time: {current_time}, step: {step}. Received value: load_powers = {load_powers}")

    # load
    for load_name, set_point in load_powers.items():
        dss.set_power(load_name, element='Load', p=set_point)

    # solve OpenDSS network model
    dss.run_dss()

    # Publish voltage results
    h.helicsPublicationPublishString(pubid[0], json.dumps(dss.get_all_bus_voltages(average=True)))
      
    """ Get outputs for the feeder, all voltages, and individual element voltage and power """
    main_results.append(dss.get_circuit_info())

    voltage_results.append(dss.get_all_bus_voltages(average=True))

    # element_results.append({
    #     'Load Power (kW)': dss.get_power('S22', element='Load', total=True)[0],
    #     'Load Power (kW)': dss.get_power('S27', element='Load', total=True)[0],
    #     'Load Power (kW)': dss.get_power('S35', element='Load', total=True)[0],
    #     'Load Voltage (p.u.)': dss.get_voltage('S22', element='Load', average=True),
    #     'Load Voltage (p.u.)': dss.get_voltage('S27', element='Load', average=True),
    #     'Load Voltage (p.u.)': dss.get_voltage('S35', element='Load', average=True)
    # })
    
    # Get load data
    load_powers_data = {}
    for load_name in load_powers:
        load_powers_data.update({load_name: dss.get_power(load_name, element='Load', total=True)[0]})    
    load_powers_results.append(load_powers_data)
    

""" Save results files """
pd.DataFrame(main_results).to_csv(main_results_file)
pd.DataFrame(voltage_results).to_csv(voltage_file)
# pd.DataFrame(element_results).to_csv(elements_file)
pd.DataFrame(load_powers_results).to_csv(load_powers_results_file)

# finalize and close the federate
h.helicsFederateDestroy(fed)
h.helicsCloseLibrary()