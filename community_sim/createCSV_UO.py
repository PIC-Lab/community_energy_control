import json
import pandas as pd
import os
import shutil

def Main():

    with open('simParams.json') as fp:
        simParams = json.load(fp)
    buildingList = [str(i) for i in range(1,29)]
    controlList = simParams['controlledAliases']

    resultsDir = 'results/'

    df_list = {}
    for i in buildingList:
        temp_df = pd.read_csv(resultsDir+str(i)+'_out.csv', index_col=0)
        temp_df.index = pd.to_datetime(temp_df.index)
        df_list[i] = temp_df

    control_list = {}
    for i in controlList:
        temp_df = pd.read_csv(resultsDir+str(i)+'_control.csv', index_col=0)
        temp_df.index = pd.to_datetime(temp_df.index)
        control_list[i] = temp_df

    battery_df = pd.read_csv(resultsDir+'battery_results.csv', index_col=0)
    battery_dispatch_df = pd.read_csv(resultsDir+'battery_dispatch.csv')

    saveDir = 'alf_schedules'
    if not(os.path.exists(saveDir)):
        os.mkdir(saveDir)

    for alias in controlList:
        temp_df = pd.DataFrame
        temp_df['Time'] = df_list[alias]['Time']
        temp_df['HVAC Power'] = df_list[alias]['Electricity:HVAC']
        temp_df['Battery Dispatch Power'] = battery_dispatch_df['Storage.battery'+alias]
        temp_df.to_csv(saveDir+alias+'.csv')