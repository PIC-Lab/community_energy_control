import json
import pandas as pd
import os
import shutil
import argparse

def Main():
    parser = argparse.ArgumentParser(
        prog='CreateCSV_UO',
        description='Processes alfalfa results and create csv files for use in urbanopt'
    )
    parser.add_argument('resultsDir', help='Name of location within the standard results directory')

    args = parser.parse_args()
    resultsDir = f'results/{args.resultsDir}/'

    with open('simParams.json') as fp:
        simParams = json.load(fp)
    buildingList = [str(i) for i in range(1,29)]
    controlList = simParams['controlledAliases']

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

    battery_dispatch_df = pd.read_csv(resultsDir+'battery_dispatch.csv', index_col=0)
    battery_dispatch_df.index = df_list[controlList[0]].index

    saveDir = 'alf_schedules/'
    if not(os.path.exists(saveDir)):
        os.mkdir(saveDir)

    for alias in controlList:
        temp_df = pd.DataFrame()
        temp_df.index = df_list[alias].index
        temp_df['HVAC Power'] = df_list[alias]['Electricity:HVAC']
        temp_df['Water Heater Power'] = df_list[alias]['WaterSystems:Electricity']
        temp_df['Battery Dispatch Power'] = battery_dispatch_df['Storage.battery'+alias]
        temp_df.to_csv(saveDir+alias+'.csv')
    
    print(f"csv files saved to {saveDir}")

if __name__ == "__main__":
    Main()