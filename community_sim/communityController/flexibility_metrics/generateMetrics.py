import numpy as np
import pandas as pd
import os
from pathlib import Path

def CreateMetrics(df_list):
    dataPath = Path('../../results/summer/')

    batRated = 7.6

    for file in dataPath.iterdir():
        df = pd.read_csv(file)
        df_metric = pd.DataFrame()
        
        heatingLow = df.get('Heating (kW)').max() * 0.05
        coolingLow = df.get('Cooling (kW)').max() * 0.05
        waterLow = df.get('Water Systems (kW)').max() * 0.05
        numRows = df.shape[0]
        numCols = df.shape[1]

        df['Heating On'] = df['HVAC Mode'].replace([1,2,3,4], [0,0,1,1])
        df['Cooling On'] = df['HVAC Mode'].replace([1,2,3,4], [1,1,0,0])
        df['Water Heater On'] = np.where(df['Water Systems (kW)'] > 0, 1, 0)
        df['Water Heater On'] = df['Water Heater On'].replace([0,1], [1,0])
        
        # Calculate flexibility envelope   
        df_metric['low'] = df.get('Whole Building Electricity').to_numpy() - \
                           df.get('Electricity:HVAC').to_numpy() - \
                           df.get('WaterSystems:Electricity').to_numpy() - \
                           batRated
        
        heatLim = df['Heating (kW)'].rolling(window=14400, min_periods=2880)
        coolLim = df['Cooling (kW)'].rolling(window=14400, min_periods=2880)
        waterLim = df['Water Systems (kW)'].rolling(window=14400, min_periods=2880)
            
        df_metric['High heat'] = heatLim - df['Heating (kW)'] + waterLim - df['Water Systems (kW)']
        df_metric['High cool'] = coolLim - df['Cooling (kW)'] + waterLim - df['Water Systems (kW)']

        # Apply smoothing to flexibility bounds
        df_metric['low'] = df_metric['low'].rolling(20, min_periods=0).mean()
        df_metric['High heat'] = df_metric['High heat'].rolling(20, min_periods=0).mean()
        df_metric['High cool'] = df_metric['High cool'].rolling(20, min_periods=0).mean()

        df_metric.to_csv('../../results/summer/metric.csv')

        print('Created metrics for file %s' % file.name)