import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose, MSTL
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt

class FlexibilityMetricPredictor():
    '''
    '''

    def __init__(self):
        '''
        '''
        pass

    def LoadPredictor(self):
        '''
        '''
        self.model = sm.load('arima.pickle')

    def TrainPredictor(self):
        '''
        '''
        details = {
            "name": 'Manufactured Home w/HP',
            "path": './data/SS_Boise_5B_HUD_heatpump_v22',
            "fudge": 2,
            "nameConv": ['LIVING_UNIT1', 'WATER HEATER_UNIT1'],
            "skip": ['Heating:NaturalGas', 'WaterSystems:NaturalGas'],
            "run": 1,
            "timesteps": 20}
        filePath = details['path'] + '_metricsNew.csv'
        timesteps = details['timesteps']

        # Comment/uncomment these based on which metric you're forecasting
        # -------------- Flexibility bounds forecasting --------------
        # The order of this apparently matters, the prediction columns will get switched up in the other order for some reason
        # column_names = ['Date/Time', 'Net Power (kW)', 'HVAC Past State', 'Outdoor Air Temperature (C)',
        #                 'Indoor Air Temperature (C)', 'Hot Water Tank Temperature (C)',
        #                 'High Envelope (kW)', 'Low Envelope (kW)']
        # targetName = ['Low Envelope (kW)', 'High Envelope (kW)']
        # inputs = column_names[:-len(targetName)]
        # metric_abb = 'FB'
        # ------------------------------------------------------------

        # -------------- Energy flexibility forecasting --------------
        # --- Water heater ---
        # column_names = ['Date/Time', 'Outdoor Air Temperature (C)',
        #                 'Hot Water Tank Temperature (C)', 'Water Heater On',
        #                 'Maintainable Duration:Water (min)']
        # targetName = ['Maintainable Duration:Water (min)']
        # inputs = columns_names[:-len(targetName)]
        # metric_abb = 'EF_WH'

        # --- Heating/Cooling ---
        column_names = ['Date/Time', 'HVAC Past State', 'Outdoor Air Temperature (C)',
                        'Indoor Air Temperature (C)', 'Heating On',
                        'Maintainable Duration:Heating (min)']
        targetName = ['Maintainable Duration:Heating (min)']
        inputs = column_names[:-len(targetName)]
        metric_abb = 'EF_HVAC'
        # ------------------------------------------------------------

        # Set up file structure for saving models and figures
        # Path('Saved Models/Forecast/' + metric_abb).mkdir(parents=True, exist_ok=True)
        # Path('Saved Figures/Forecast/' + metric_abb).mkdir(parents=True, exist_ok=True)

        full_dataset = pd.read_csv(filePath, usecols=column_names)
        fig = plt.figure(figsize=(12,5))
        plt.plot(full_dataset['Maintainable Duration:Heating (min)'])
        fig.savefig('Saved Figures/full label.png')
        
        raw_dataset = pd.read_csv(filePath, usecols=column_names, nrows=14400)
        
        dataset = raw_dataset.copy()
        # dataset = raw_dataset.iloc[0:20*24*timesteps].copy()
        print(dataset.describe().transpose())

        # Dataset feature engineering
        dataset['HVAC Past State'].replace([1,2,3,4], [-1,-1,1,1], inplace=True)
        
        date_time = pd.DatetimeIndex(dataset.pop('Date/Time'), freq='infer')
        dataset.index = date_time
        # timestamp_s = date_time.map(pd.Timestamp.timestamp)
        
        # hour = timesteps
        # day = timesteps * 24 * 60 * 60
        # year = day * 365
        
        # dataset['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        # dataset['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))

        # Remove labels before normalizing
        labels_temp = dataset[targetName].copy()
        dataset.drop(targetName, axis=1, inplace=True)
        
        # Normalization
        # Probably should be done using moving averages
        train_mean = dataset.mean()
        train_std = dataset.std()
        
        dataset = (dataset - train_mean) / train_std

        dataset = pd.concat([dataset, labels_temp], axis=1)

        print(dataset.describe().transpose())

        print(dataset.index.freq)

        # Split the data
        column_indices = {name: i for i, name in enumerate(dataset.columns)}
        
        n = len(dataset)
        train_df = dataset[0:int(n*0.9)]
        self.test_df = dataset[int(n*0.9):]              
        
        num_labels = len(targetName)
        label_indices = [column_indices[name] for name in targetName]

        train_labels = train_df.pop(targetName[0])
        fig, ax = plt.subplots(figsize=(12,8))
        train_labels.plot(ax=ax)
        fig.savefig('Saved Figures/label.png')

        fig, ax = plt.subplots(1,2, figsize=(12,8))
        sm.graphics.tsa.plot_acf(train_labels.values.squeeze(), lags=40, ax=ax[0])
        sm.graphics.tsa.plot_pacf(train_labels, lags=40, ax=ax[1])
        fig.savefig('Saved Figures/autocorrelations.png')

        adfuller_test(train_labels)

        result = seasonal_decompose(train_labels, model='additive', period=20)
        fig = result.plot()
        fig.savefig('Saved Figures/seasonal decomp.png')

        stl = MSTL(train_labels, periods=(6, 20*24))
        res = stl.fit()
        fig = res.plot()
        fig.savefig('Saved Figures/mstl res.png')

        # Model setup and fitting
        model = ARIMA(train_labels, exog=train_df, order=(3,1,1))
        # model = sm.tsa.statespace.SARIMAX(train_labels, exog=train_df, order=(1,1,1), seasonal_order=(1,1,1,12))

        self.model = model.fit()

        self.model.save('model.pickle')
        
    def PlotPredictor(self):
        resid = self.model.resid

        fig, ax = plt.subplots(figsize=(12,8))
        self.model.resid.plot(ax=ax)
        fig.savefig('Saved Figures/residuals.png')

        fig, ax = plt.subplots(figsize=(12,8))
        qqplot(resid, line="q", ax=ax, fit=True)
        fig.savefig('Saved Figures/qqplot.png')

        fig, ax = plt.subplots(1,2, figsize=(12,8))
        sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax[0])
        sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax[1])
        fig.savefig('Saved Figures/residual autocorrelations.png')

        self.test_df['forecast'] = self.model.predict(start=self.test_df.index[0], end=self.test_df.index[-1], dynamic=True,
                                                      exog=self.test_df[['HVAC Past State', 'Outdoor Air Temperature (C)', 'Indoor Air Temperature (C)', 'Heating On']])

        fig, ax = plt.subplots(figsize=(12,8))
        self.test_df[['Maintainable Duration:Heating (min)', 'forecast']].plot(ax=ax)
        fig.savefig('Saved Figures/forecast.png')

    def PredictMetrics(self):
        '''
        '''
        pass

def adfuller_test(sales):
    result = adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label + ' : ' + str(value))
    if result[1] <= 0.05:
        print("P value is less than 0.05 that means we can reject the null hypothesis(Ho). Therefore we can conclude that data has no unit root and is stationary")
    else:
        print("Weak evidence against null hypothesis that means time series has a unit root which indicates that it is non-stationary ")