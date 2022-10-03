# Import libraries to work with DataFrames and financial data
import pandas as pd
import pandas_ta as ta
import numpy as np

# Import libraries to visualiztion for data and financial data
import mplfinance as mpf
import matplotlib.pyplot as plt
import seaborn as sns

# Import Statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.stattools import acf



# Create your own Custom Strategy just comment/uncomment out accordingly
def wrangle_data(sp=pd.read_csv('BKR.csv'),nabors=pd.read_csv('NBR.csv'),djogi=pd.read_csv('oil_and_gas_index.csv',skiprows=1)):

    CustomStrategy = ta.Strategy(
        name="Momo and Volatility",
        description="SMA, BBANDS, RSI, MACD and Volume SMA 20",
        ta=[
            #{"kind": "sma", "length": 12},
            # {"kind": "bbands", "length":12 },
            {"kind": "rsi"},
            #{"kind": "macd"},
            {"kind": "ema" }])

    sp.ta.strategy(CustomStrategy)

    sp.columns=sp.columns.str.lower()

    sp.date=pd.to_datetime(sp.date)

    sp=sp.set_index('date').sort_index()

    nabors.columns= nabors.columns.str.lower()

    nabors.ta.strategy(CustomStrategy)

    nabors.date=pd.to_datetime(nabors.date)

    nabors=nabors.set_index('date').sort_index()


    mpf.plot(nabors,type='candle',volume=True, title= 'Nabors Drilling common stock 180 day',mav=(4,12))

    nabors.rename(columns = {'adj close':'nbrs_adj_close','volume':'nbrs_volume',
    'RSI_14':'nbrs_rsi14','EMA_10':'nbrs_ema_10'}, inplace = True)
    nabors=nabors.drop(columns={'open','high','low','close'})

        #Get data the right size and remove nulls
    djogi=djogi.dropna(how='all')
    djogi=djogi.dropna(how="all",axis=1)
    djogi=djogi.drop(261)
    #make columns lower case and index a datetime index
    djogi.columns=djogi.columns.str.lower()
    djogi.date=pd.to_datetime(djogi.date)
    djogi=djogi.set_index('date').sort_index()

    both_stock_data=sp.merge(nabors,how='inner',left_index=True, right_index=True)

    stocks_and_index=both_stock_data.merge(djogi,how='inner',left_index=True, right_index=True)

    stocks_and_index=stocks_and_index.dropna()

    stocks_and_index=stocks_and_index.round(3)

    print(stocks_and_index.isnull().sum())
    
    print(stocks_and_index.dtypes)
    
    print(stocks_and_index.describe().T)

    mpf.plot(sp,type='candle',volume=True, title= 'Baker Hughes Common Stock 180 day',mav=(4,12))


    return stocks_and_index
    

def tar_var_dist(stocks_and_index):
    plt.hist(stocks_and_index.close,bins=20)
    plt.title('Distribution of target variable')
    plt.xlabel('closing price of Baker Hughes Stock')
    plt.ylabel('occurances')


# Plot

def display_historical_data(stocks_and_index):
    fig, axes = plt.subplots(nrows=6, ncols=2, dpi=120, figsize=(10,6))
    for i, ax in enumerate(axes.flatten()):
        data = stocks_and_index[stocks_and_index.columns[i]]
        ax.plot(data, color='red', linewidth=1)
        # Decorations
        ax.set_title(stocks_and_index.columns[i])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)

    plt.tight_layout()



maxlag=12
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.") 


def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

def compare_forecast_to_actual(stocks_and_index,df_results):
    nobs=7
    fig, axes = plt.subplots(nrows=int((len(stocks_and_index.columns)+1)/2), ncols=2, dpi=150, figsize=(10,15))
    for i, (col,ax) in enumerate(zip(stocks_and_index.columns, axes.flatten())):
        df_results[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
        test[col][-nobs:].plot(legend=True, ax=ax);
        ax.set_title(col + ": Forecast vs Actuals")
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)

    plt.tight_layout();



# def forecast_accuracy(forecast, actual):
#     mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
#     me = np.mean(forecast - actual)             # ME
#     mae = np.mean(np.abs(forecast - actual))    # MAE
#     mpe = np.mean((forecast - actual)/actual)   # MPE
#     rmse = np.mean((forecast - actual)**2)**.5  # RMSE
#     corr = np.corrcoef(forecast, actual)[0,1]   # corr
#     mins = np.amin(np.hstack([forecast[:,None], 
#                               actual[:,None]]), axis=1)
#     maxs = np.amax(np.hstack([forecast[:,None], 
#                               actual[:,None]]), axis=1)
#     minmax = 1 - np.mean(mins/maxs)             # minmax
#     return({'mape':mape, 'me':me, 'mae': mae, 
#             'mpe': mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax})

# def adjust(val, length= 6): return str(val).ljust(length)

# print('Forecast Accuracy of: open')
# accuracy_prod = forecast_accuracy(df_results['open_forecast'].values, test['open'])
# for k, v in accuracy_prod.items():
#     print(adjust(k), ': ', round(v,4))

# print('\nForecast Accuracy of: high')
# accuracy_prod = forecast_accuracy(df_results['high_forecast'].values, test['high'])
# for k, v in accuracy_prod.items():
#     print(adjust(k), ': ', round(v,4))

# print('\nForecast Accuracy of: low')
# accuracy_prod = forecast_accuracy(df_results['low_forecast'].values, test['low'])
# for k, v in accuracy_prod.items():
#     print(adjust(k), ': ', round(v,4))

# print('\nForecast Accuracy of: close')
# accuracy_prod = forecast_accuracy(df_results['close_forecast'].values, test['close'])
# for k, v in accuracy_prod.items():
#     print(adjust(k), ': ', round(v,4))

# print('\nForecast Accuracy of: adj close')
# accuracy_prod = forecast_accuracy(df_results['adj close_forecast'].values, test['adj close'])
# for k, v in accuracy_prod.items():
#     print(adjust(k), ': ', round(v,4))

# print('\nForecast Accuracy of: volume')
# accuracy_prod = forecast_accuracy(df_results['volume_forecast'].values, test['volume'])
# for k, v in accuracy_prod.items():
#     print(adjust(k), ': ', round(v,4))

# print('\nForecast Accuracy of: rsi_14')
# accuracy_prod = forecast_accuracy(df_results['rsi_14_forecast'].values, test['rsi_14'])
# for k, v in accuracy_prod.items():
#     print(adjust(k), ': ', round(v,4))

# print('\nForecast Accuracy of: ema_10')
# accuracy_prod = forecast_accuracy(df_results['ema_10_forecast'].values, test['ema_10'])
# for k, v in accuracy_prod.items():
#     print(adjust(k), ': ', round(v,4))

# print('\nForecast Accuracy of: nbrs_adj_close')
# accuracy_prod = forecast_accuracy(df_results['nbrs_adj_close_forecast'].values, test['nbrs_adj_close'])
# for k, v in accuracy_prod.items():
#     print(adjust(k), ': ', round(v,4))

# print('\nForecast Accuracy of: nbrs_volume')
# accuracy_prod = forecast_accuracy(df_results['nbrs_volume_forecast'].values, test['nbrs_volume'])
# for k, v in accuracy_prod.items():
#     print(adjust(k), ': ', round(v,4))

# print('\nForecast Accuracy of: nbrs_rsi14')
# accuracy_prod = forecast_accuracy(df_results['nbrs_rsi14_forecast'].values, test['nbrs_rsi14'])
# for k, v in accuracy_prod.items():
#     print(adjust(k), ': ', round(v,4))

# print('\nForecast Accuracy of: nbrs_ema_10')
# accuracy_prod = forecast_accuracy(df_results['nbrs_ema_10_forecast'].values, test['nbrs_ema_10'])
# for k, v in accuracy_prod.items():
#     print(adjust(k), ': ', round(v,4))