import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import talib
from datetime import datetime
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from xgboost import plot_importance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import TimeSeriesSplit
import pickle
import requests
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
import time
import urllib.parse
import hashlib
import hmac
from scipy.signal import argrelextrema

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

api_key = 'IbIgJihiEgl4rEjWnOFazg7F4YVzJXVG8if3iKcGsurgspgblDN2F73XMPdUzOcH'
api_sec = 'kN7vx7TDa207GdVbE5DL6Vf6f8xs2nXaYdX0xlKJQuibieOv2laiMQ53rQoUZGjc'

def load_and_preprocess_data(filepath, window_size=2):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, names=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Num Trades', 'Label', 'ATR', 'Price_Change', 'Buy_Threshold', 'Sell_Threshold'])
    df.drop(['ATR', 'Price_Change', 'Buy_Threshold', 'Sell_Threshold'], axis=1, inplace=True)
    df.drop(0, inplace=True)
    #Drop useless row
    df = df.set_index('Timestamp')
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)

    df["Returns"] = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(window=24).std()

    #Replaces 0's with nan to avoid infinity being present in Volume_Change
    #Since the label column has 0's present, we need to make sure that they are not replaced
    columns_to_replace = df.columns.difference(['Label'])
    df[columns_to_replace] = df[columns_to_replace].replace(0, pd.NA)
    df["Volume_Change"] = df["Volume"].pct_change()

    df['LOW_EMA'] = talib.EMA(df['Close'], timeperiod=9)
    df['HIGH_EMA'] = talib.EMA(df['Close'], timeperiod=21)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['Aroon'] = talib.AROONOSC(df['High'], df['Low'], timeperiod=14)

    upper, middle, lower = talib.BBANDS(df['Close'], 20)
    df['BB_Width'] = upper - lower

    macd, macdsignal, macdhist = talib.MACDFIX(df['Close'])
    df['MACD'] = macd
    df['MACDSignal'] = macdsignal
    df['MACDHist'] = macdhist

    df['PP'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
    df['R1'] = 2 * df['PP'] - df['Low'].shift(1)
    df['R2'] = df['PP'] + (df['High'].shift(1) - df['Low'].shift(1))
    df['S1'] = 2 * df['PP'] - df['High'].shift(1)
    df['S2'] = df['PP'] - (df['High'].shift(1) - df['Low'].shift(1))

    # Wave analysis
    n = 1
    df['Smoothed_Close'] = df['Close'].rolling(window=n).mean()

    max_indices = argrelextrema(df['Smoothed_Close'].values, np.greater_equal, order=n)[0]
    min_indices = argrelextrema(df['Smoothed_Close'].values, np.less_equal, order=n)[0]

    df['Wave_Amplitude'] = np.nan
    df['Wave_Direction'] = 0
    df['Wave_Length_Bars'] = np.nan
    df['Wave_Length_Time'] = pd.NaT

    last_peak_idx = 0
    last_trough_idx = 0

    for i in range(len(df)):
        idx = df.index[i]
        current_close = df['Smoothed_Close'].iloc[i]

        if last_trough_idx >= last_peak_idx:  # need this extra condition to make sure the amplitude and length are not overwritten by the peak code block (seperating peaks/troughs)
            trough_close = df['Smoothed_Close'].iloc[last_trough_idx]
            df.loc[idx, 'Wave_Amplitude'] = current_close - trough_close

            bars_length = i - last_trough_idx
            time_length = idx - df.index[last_trough_idx]
            df.loc[idx, 'Wave_Length_Bars'] = bars_length
            df.loc[idx, 'Wave_Length_Time'] = time_length

        if last_peak_idx >= last_trough_idx:
            peak_close = df['Smoothed_Close'].iloc[last_peak_idx]
            df.loc[idx, 'Wave_Amplitude'] = peak_close - current_close

            bars_length = i - last_peak_idx
            time_length = idx - df.index[last_peak_idx]
            df.loc[idx, 'Wave_Length_Bars'] = bars_length
            df.loc[idx, 'Wave_Length_Time'] = time_length

        if i in max_indices:
            df.loc[idx, 'Wave_Direction'] = -1
            last_peak_idx = i

        if i in min_indices:
            df.loc[idx, 'Wave_Direction'] = 1
            last_trough_idx = i

    df['Wave_Length_Time'] = df['Wave_Length_Time'] / pd.Timedelta(minutes=1)

    # calculate extra wave features
    df['Wave_Velocity'] = df['Wave_Amplitude'] / df['Wave_Length_Time']
    df['Wave_Frequency'] = 1 / df['Wave_Length_Time']
    df['Wave_Sharpness'] = df['Wave_Amplitude'] / (df['Wave_Length_Time'] ** 2)
    df['Wave_Slope'] = df['Wave_Amplitude'] / df['Wave_Length_Bars']
    df['Wave_Energy'] = df['Wave_Amplitude'] ** 2
    df['Wave_Acceleration'] = df['Wave_Slope'].diff() / df['Wave_Length_Bars']
    df['Wave_Strength'] = df['Wave_Slope'] * df['Wave_Direction']
    df['Normalized_Amplitude'] = df['Wave_Amplitude'] / df['Wave_Amplitude'].rolling(window=window_size).mean()
    df['Normalized_Slope'] = df['Wave_Slope'] / df['Wave_Slope'].rolling(window=window_size).mean()

    df.dropna(subset=['Smoothed_Close'], inplace=True)

    #Need this because there was a weird error were the data in these columns were not classified as floats, this caused a problem with the pipeline as I'm not using a target encoder
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df['Num Trades'] = pd.to_numeric(df['Num Trades'], errors='coerce')
    df['Returns'] = pd.to_numeric(df['Returns'], errors='coerce')

    #calculate aggregate features
    df['Close_Mean'] = df['Close'].rolling(window=window_size).mean()
    df['High_Mean'] = df['High'].rolling(window=window_size).mean()
    df['Open_Mean'] = df['Open'].rolling(window=window_size).mean()
    df['Low_Mean'] = df['Low'].rolling(window=window_size).mean()
    df['Returns_Mean'] = df['Returns'].rolling(window=window_size).mean()
    df['Volatility_Mean'] = df['Volatility'].rolling(window=window_size).mean()
    df['LOW_EMA_Mean'] = df['LOW_EMA'].rolling(window=window_size).mean()
    df['HIGH_EMA_Mean'] = df['HIGH_EMA'].rolling(window=window_size).mean()
    df['RSI_Mean'] = df['RSI'].rolling(window=window_size).mean()
    df['Aroon_Mean'] = df['Aroon'].rolling(window=window_size).mean()
    df['BB_Width_Mean'] = df['BB_Width'].rolling(window=window_size).mean()
    df['MACD_Mean'] = df['MACD'].rolling(window=window_size).mean()
    df['MACDSignal_Mean'] = df['MACDSignal'].rolling(window=window_size).mean()
    df['MACDHist_Mean'] = df['MACDHist'].rolling(window=window_size).mean()
    df['PP_Mean'] = df['PP'].rolling(window=window_size).mean()
    df['S1_Mean'] = df['S1'].rolling(window=window_size).mean()
    df['S2_Mean'] = df['S2'].rolling(window=window_size).mean()
    df['R1_Mean'] = df['R1'].rolling(window=window_size).mean()
    df['R2_Mean'] = df['R2'].rolling(window=window_size).mean()
    df['Wave_Direction_Mean'] = df['Wave_Direction'].rolling(window=window_size).mean()
    df['Wave_Amplitude_Mean'] = df['Wave_Amplitude'].rolling(window=window_size).mean()
    df['Wave_Velocity_Mean'] = df['Wave_Velocity'].rolling(window=window_size).mean()
    df['Wave_Frequency_Mean'] = df['Wave_Frequency'].rolling(window=window_size).mean()
    df['Wave_Sharpness_Mean'] = df['Wave_Sharpness'].rolling(window=window_size).mean()
    df['Wave_Slope_Mean'] = df['Wave_Slope'].rolling(window=window_size).mean()
    df['Wave_Energy_Mean'] = df['Wave_Energy'].rolling(window=window_size).mean()
    df['Wave_Acceleration_Mean'] = df['Wave_Acceleration'].rolling(window=window_size).mean()
    df['Wave_Strength_Mean'] = df['Wave_Strength'].rolling(window=window_size).mean()

    df['Close_Dev'] = df['Close'].rolling(window=window_size).std()
    df['High_Dev'] = df['High'].rolling(window=window_size).std()
    df['Open_Dev'] = df['Open'].rolling(window=window_size).std()
    df['Low_Dev'] = df['Low'].rolling(window=window_size).std()
    df['Returns_Dev'] = df['Returns'].rolling(window=window_size).std()
    df['Volatility_Dev'] = df['Volatility'].rolling(window=window_size).std()
    df['LOW_EMA_Dev'] = df['LOW_EMA'].rolling(window=window_size).std()
    df['HIGH_EMA_Dev'] = df['HIGH_EMA'].rolling(window=window_size).std()
    df['RSI_Dev'] = df['RSI'].rolling(window=window_size).std()
    df['Aroon_Dev'] = df['Aroon'].rolling(window=window_size).std()
    df['BB_Width_Dev'] = df['BB_Width'].rolling(window=window_size).std()
    df['MACD_Dev'] = df['MACD'].rolling(window=window_size).std()
    df['MACDSignal_Dev'] = df['MACDSignal'].rolling(window=window_size).std()
    df['MACDHist_Dev'] = df['MACDHist'].rolling(window=window_size).std()
    df['PP_Dev'] = df['PP'].rolling(window=window_size).std()
    df['S1_Dev'] = df['S1'].rolling(window=window_size).std()
    df['S2_Dev'] = df['S2'].rolling(window=window_size).std()
    df['R1_Dev'] = df['R1'].rolling(window=window_size).std()
    df['R2_Dev'] = df['R2'].rolling(window=window_size).std()
    df['Wave_Direction_Dev'] = df['Wave_Direction'].rolling(window=window_size).std()
    df['Wave_Amplitude_Dev'] = df['Wave_Amplitude'].rolling(window=window_size).std()
    df['Wave_Velocity_Mean'] = df['Wave_Velocity'].rolling(window=window_size).std()
    df['Wave_Frequency_Mean'] = df['Wave_Frequency'].rolling(window=window_size).std()
    df['Wave_Sharpness_Mean'] = df['Wave_Sharpness'].rolling(window=window_size).std()
    df['Wave_Slope_Mean'] = df['Wave_Slope'].rolling(window=window_size).std()
    df['Wave_Energy_Mean'] = df['Wave_Energy'].rolling(window=window_size).std()
    df['Wave_Acceleration_Mean'] = df['Wave_Acceleration'].rolling(window=window_size).std()
    df['Wave_Strength_Mean'] = df['Wave_Strength'].rolling(window=window_size).std()

    df['Close_Drawdown'] = (df['Close'] / df['Close'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['High_Drawdown'] = (df['High'] / df['High'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['Open_Drawdown'] = (df['Open'] / df['Open'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['Low_Drawdown'] = (df['Low'] / df['Low'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['Returns_Drawdown'] = (df['Returns'] / df['Returns'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['Volatility_Drawdown'] = (df['Volatility'] / df['Volatility'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['LOW_EMA_Drawdown'] = (df['LOW_EMA'] / df['LOW_EMA'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['HIGH_EMA_Drawdown'] = (df['HIGH_EMA'] / df['HIGH_EMA'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['RSI_Drawdown'] = (df['RSI'] / df['RSI'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['Aroon_Drawdown'] = (df['Aroon'] / df['Aroon'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['BB_Width_Drawdown'] = (df['BB_Width'] / df['BB_Width'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['MACD_Drawdown'] = (df['MACD'] / df['MACD'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['MACDSignal_Drawdown'] = (df['MACDSignal'] / df['MACDSignal'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['MACDHist_Drawdown'] = (df['MACDHist'] / df['MACDHist'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['PP_Drawdown'] = (df['PP'] / df['PP'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['S1_Drawdown'] = (df['S1'] / df['S1'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['S2_Drawdown'] = (df['S2'] / df['S2'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['R1_Drawdown'] = (df['R1'] / df['R1'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['R2_Drawdown'] = (df['R2'] / df['R2'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['Wave_Direction_Drawdown'] = (df['Wave_Direction'] / df['Wave_Direction'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['Wave_Amplitude_Drawdown'] = (df['Wave_Amplitude'] / df['Wave_Amplitude'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['Wave_Velocity_Drawdown'] = (df['Wave_Velocity'] / df['Wave_Velocity'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['Wave_Frequency_Drawdown'] = (df['Wave_Frequency'] / df['Wave_Frequency'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['Wave_Sharpness_Drawdown'] = (df['Wave_Sharpness'] / df['Wave_Sharpness'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['Wave_Slope_Drawdown'] = (df['Wave_Slope'] / df['Wave_Slope'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['Wave_Energy_Drawdown'] = (df['Wave_Energy'] / df['Wave_Energy'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['Wave_Acceleration_Drawdown'] = (df['Wave_Acceleration'] / df['Wave_Acceleration'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['Wave_Strength_Drawdown'] = (df['Wave_Strength'] / df['Wave_Strength'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()

    df.dropna(inplace=True)

    training_df = df[(df.index >= "2023-01-02")]
    eval_df = df[((df.index >= "2022-01-01") & (df.index <= "2022-06-01"))]
    out_of_training_df = df[((df.index >= "2022-06-02") & (df.index <= "2023-01-01"))]

    return training_df, eval_df, out_of_training_df

def train_hmm(data, features, scaler, n_components=3):
    print(f"Training HMM with {n_components} components...")
    X = data[features].values

    print("Normalizing features...")
    X_scaled = scaler.fit_transform(X)

    print("Fitting HMM model...")
    model = hmm.GaussianHMM(n_components=n_components, covariance_type='full', n_iter=500, random_state=42, verbose=True)
    model.fit(X_scaled)

    print("HMM training complete")

    filepath = "hmm_model.pkl"
    with open("hmm_model.pkl", "wb") as f:
        pickle.dump(model, f)

    return filepath

def load_hmm(filepath):
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model

def predict_states(model, data, features, scaler):
     X = data[features].values
     X_scaled = scaler.transform(X)
     states = model.predict(X_scaled)
     print(f"States precicted. Unique states: {np.unique(states)}")
     return states

def plot_results_hmm(data, states, n_components):
    print("Plotting results...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,10), sharex=True)

    ax1.plot(data.index, data['Close'])
    ax1.set_title('Bitcoin Price and HMM States')
    ax1.set_ylabel('Price')

    for state in range(n_components):
        mask = (states == state)
        ax1.fill_between(data.index, data['Close'].min(), data['Close'].max(), where=mask, alpha=0.3, label=f'State {state}')
    ax1.legend()

    ax2.plot(data.index, data['Returns'])
    ax2.set_title('Bitcoin Returns')
    ax2.set_ylabel('Returns')
    ax2.set_xlabel('Date')

    plt.tight_layout()
    print("Saving plot...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    plt.savefig('plots/hmm' + str(timestamp))

def plot_results_rf(data, labels_true, labels_pred):
    print("Plotting results...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,10), sharex=True)

    ax1.plot(data.index, data['Close'])
    ax1.set_title('Bitcoin Price and Real Labels')
    ax1.set_ylabel('Price')

    for label in range(3):
        mask = (labels_true == label)
        ax1.fill_between(data.index, data['Close'].min(), data['Close'].max(), where=mask, alpha=0.3, label=f'Signal {label}')
    ax1.legend()

    ax2.plot(data.index, data['Close'])
    ax2.set_title('Bitcoin Price and Predicted Labels')
    ax2.set_ylabel('Price')

    for label in range(3):
        mask = (labels_pred == label)
        ax2.fill_between(data.index, data['Close'].min(), data['Close'].max(), where=mask, alpha=0.3, label=f'Signal {label}')
    ax2.legend()

    plt.tight_layout()
    print("Saving plot...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    plt.savefig('plots/rf' + str(timestamp))

def train_xgboost(data_train, labels_train, data_test, labels_test):
    tscv = TimeSeriesSplit(n_splits=5)

    pipeline = ImbPipeline(steps=[
        ('smote', SMOTE(random_state=42)),
        ('xgb', XGBClassifier(random_state=42, eval_metric='mlogloss', early_stopping_rounds=50))
    ])

    search_space = {
        'xgb__max_depth': Integer(2, 8),
        'xgb__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
        'xgb__subsample': Real(0.5, 1.0),
        'xgb__colsample_bytree': Real(0.5, 1.0),
        'xgb__colsample_bylevel': Real(0.5, 1.0),
        'xgb__colsample_bynode': Real(0.5, 1.0),
        'xgb__reg_alpha': Real(0.0, 10.0),
        'xgb__reg_lambda': Real(0.0, 10.0),
        'xgb__gamma': Real(0.0, 10.0),
        'xgb__n_estimators': Integer(100, 500),
        'xgb__min_child_weight': Integer(1, 15)
    }
    opt = BayesSearchCV(pipeline, search_space, cv=tscv, n_iter=3, scoring='balanced_accuracy', random_state=42)

    opt.fit(data_train, labels_train, xgb__eval_set=[(data_test, labels_test)])
    print("Best estimator: ", opt.best_estimator_)
    print("Best score: ", opt.best_score_)
    print("Best params: ", opt.best_params_)
    print(opt.score(data_test, labels_test))
    labels_pred = opt.predict(data_test)
    labels_pred = labels_pred.tolist()
    print(labels_pred.count(0))
    print(labels_pred.count(1))
    print(labels_pred.count(2))

    xgboost_step = opt.best_estimator_.steps[1]
    xgboost_model = xgboost_step[1]
    plot_importance(xgboost_model, max_num_features=50)
    plt.show()

    #save the model
    filepath = "xgboost_pipeline.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(opt.best_estimator_, f)

    return filepath

def load_xgboost(filepath):
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model

def get_binanceus_signature(data, secret):
    postdata = urllib.parse.urlencode(data)
    message = postdata.encode()
    byte_key = bytes(secret, 'UTF-8')
    mac = hmac.new(byte_key, message, hashlib.sha256).hexdigest()
    return mac

def exchange_btc(side, quoteOrderQty):
    data = {
        "symbol": 'BTCUSDC',
        "side": side,
        "type": 'MARKET',
        "quoteOrderQty": quoteOrderQty,
        "timestamp": int(round(time.time() * 1000))
    }
    headers = {
        'X-MBX-APIKEY': api_key
    }
    signature = get_binanceus_signature(data, api_sec)
    payload = {
        **data,
        "signature": signature,
    }
    req = requests.post(('https://api.binance.us/api/v3/order'), headers=headers, data=payload)
    return req.text

def get_historical_data(count=1000):
    url = 'https://api.binance.us/api/v3/klines'
    parameters = {
        'symbol': 'BTCUSDC',
        'interval': '3m',
        'limit': f'{count}'
    }
    headers = {
        'X-MBX-APIKEY': api_key,
    }

    session = requests.Session()
    session.headers.update(headers)

    try:
        response = session.get(url, params=parameters)
        data = json.loads(response.text)
    except (ConnectionError, Timeout, TooManyRedirects) as e:
        print(e)

    df = pd.DataFrame(columns=["Open", "High", "Low", "Close"])

    for i in range(len(data)):
        df.loc[len(df)] = {"Open": float(data[i][1]),
                           "High": float(data[i][2]),
                           "Low": float(data[i][3]),
                           "Close": float(data[i][4])}

    return df

def calculate_indicators(df, window_size=2):
    df["Returns"] = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(window=24).std()
    df['LOW_EMA'] = talib.EMA(df['Close'], timeperiod=9)
    df['HIGH_EMA'] = talib.EMA(df['Close'], timeperiod=21)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['Aroon'] = talib.AROONOSC(df['High'], df['Low'], timeperiod=14)

    upper, middle, lower = talib.BBANDS(df['Close'], 20)
    df['BB_Width'] = upper - lower

    macd, macdsignal, macdhist = talib.MACDFIX(df['Close'])
    df['MACD'] = macd
    df['MACDSignal'] = macdsignal
    df['MACDHist'] = macdhist

    df['PP'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
    df['R1'] = 2 * df['PP'] - df['Low'].shift(1)
    df['R2'] = df['PP'] + (df['High'].shift(1) - df['Low'].shift(1))
    df['S1'] = 2 * df['PP'] - df['High'].shift(1)
    df['S2'] = df['PP'] - (df['High'].shift(1) - df['Low'].shift(1))

    # calculate aggregate features
    df['Close_Mean'] = df['Close'].rolling(window=window_size).mean()
    df['High_Mean'] = df['High'].rolling(window=window_size).mean()
    df['Open_Mean'] = df['Open'].rolling(window=window_size).mean()
    df['Low_Mean'] = df['Low'].rolling(window=window_size).mean()
    df['Returns_Mean'] = df['Returns'].rolling(window=window_size).mean()
    df['Volatility_Mean'] = df['Volatility'].rolling(window=window_size).mean()
    df['LOW_EMA_Mean'] = df['LOW_EMA'].rolling(window=window_size).mean()
    df['HIGH_EMA_Mean'] = df['HIGH_EMA'].rolling(window=window_size).mean()
    df['RSI_Mean'] = df['RSI'].rolling(window=window_size).mean()
    df['Aroon_Mean'] = df['Aroon'].rolling(window=window_size).mean()
    df['BB_Width_Mean'] = df['BB_Width'].rolling(window=window_size).mean()
    df['MACD_Mean'] = df['MACD'].rolling(window=window_size).mean()
    df['MACDSignal_Mean'] = df['MACDSignal'].rolling(window=window_size).mean()
    df['MACDHist_Mean'] = df['MACDHist'].rolling(window=window_size).mean()

    df['Close_Dev'] = df['Close'].rolling(window=window_size).std()
    df['High_Dev'] = df['High'].rolling(window=window_size).std()
    df['Open_Dev'] = df['Open'].rolling(window=window_size).std()
    df['Low_Dev'] = df['Low'].rolling(window=window_size).std()
    df['Returns_Dev'] = df['Returns'].rolling(window=window_size).std()
    df['Volatility_Dev'] = df['Volatility'].rolling(window=window_size).std()
    df['LOW_EMA_Dev'] = df['LOW_EMA'].rolling(window=window_size).std()
    df['HIGH_EMA_Dev'] = df['HIGH_EMA'].rolling(window=window_size).std()
    df['RSI_Dev'] = df['RSI'].rolling(window=window_size).std()
    df['Aroon_Dev'] = df['Aroon'].rolling(window=window_size).std()
    df['BB_Width_Dev'] = df['BB_Width'].rolling(window=window_size).std()
    df['MACD_Dev'] = df['MACD'].rolling(window=window_size).std()
    df['MACDSignal_Dev'] = df['MACDSignal'].rolling(window=window_size).std()
    df['MACDHist_Dev'] = df['MACDHist'].rolling(window=window_size).std()

    df['Close_Drawdown'] = (df['Close'] / df['Close'].rolling(window=window_size).max() - 1).rolling(
        window=window_size).min()
    df['High_Drawdown'] = (df['High'] / df['High'].rolling(window=window_size).max() - 1).rolling(
        window=window_size).min()
    df['Open_Drawdown'] = (df['Open'] / df['Open'].rolling(window=window_size).max() - 1).rolling(
        window=window_size).min()
    df['Low_Drawdown'] = (df['Low'] / df['Low'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['Returns_Drawdown'] = (df['Returns'] / df['Returns'].rolling(window=window_size).max() - 1).rolling(
        window=window_size).min()
    df['Volatility_Drawdown'] = (df['Volatility'] / df['Volatility'].rolling(window=window_size).max() - 1).rolling(
        window=window_size).min()
    df['LOW_EMA_Drawdown'] = (df['LOW_EMA'] / df['LOW_EMA'].rolling(window=window_size).max() - 1).rolling(
        window=window_size).min()
    df['HIGH_EMA_Drawdown'] = (df['HIGH_EMA'] / df['HIGH_EMA'].rolling(window=window_size).max() - 1).rolling(
        window=window_size).min()
    df['RSI_Drawdown'] = (df['RSI'] / df['RSI'].rolling(window=window_size).max() - 1).rolling(window=window_size).min()
    df['Aroon_Drawdown'] = (df['Aroon'] / df['Aroon'].rolling(window=window_size).max() - 1).rolling(
        window=window_size).min()
    df['BB_Width_Drawdown'] = (df['BB_Width'] / df['BB_Width'].rolling(window=window_size).max() - 1).rolling(
        window=window_size).min()
    df['MACD_Drawdown'] = (df['MACD'] / df['MACD'].rolling(window=window_size).max() - 1).rolling(
        window=window_size).min()
    df['MACDSignal_Drawdown'] = (df['MACDSignal'] / df['MACDSignal'].rolling(window=window_size).max() - 1).rolling(
        window=window_size).min()
    df['MACDHist_Drawdown'] = (df['MACDHist'] / df['MACDHist'].rolling(window=window_size).max() - 1).rolling(
        window=window_size).min()

    return df

def convert_data_to_windows(data, window_size=2):
    rows = []
    for i in range(len(data) - window_size - 1):
        row = {}

        for feature in data.columns:
            if feature != 'Label':
                for t in range(window_size):
                    row[f"{feature}_t-{window_size-t}"] = data[feature].iloc[i + t + 1]

        row['Label'] = data['Label'].iloc[i + window_size + 1]
        rows.append(row)

    window_data = pd.DataFrame(rows)

    return window_data

def trading_simulation(data, starting_money=500, buy_percentage=0.1):
    money = starting_money
    bitcoin = 0
    labels = data['Label'].tolist()
    close_prices = data['Close_t-1'].tolist()
    sell_order = False
    buy_order = False
    previous_trade_assets = money

    #0 is sell, 2 is buy
    count = 0
    for label in labels:
        if label == 0:
            sell_order = True
        elif label == 2:
            buy_order = True

        if sell_order and label == 2:
            money += bitcoin * close_prices[count]
            print(f"Sold {bitcoin} bitcoin at {close_prices[count]}")
            bitcoin = 0
            sell_order = False
            # total_assets = money + bitcoin * close_prices[count]
            # if total_assets < previous_trade_assets:
            #     print("Loss detected on this trade. Stopping simulation.")
            #     break
            # previous_trade_assets = total_assets

        if buy_order and label == 0:
            bitcoin += money * buy_percentage / close_prices[count]
            money -= money * buy_percentage
            buy_order = False
            print(f"Bought {bitcoin} bitcoin at {close_prices[count]}")
            # total_assets = money + bitcoin * close_prices[count]
            # if total_assets < previous_trade_assets:
            #     print("Loss detected on this trade. Stopping simulation.")
            #     break
            # previous_trade_assets = total_assets

        total_assets = money + bitcoin * close_prices[count]
        print(f"Money: {money}, Bitcoin: {bitcoin}, Count: {count}")
        count += 1

    print("Final money: ", money)
    print("Profit: ", money - starting_money)

def get_next_interval(interval_seconds):
    now = time.time()
    next_interval = ((now // interval_seconds) + 1) * interval_seconds
    return next_interval - now

print("Starting main execution...")
window_size = 10
data_train, data_eval, data_test = load_and_preprocess_data("btc_15m_data_2018_to_2024-2024-10-10_labeled.csv", window_size)
scaler = StandardScaler()
hmm_features = ["Close", "High", "Low", "Open", "LOW_EMA", "HIGH_EMA", "RSI", "Aroon", "BB_Width", "MACD", "MACDSignal", "MACDHist", "Volatility", "Returns", "Wave_Direction", "Wave_Amplitude"]
xg_features = ["Volatility", "Returns", "Wave_Direction", "Wave_Amplitude", "Wave_Velocity", "Wave_Frequency", "Wave_Sharpness", "Wave_Energy", "Wave_Acceleration", "Wave_Strength", "State", "MACD", "MACDHist", "MACDSignal", "Label"]
agg_features = ["Wave_Direction_Mean", "Wave_Direction_Dev", "Wave_Direction_Drawdown", "Wave_Amplitude_Mean", "Wave_Amplitude_Dev", "Wave_Amplitude_Drawdown", "Returns_Mean", "Returns_Dev", "Returns_Drawdown",
                "Volatility_Mean", "Volatility_Dev", "Volatility_Drawdown", "MACD_Mean", "MACD_Dev", "MACD_Drawdown", "MACDHist_Mean", "MACDHist_Dev", "MACDHist_Drawdown", "MACDSignal_Mean", "MACDSignal_Dev", "MACDSignal_Drawdown",
                "Wave_Velocity_Mean", "Wave_Velocity_Dev", "Wave_Velocity_Drawdown", "Wave_Frequency_Mean", "Wave_Frequency_Dev", "Wave_Frequency_Drawdown", "Wave_Sharpness_Mean", "Wave_Sharpness_Dev", "Wave_Sharpness_Drawdown",
                "Wave_Energy_Mean", "Wave_Energy_Dev", "Wave_Energy_Drawdown", "Wave_Acceleration_Mean", "Wave_Acceleration_Dev", "Wave_Acceleration_Drawdown", "Wave_Strength_Mean", "Wave_Strength_Dev", "Wave_Strength_Drawdown"]


hmm_path = train_hmm(data_train, hmm_features, scaler, 15)
hmm_model = load_hmm(hmm_path)
print("Predicting states...")
states = predict_states(hmm_model, data_train, hmm_features, scaler)
states_eval = predict_states(hmm_model, data_eval, hmm_features, scaler)
states_out = predict_states(hmm_model, data_test, hmm_features, scaler)
data_train['State'] = states
data_eval['State'] = states_eval
data_test['State'] = states_out

#Build xgboost model
xg_data_train = convert_data_to_windows(data_train[xg_features], window_size)
xg_data_eval = convert_data_to_windows(data_eval[xg_features], window_size)
xg_data_test = convert_data_to_windows(data_test[xg_features], window_size)
#dropping the first 50 rows to line up data with the xgboost data
data_train = data_train.drop(index=data_train.index[:window_size])
data_eval = data_eval.drop(index=data_eval.index[:window_size])
data_test = data_test.drop(index=data_test.index[:window_size])

#need to reset index so that the concat works properly
xg_data_train = xg_data_train.reset_index(drop=True)
data_train = data_train.reset_index(drop=True)
xg_data_eval = xg_data_eval.reset_index(drop=True)
data_eval = data_eval.reset_index(drop=True)
xg_data_test = xg_data_train.reset_index(drop=True)
data_test = data_train.reset_index(drop=True)
xg_data_train = pd.concat([xg_data_train, data_train[agg_features]], axis=1)
xg_data_eval = pd.concat([xg_data_eval, data_eval[agg_features]], axis=1)
xg_data_test = pd.concat([xg_data_test, data_test[agg_features]], axis=1)

xg_data_train.dropna(inplace=True)
xg_data_eval.dropna(inplace=True)
xg_data_test.dropna(inplace=True)

xg_labels_train = xg_data_train.pop('Label').tolist()
xg_labels_train = [int(x) for x in xg_labels_train]
xg_labels_eval = xg_data_eval.pop('Label').tolist()
xg_labels_eval = [int(x) for x in xg_labels_eval]
xg_labels_test = xg_data_test.pop('Label').tolist()
xg_labels_test = [int(x) for x in xg_labels_test]

xgboost_path = train_xgboost(xg_data_train, xg_labels_train, xg_data_eval, xg_labels_eval)
#xgboost_path = "xgboost_pipeline.pkl"
xgboost_model = load_xgboost(xgboost_path)
xg_labels_pred = xgboost_model.predict(xg_data_test)
xg_labels_pred = xg_labels_pred.tolist()

# #Filtering lists so that there is only entries where either the real list or the predicted list has a 0 or 2 in them
# #Since buying and selling are the more important predictions in an actual algo trader I want to see only the buy/sell accuracy
buy_sell_label, buy_sell_label_pred = zip(*[(x, y) for x, y in zip(xg_labels_test, xg_labels_pred) if (x in [0, 2] or y in [0, 2])])
buy_sell_label = list(buy_sell_label)
buy_sell_label_pred = list(buy_sell_label_pred)
buy_sell_accuracy = accuracy_score(buy_sell_label, buy_sell_label_pred)
print(f"buy_sell accuracy: {buy_sell_accuracy}")

# recent_df = get_historical_data(1000)
# recent_df = calculate_indicators(recent_df)
# recent_df = recent_df.dropna()
# recent_df['State'] = predict_states(hmm_model, recent_df, hmm_features, scaler)
#
# #Need this to make recent_df compatible with convert_data_to_windows method
# recent_df['Label'] = 0
# recent_df_windowed = convert_data_to_windows(recent_df[xg_features], window_size)
#
# recent_df = recent_df.drop(index=recent_df.index[:window_size])
# recent_df = recent_df.reset_index(drop=True)
# recent_df_windowed = recent_df_windowed.reset_index(drop=True)
# recent_df = pd.concat([recent_df_windowed, recent_df[agg_features]], axis=1)
# recent_df.drop(['Label'], axis=1, inplace=True)
#
# labels_recent_pred = xgboost_model.predict(recent_df)
# labels_recent_pred = labels_recent_pred.tolist()
# recent_df['Label'] = labels_recent_pred
# recent_df['PriceDiff'] = recent_df['Close_t-1'].diff()
#
# trading_df = recent_df[['Close_t-1', 'PriceDiff', 'Label']]
# print(trading_df)
#trading_simulation(trading_df, starting_money=1500, buy_percentage=0.5)
# while True:
#     start_time = time.time()
#
#     recent_df = calculate_indicators(recent_df)
#     recent_df = recent_df.dropna()
#     recent_df['State'] = predict_states(hmm_model, recent_df, hmm_features, scaler)
#
#     #Need this to make recent_df compatible with convert_data_to_windows method
#     recent_df['Label'] = 0
#     recent_df_windowed = convert_data_to_windows(recent_df[xg_features], window_size)
#
#     recent_df = recent_df.drop(index=recent_df.index[:window_size])
#     recent_df = recent_df.reset_index(drop=True)
#     recent_df_windowed = recent_df_windowed.reset_index(drop=True)
#     recent_df = pd.concat([recent_df_windowed, recent_df[agg_features]], axis=1)
#     print(recent_df)
#     recent_df.drop(['Label'], axis=1, inplace=True)
#
#     labels_recent_pred = xgboost_model.predict(recent_df)
#     labels_recent_pred = labels_recent_pred.tolist()
#     recent_df['Label'] = labels_recent_pred
#     recent_df['PriceDiff'] = recent_df['Close_t-1'].diff()
#
#     trading_df = recent_df[['Close_t-1', 'Label']]
#     print(trading_df)
#
#     time.sleep(180)
#     recent_df = get_historical_data(54)
