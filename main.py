import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
from keras.src.layers import BatchNormalization
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier
import talib
from datetime import datetime
#from xgboost import XGBClassifier
from skopt import BayesSearchCV, gp_minimize
from skopt.space import Real, Integer, Categorical
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import f1_score, accuracy_score, make_scorer, roc_auc_score
#from xgboost import plot_importance
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# from sklearn_genetic import GAFeatureSelectionCV
# from genetic_selection import GeneticSelectionCV
import requests
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
import time
import datetime
import tkinter as tk
from tkinter import messagebox
import warnings
import joblib
import shap
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
from collections import Counter
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import sys
from scipy.special import logsumexp
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, Input, Dropout, GaussianNoise, Lambda, Concatenate
from keras.optimizers import Adam
from keras.metrics import AUC
from keras.initializers import GlorotUniform, Orthogonal
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from keras.saving import register_keras_serializable
from keras import backend
#import keras_tuner as kt
#from arch import arch_model
import tensorflow as tf
import random
import os
import logging
from logging.handlers import RotatingFileHandler
import math
from decimal import Decimal, ROUND_DOWN
from trendline import walkforward_model, trendline_breakout_dataset
from api import (get_historical_data, get_recent_data, safe_get_balances, buy_btc_market,
                 sell_btc_market, buy_btc_limit, sell_btc_limit, get_all_open_orders,
                 get_exchange_info, cancel_open_orders, get_symbol_filters)

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISM"] = "1"
os.environ["PYTHONHASHSEED"] = "42"
backend.clear_session()
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
#pd.set_option('display.float_format', '{:.0f}'.format)

logger = logging.getLogger("continuous_trading")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler("trading.log", maxBytes=20_000_000, backupCount=3)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(handler)

def load_and_preprocess_data(filepath, xg_features=None, file=True, data=None, eth_data=None):
    if file:
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath, names=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df.drop(0, inplace=True)
        #Drop useless row
        df = df.astype(float)

        #eth pre-processing
        eth_df = pd.read_csv('ETHUSDC_15m.csv', names=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        eth_df.drop(['Open', 'Volume'], axis=1, inplace=True)
        eth_df.drop(0, inplace=True)
        eth_df = eth_df.astype(float)
    else:
        df = data.astype(float)
        eth_data.drop(['Open', 'Volume'], axis=1, inplace=True)
        eth_data.drop(0, inplace=True)
        eth_df = eth_data.astype(float)

    df['Label'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    days = []
    hours = []
    sessions = []
    for timestamp in df['Timestamp']:
        utc_time = datetime.datetime.utcfromtimestamp(int(timestamp) / 1000)
        days.append(utc_time.weekday())
        hour = utc_time.hour
        hours.append(hour)
        if 0 <= hour < 9:
            session = 0
        elif 7 <= hour < 16:
            session = 1
        elif 12 <= hour < 21:
            session = 2
        else:
            session = 0
        sessions.append(session)

    df['Day'] = days
    df['Session'] = sessions
    df['Hour'] = hours
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24.0)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24.0)

    #Intramarket Difference
    period = 24
    btc_cmma = cmma(df['High'], df['Low'], df['Close'], period)
    eth_cmma = cmma(eth_df['High'], eth_df['Low'], eth_df['Close'], period)
    df['BTC-ETH_Diff'] = btc_cmma - eth_cmma

    df['Price_Diff'] = df['Close'].diff()
    df["Returns"] = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(window=24).std()
    df['Price_Diff_Future'] = df['Price_Diff'].shift(-1)
    df['Price_Diff_Future_Abs'] = df['Price_Diff_Future'].abs()

    #prob(buy) - prob(sell) = prob(total)
    #if prob(total) is positive that means the buy signal is stronger, if negative that means the sell signal is stronger
    #prob(total) * price_diff

    #Replaces 0's with a number close to 0 to avoid infinity being present in Volume_Change
    #Since the label column has 0's present, we need to make sure that they are not replaced
    df['Volume'] = df['Volume'].replace(0, 0.00000000000000001)
    df["Volume_Change"] = df["Volume"].pct_change()

    # Need this because there was a weird error were the data in these columns were not classified as floats, this caused a problem with the pipeline as I'm not using a target encoder
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df['Returns'] = pd.to_numeric(df['Returns'], errors='coerce')

    df['OBV'] = talib.OBV(df['Close'], df['Volume'])

    df['MFV'] = (((df['Close'] - df['Low']) - (df['High'] - df['Close']) / (df['High'] - df['Low'])) * df['Volume'])
    df['MFV'] = df['MFV'].fillna(0) #need this incase nans are generated by dividing by 0
    df['A/D'] = 0
    for i in range(2, len(df)):
        currentMFV = df.loc[i, 'MFV']
        df.loc[i, 'A/D'] = df.loc[i-1, 'A/D'] + currentMFV
    df['CMF'] = df['MFV'].rolling(window=21).sum() / df['Volume'].rolling(window=21).sum()

    df['CumulativeVolume'] = 0
    for i in range(2, len(df)):
        new_volume = df.loc[i-1, 'CumulativeVolume'] + df.loc[i, 'Volume']
        df.loc[i, 'CumulativeVolume'] = new_volume
        if i % 97 == 0: #reset the cumulative volume everyday
            df.loc[i, 'CumulativeVolume'] = df.loc[i, 'Volume']

    df['VWAP'] = (((df['High'] + df['Low'] + df['Close']) / 3) * df['Volume']) / df['CumulativeVolume']

    df['LOW_EMA'] = talib.EMA(df['Close'], timeperiod=9)
    df['HIGH_EMA'] = talib.EMA(df['Close'], timeperiod=21)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['Aroon'] = talib.AROONOSC(df['High'], df['Low'], timeperiod=14)
    df['Fast_%K'], df['Fast_%D'] = talib.STOCHF(df['High'], df['Low'], df['Close'], fastk_period=14)
    df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'])
    df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'])
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'])
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'])

    df['Momentum'] = df['Close'].diff()
    df['Absolute_Momentum'] = df['Momentum'].abs()
    df['Short_EMA_Momentum'] = talib.EMA(df['Momentum'], timeperiod=13)
    df['Short_EMA_Absolute'] = talib.EMA(df['Absolute_Momentum'], timeperiod=13)
    df['Double_EMA_Momentum'] = talib.EMA(df['Short_EMA_Momentum'], timeperiod=25)
    df['Double_EMA_Absolute'] = talib.EMA(df['Short_EMA_Absolute'], timeperiod=25)
    df['TSI'] = 100 * (df['Double_EMA_Momentum'] / df['Double_EMA_Absolute'])

    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(df['Close'], 20)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']

    macd, macdsignal, macdhist = talib.MACDFIX(df['Close'])
    df['MACD'] = macd
    df['MACDSignal'] = macdsignal
    df['MACDHist'] = macdhist

    df['PP'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
    df['R1'] = 2 * df['PP'] - df['Low'].shift(1)
    df['R2'] = df['PP'] + (df['High'].shift(1) - df['Low'].shift(1))
    df['S1'] = 2 * df['PP'] - df['High'].shift(1)
    df['S2'] = df['PP'] - (df['High'].shift(1) - df['Low'].shift(1))

    #Ichimoku Cloud
    df['Tenkan_Sen'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
    df['Kijun_Sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
    df['Senkou_Span_A'] = (df['Tenkan_Sen'] + df['Kijun_Sen']) / 2
    df['Senkou_Span_B'] = (df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2
    df['Chikou_Span'] = df['Close'].shift(26)

    #Keltner Channels
    df['KC_Middle'] = talib.EMA(df['Close'], timeperiod=20)
    df['KC_Upper'] = df['KC_Middle'] + df['ATR'] * 2
    df['KC_Lower'] = df['KC_Middle'] - df['ATR'] * 2
    df['KC_Width'] = df['KC_Upper'] - df['KC_Lower']

    #max/min points
    close = df['Close'].values
    length = len(close)
    slope = (talib.EMA(close, timeperiod=50) - talib.EMA(close, timeperiod=10)).tolist()
    extrema_unconfirmed = [0] * length

    for i in range(1, length - 1):
        if slope[i] < 0 and slope[i - 1] >= 0:
            extrema_unconfirmed[i] = -1
        if slope[i] > 0 and slope[i - 1] <= 0:
            extrema_unconfirmed[i] = 1
    df['Extrema_Unconfirmed'] = extrema_unconfirmed

    #Candlestick patterns
    df['Body_Size'] = abs(df['Close'] - df['Open'])
    df['Upper_Wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Lower_Wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['Range'] = df['High'] - df['Low']

    #Add small number to avoid division by zero
    df['Body_Ratio'] = df['Body_Size'] / (df['Range'] + 1e-9)
    df['Upper_Wick_Ratio'] = df['Upper_Wick'] / (df['Range'] + 1e-9)
    df['Lower_Wick_Ratio'] = df['Lower_Wick'] / (df['Range'] + 1e-9)

    df['Two_Crows'] = talib.CDL2CROWS(df['Open'], df['High'], df['Low'], df['Close'])
    df['Three_Crows'] = talib.CDL3BLACKCROWS(df['Open'], df['High'], df['Low'], df['Close'])
    df['Three_Inside'] = talib.CDL3INSIDE(df['Open'], df['High'], df['Low'], df['Close'])
    df['Three_Strike'] = talib.CDL3LINESTRIKE(df['Open'], df['High'], df['Low'], df['Close'])
    df['Three_Outside'] = talib.CDL3OUTSIDE(df['Open'], df['High'], df['Low'], df['Close'])
    df['Three_Stars_South'] = talib.CDL3STARSINSOUTH(df['Open'], df['High'], df['Low'], df['Close'])
    df['Three_Soldiers'] = talib.CDL3WHITESOLDIERS(df['Open'], df['High'], df['Low'], df['Close'])
    df['Abandoned_Baby'] = talib.CDLABANDONEDBABY(df['Open'], df['High'], df['Low'], df['Close'])
    df['Advance_Block'] = talib.CDLADVANCEBLOCK(df['Open'], df['High'], df['Low'], df['Close'])
    df['Belt_Hold'] = talib.CDLBELTHOLD(df['Open'], df['High'], df['Low'], df['Close'])
    df['Breakaway'] = talib.CDLBREAKAWAY(df['Open'], df['High'], df['Low'], df['Close'])
    df['Closing_Marubozu'] = talib.CDLCLOSINGMARUBOZU(df['Open'], df['High'], df['Low'], df['Close'])
    df['Baby_Swallow'] = talib.CDLCONCEALBABYSWALL(df['Open'], df['High'], df['Low'], df['Close'])
    df['Counterattack'] = talib.CDLCOUNTERATTACK(df['Open'], df['High'], df['Low'], df['Close'])
    df['Dark_Cloud'] = talib.CDLDARKCLOUDCOVER(df['Open'], df['High'], df['Low'], df['Close'])
    df['Doji'] = talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])
    df['Doji_Star'] = talib.CDLDOJISTAR(df['Open'], df['High'], df['Low'], df['Close'])
    df['Dragonfly_Doji'] = talib.CDLDRAGONFLYDOJI(df['Open'], df['High'], df['Low'], df['Close'])
    df['Engulfing'] = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])
    df['Evening_Doji'] = talib.CDLEVENINGDOJISTAR(df['Open'], df['High'], df['Low'], df['Close'])
    df['Evening_Star'] = talib.CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
    df['Up_Down_Gap'] = talib.CDLGAPSIDESIDEWHITE(df['Open'], df['High'], df['Low'], df['Close'])
    df['Gravestone_Doji'] = talib.CDLGRAVESTONEDOJI(df['Open'], df['High'], df['Low'], df['Close'])
    df['Hammer'] = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
    df['Hanging_Man'] = talib.CDLHANGINGMAN(df['Open'], df['High'], df['Low'], df['Close'])
    df['Harami'] = talib.CDLHARAMI(df['Open'], df['High'], df['Low'], df['Close'])
    df['Harami_Cross'] = talib.CDLHARAMICROSS(df['Open'], df['High'], df['Low'], df['Close'])
    df['High_Wave'] = talib.CDLHIGHWAVE(df['Open'], df['High'], df['Low'], df['Close'])
    df['Hikkake'] = talib.CDLHIKKAKE(df['Open'], df['High'], df['Low'], df['Close'])
    df['Modified_Hikkake'] = talib.CDLHIKKAKEMOD(df['Open'], df['High'], df['Low'], df['Close'])
    df['Homing_Pigeon'] = talib.CDLHOMINGPIGEON(df['Open'], df['High'], df['Low'], df['Close'])
    df['Identical_Crows'] = talib.CDLIDENTICAL3CROWS(df['Open'], df['High'], df['Low'], df['Close'])
    df['In_Neck'] = talib.CDLINNECK(df['Open'], df['High'], df['Low'], df['Close'])
    df['Inverted_Hammer'] = talib.CDLINVERTEDHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
    df['Kicking'] = talib.CDLKICKING(df['Open'], df['High'], df['Low'], df['Close'])
    df['Kicking_Length'] = talib.CDLKICKINGBYLENGTH(df['Open'], df['High'], df['Low'], df['Close'])
    df['Ladder_Bottom'] = talib.CDLLADDERBOTTOM(df['Open'], df['High'], df['Low'], df['Close'])
    df['Long_Doji'] = talib.CDLLONGLEGGEDDOJI(df['Open'], df['High'], df['Low'], df['Close'])
    df['Long_Candle'] = talib.CDLLONGLINE(df['Open'], df['High'], df['Low'], df['Close'])
    df['Marubozu'] = talib.CDLMARUBOZU(df['Open'], df['High'], df['Low'], df['Close'])
    df['Matching_Low'] = talib.CDLMATCHINGLOW(df['Open'], df['High'], df['Low'], df['Close'])
    df['Mat_Hold'] = talib.CDLMATHOLD(df['Open'], df['High'], df['Low'], df['Close'])
    df['Morning_Doji'] = talib.CDLMORNINGDOJISTAR(df['Open'], df['High'], df['Low'], df['Close'])
    df['Morning_Star'] = talib.CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
    df['On_Neck'] = talib.CDLONNECK(df['Open'], df['High'], df['Low'], df['Close'])
    df['Piercing'] = talib.CDLPIERCING(df['Open'], df['High'], df['Low'], df['Close'])
    df['Rickshaw_Man'] = talib.CDLRICKSHAWMAN(df['Open'], df['High'], df['Low'], df['Close'])
    df['Rising_Falling'] = talib.CDLRISEFALL3METHODS(df['Open'], df['High'], df['Low'], df['Close'])
    df['Separating_Lines'] = talib.CDLSEPARATINGLINES(df['Open'], df['High'], df['Low'], df['Close'])
    df['Shooting_Star'] = talib.CDLSHOOTINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
    df['Short_Candle'] = talib.CDLSHORTLINE(df['Open'], df['High'], df['Low'], df['Close'])
    df['Spinning_Top'] = talib.CDLSPINNINGTOP(df['Open'], df['High'], df['Low'], df['Close'])
    df['Stalled'] = talib.CDLSTALLEDPATTERN(df['Open'], df['High'], df['Low'], df['Close'])
    df['Stick_Sandwich'] = talib.CDLSTICKSANDWICH(df['Open'], df['High'], df['Low'], df['Close'])
    df['Takuri'] = talib.CDLTAKURI(df['Open'], df['High'], df['Low'], df['Close'])
    df['Tasuki_Gap'] = talib.CDLTASUKIGAP(df['Open'], df['High'], df['Low'], df['Close'])
    df['Thrusting'] = talib.CDLTHRUSTING(df['Open'], df['High'], df['Low'], df['Close'])
    df['Tristar'] = talib.CDLTRISTAR(df['Open'], df['High'], df['Low'], df['Close'])
    df['Unique_River'] = talib.CDLUNIQUE3RIVER(df['Open'], df['High'], df['Low'], df['Close'])
    df['Upside_Gap_Crows'] = talib.CDLUPSIDEGAP2CROWS(df['Open'], df['High'], df['Low'], df['Close'])
    df['Three_Crows'] = talib.CDL3BLACKCROWS(df['Open'], df['High'], df['Low'], df['Close'])
    df['Up_Down_3Gap'] = talib.CDLXSIDEGAP3METHODS(df['Open'], df['High'], df['Low'], df['Close'])

    #Fractal Dimension
    # df['Hurst'] = df['Close'].rolling(window=100).apply(lambda x: compute_Hc(x, kind='price')[0], raw=False)
    # df['Fractal_Dimension'] = 2 - df['Hurst']

    #calculate interaction features
    df['Volume-ATR'] = df['Volume'] / df['ATR']
    df['VWAP-ATR'] = (df['Close'] - df['VWAP']) / df['ATR']
    df['RSI-MACD'] = df['RSI'] * df['MACD']
    df['Stochastic-RSI'] = df['Fast_%K'] / df['RSI']
    df['BB-KC'] = df['BB_Width'] / df['KC_Width']
    df['Ichimoku_Overlap'] = (df['Close'] - df['Kijun_Sen']) / (df['Senkou_Span_B'] - df['Senkou_Span_A'])
    df['LOW-HIGH_EMA'] = df['LOW_EMA'] / df['HIGH_EMA']

    #Candlestick interactions
    df['Closing_Marubozu-OBV'] = df['Closing_Marubozu'] * df['OBV']
    df['Closing_Marubozu-ATR'] = df['Closing_Marubozu'] * df['ATR']
    df['Closing_Marubozu-RSI'] = df['Closing_Marubozu'] * df['RSI']
    df['Marubozu-OBV'] = df['Marubozu'] * df['OBV']
    df['Marubozu-ATR'] = df['Marubozu'] * df['ATR']
    df['Marubozu-RSI'] = df['Marubozu'] * df['RSI']
    df['Short_Candle-OBV'] = df['Short_Candle'] * df['OBV']
    df['Short_Candle-ATR'] = df['Short_Candle'] * df['ATR']
    df['Short_Candle-RSI'] = df['Short_Candle'] * df['RSI']
    df['Long_Candle-OBV'] = df['Long_Candle'] * df['OBV']
    df['Long_Candle-ATR'] = df['Long_Candle'] * df['ATR']
    df['Long_Candle-RSI'] = df['Long_Candle'] * df['RSI']
    df['Doji-OBV'] = df['Doji'] * df['OBV']
    df['Doji-ATR'] = df['Doji'] * df['ATR']
    df['Doji-RSI'] = df['Doji'] * df['RSI']
    df['Long_Doji-OBV'] = df['Long_Doji'] * df['OBV']
    df['Long_Doji-ATR'] = df['Long_Doji'] * df['ATR']
    df['Long_Doji-RSI'] = df['Long_Doji'] * df['RSI']
    df['Dragonfly_Doji-OBV'] = df['Dragonfly_Doji'] * df['OBV']
    df['Dragonfly_Doji-ATR'] = df['Dragonfly_Doji'] * df['ATR']
    df['Dragonfly_Doji-RSI'] = df['Dragonfly_Doji'] * df['RSI']
    df['Belt_Hold-OBV'] = df['Belt_Hold'] * df['OBV']
    df['Belt_Hold-ATR'] = df['Belt_Hold'] * df['ATR']
    df['Belt_Hold-RSI'] = df['Belt_Hold'] * df['RSI']

    #Donchain Breakout
    df['Upper'] = df['Close'].rolling(288 - 1).max().shift(1)
    df['Lower'] = df['Close'].rolling(288 - 1).min().shift(1)
    df['Don_Signal'] = np.nan
    df.loc[df['Close'] > df['Upper'], 'Don_Signal'] = 1
    df.loc[df['Close'] < df['Lower'], 'Don_Signal'] = -1
    df['Don_Signal'] = df['Don_Signal'].ffill()
    df.drop(['Upper', 'Lower'], axis=1, inplace=True)

    #trendline breakout
    trades, data_x, data_y = trendline_breakout_dataset(df[['Open', 'High', 'Low', 'Close', 'Volume']], 72)
    signal, prob = walkforward_model(
        np.log(df['Close']).to_numpy(),
        trades, data_x, data_y,
        365 * 24 * 2, 365 * 24
    )
    df['Sig'] = signal
    df['Proba_Sig'] = prob

    # Generate Scree Plot
    # explained_variance_ratio = pca.explained_variance_ratio_
    # components = range(1, len(explained_variance_ratio) + 1)
    #
    # plt.figure(figsize=(8, 5))
    # plt.bar(components, explained_variance_ratio, color='blue', alpha=0.7, label='Explained Variance Ratio')
    # plt.plot(components, explained_variance_ratio, 'ro-', label='Cumulative Explained Variance')
    # plt.xlabel('Principal Components')
    # plt.ylabel('Explained Variance Ratio')
    # plt.title('Scree Plot for RSI PCA Analysis')
    # plt.legend()
    # plt.show()

    #drop temporary columns
    df.drop(['CumulativeVolume', 'Absolute_Momentum', 'Short_EMA_Momentum', 'Short_EMA_Absolute', 'Double_EMA_Momentum', 'Double_EMA_Absolute'], axis=1, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    agg_excluded = ['Timestamp', 'Label', 'Price_Diff_Future_Abs', 'Price_Diff_Future']
    agg_period = [4, 16, 48, 96, 384, 1152] #1h, 4h, 12h, 1d, 4d, 12d
    agg_df = pd.DataFrame()
    for period in agg_period:
        for feature in df.columns:
            if feature not in agg_excluded:
                agg_df[f'{feature}_Mean_{period}'] = df[feature].rolling(window=period).mean()
                agg_df[f'{feature}_Dev_{period}'] = df[feature].rolling(window=period).std()
                agg_df[f'{feature}_Drawdown_{period}'] = (df[feature] / df[feature].rolling(window=period).max() - 1).rolling(window=period).min()

    df['Label_Mean_4'] = df['Label'].shift(1).rolling(window=4).mean()
    df['Label_Mean_16'] = df['Label'].shift(1).rolling(window=16).mean()
    df['Label_Mean_48'] = df['Label'].shift(1).rolling(window=48).mean()
    df['Label_Mean_96'] = df['Label'].shift(1).rolling(window=96).mean()
    df['Label_Mean_384'] = df['Label'].shift(1).rolling(window=384).mean()
    df['Label_Mean_1152'] = df['Label'].shift(1).rolling(window=1152).mean()
    df['Label_Dev_4'] = df['Label'].shift(1).rolling(window=4).std()
    df['Label_Dev_16'] = df['Label'].shift(1).rolling(window=16).std()
    df['Label_Dev_48'] = df['Label'].shift(1).rolling(window=48).std()
    df['Label_Dev_96'] = df['Label'].shift(1).rolling(window=96).std()
    df['Label_Dev_384'] = df['Label'].shift(1).rolling(window=384).std()
    df['Label_Dev_1152'] = df['Label'].shift(1).rolling(window=1152).std()
    df['Label_Drawdown_4'] = (df['Label'].shift(1) / df['Label'].shift(1).rolling(window=4).max() - 1).rolling(window=4).min()
    df['Label_Drawdown_16'] = (df['Label'].shift(1) / df['Label'].shift(1).rolling(window=16).max() - 1).rolling(window=16).min()
    df['Label_Drawdown_48'] = (df['Label'].shift(1) / df['Label'].shift(1).rolling(window=48).max() - 1).rolling(window=48).min()
    df['Label_Drawdown_96'] = (df['Label'].shift(1) / df['Label'].shift(1).rolling(window=96).max() - 1).rolling(window=96).min()
    df['Label_Drawdown_384'] = (df['Label'].shift(1) / df['Label'].shift(1).rolling(window=384).max() - 1).rolling(window=384).min()
    df['Label_Drawdown_1152'] = (df['Label'].shift(1) / df['Label'].shift(1).rolling(window=1152).max() - 1).rolling(window=1152).min()

    df = pd.concat([df, agg_df], axis=1)
    df = df.fillna(0)

    #defragmenting dataframe
    df = df.copy()

    if file:
        training_df = df[df['Timestamp'] > 1654056000000] #January 1st 2023 12:15 am to present
        pca_training_df = df[((df['Timestamp'] >= 1641013200000) & (df['Timestamp'] <= 1654056000000))] #Jan 1st 12am 2022 to June 1st 12am 2022

        training_df.reset_index(drop=True, inplace=True)
        pca_training_df.reset_index(drop=True, inplace=True)

        # RSI PCA analysis
        rsis = pd.DataFrame()
        for p in range(2, 33):
            rsis[p] = talib.RSI(pca_training_df['Close'], timeperiod=p)

        rsi_means = rsis.mean()
        rsis -= rsi_means
        rsis = rsis.dropna()

        rsi_pca = PCA(n_components=3, random_state=42)
        rsi_pca.fit(rsis)

        training_rsis = pd.DataFrame()
        for p in range(2, 33):
            training_rsis[p] = talib.RSI(training_df['Close'], timeperiod=p)

        training_rsis -= rsi_means
        training_rsis = training_rsis.dropna()

        pca_data = rsi_pca.transform(training_rsis)
        rsi_pca_df = pd.DataFrame(pca_data, index=training_rsis.index, columns=['RSI_PC1', 'RSI_PC2', 'RSI_PC3'])
        training_df = pd.concat([training_df, rsi_pca_df], axis=1)

        # ATR PCA analysis
        atrs = pd.DataFrame()
        for p in range(2, 33):
            atrs[p] = talib.ATR(pca_training_df['High'], pca_training_df['Low'], pca_training_df['Close'], timeperiod=p)

        atr_means = atrs.mean()
        atrs -= atr_means
        atrs = atrs.dropna()

        atr_pca = PCA(n_components=3, random_state=42)
        atr_pca.fit(atrs)

        training_atrs = pd.DataFrame()
        for p in range(2, 33):
            training_atrs[p] = talib.ATR(training_df['High'], training_df['Low'], training_df['Close'], timeperiod=p)

        training_atrs -= atr_means
        training_atrs = training_atrs.dropna()

        pca_data = atr_pca.transform(training_atrs)
        atr_pca_df = pd.DataFrame(pca_data,index=training_atrs.index, columns=['ATR_PC1', 'ATR_PC2', 'ATR_PC3'])
        training_df = pd.concat([training_df, atr_pca_df], axis=1)

        # ADX PCA analysis
        adxs = pd.DataFrame()
        for p in range(2, 33):
            adxs[p] = talib.ADX(pca_training_df['High'], pca_training_df['Low'], pca_training_df['Close'], timeperiod=p)

        adx_means = adxs.mean()
        adxs -= adx_means
        adxs = adxs.dropna()

        adx_pca = PCA(n_components=5, random_state=42)
        adx_pca.fit(adxs)

        training_adxs = pd.DataFrame()
        for p in range(2, 33):
            training_adxs[p] = talib.ADX(training_df['High'], training_df['Low'], training_df['Close'], timeperiod=p)

        training_adxs -= adx_means
        training_adxs = training_adxs.dropna()

        pca_data = adx_pca.transform(training_adxs)
        adx_pca_df = pd.DataFrame(pca_data, index=training_adxs.index, columns=['ADX_PC1', 'ADX_PC2', 'ADX_PC3', 'ADX_PC4', 'ADX_PC5'])
        training_df = pd.concat([training_df, adx_pca_df], axis=1)

        #save pca models and means for future use
        joblib.dump(rsi_pca, "rsi_pca.pkl")
        joblib.dump(atr_pca, "atr_pca.pkl")
        joblib.dump(adx_pca, "adx_pca.pkl")
        joblib.dump(rsi_means, "rsi_pca_means.pkl")
        joblib.dump(atr_means, "atr_pca_means.pkl")
        joblib.dump(adx_means, "adx_pca_means.pkl")
    else:
        training_df = df
        training_df.reset_index(drop=True, inplace=True)
        rsi_pca = joblib.load("rsi_pca.pkl")
        atr_pca = joblib.load("atr_pca.pkl")
        adx_pca = joblib.load("adx_pca.pkl")
        training_rsi_means = joblib.load("rsi_pca_means.pkl")
        training_atr_means = joblib.load("atr_pca_means.pkl")
        training_adx_means = joblib.load("adx_pca_means.pkl")

        training_rsis = pd.DataFrame()
        for p in range(2, 33):
            training_rsis[p] = talib.RSI(df['Close'], timeperiod=p)

        training_rsis -= training_rsi_means
        training_rsis = training_rsis.dropna()

        pca_data = rsi_pca.transform(training_rsis)
        rsi_pca_df = pd.DataFrame(pca_data, index=training_rsis.index, columns=['RSI_PC1', 'RSI_PC2', 'RSI_PC3'])
        training_df = pd.concat([training_df, rsi_pca_df], axis=1)

        training_atrs = pd.DataFrame()
        for p in range(2, 33):
            training_atrs[p] = talib.ATR(training_df['High'], training_df['Low'], training_df['Close'], timeperiod=p)

        training_atrs -= training_atr_means
        training_atrs = training_atrs.dropna()

        pca_data = atr_pca.transform(training_atrs)
        atr_pca_df = pd.DataFrame(pca_data,index=training_atrs.index, columns=['ATR_PC1', 'ATR_PC2', 'ATR_PC3'])
        training_df = pd.concat([training_df, atr_pca_df], axis=1)

        training_adxs = pd.DataFrame()
        for p in range(2, 33):
            training_adxs[p] = talib.ADX(training_df['High'], training_df['Low'], training_df['Close'], timeperiod=p)

        training_adxs -= training_adx_means
        training_adxs = training_adxs.dropna()

        pca_data = adx_pca.transform(training_adxs)
        adx_pca_df = pd.DataFrame(pca_data, index=training_adxs.index, columns=['ADX_PC1', 'ADX_PC2', 'ADX_PC3', 'ADX_PC4', 'ADX_PC5'])
        training_df = pd.concat([training_df, adx_pca_df], axis=1)

    training_df.fillna(0, inplace=True)
    #calculate aggregate pca features
    agg_included = ['RSI_PC1', 'RSI_PC2', 'RSI_PC3', 'ATR_PC1', 'ATR_PC2', 'ATR_PC3', 'ADX_PC1', 'ADX_PC2', 'ADX_PC3', 'ADX_PC4', 'ADX_PC5']
    agg_period = [4, 16, 48, 96, 384, 1152]  # 1h, 4h, 12h, 1d, 4d, 12d
    training_agg_df = pd.DataFrame()
    for period in agg_period:
        #print()
        for feature in training_df.columns:
            if feature in agg_included:
                #print(f"'{feature}_Mean_{period}', ", end=' ')
                training_agg_df[f'{feature}_Mean_{period}'] = training_df[feature].rolling(window=period).mean()
                training_agg_df[f'{feature}_Dev_{period}'] = training_df[feature].rolling(window=period).std()
                training_agg_df[f'{feature}_Drawdown_{period}'] = (training_df[feature] / training_df[feature].rolling(window=period).max() - 1).rolling(window=period).min()

    training_df = pd.concat([training_df, training_agg_df], axis=1)
    training_df = training_df.fillna(0)

    # magnitude_weight = np.log1p(training_df['Price_Diff_Future_Abs'].values)
    # training_df['Weights'] = magnitude_weight
    ms = MinMaxScaler(feature_range=(0, 1))
    training_df['Weights'] = ms.fit_transform(training_df[['Price_Diff_Future_Abs']]) * 1

    # labels = training_df['Label'].tolist()
    # counts = Counter(labels)
    # total = len(labels)
    # class_mult = {cls: total / (2 * cnt) for cls, cnt in counts.items()}
    # class_weight = np.array([class_mult[label] for label in labels])

    training_df.drop(['Price_Diff_Future_Abs', 'Price_Diff_Future', 'Timestamp'], axis=1, inplace=True)
    training_df = training_df.copy()

    if file:
        ex_scaler = StandardScaler()
        mask = ~training_df.columns.str.contains("Mean|Dev|Drawdown", case=False, regex=True)
        no_agg = training_df.loc[:, mask]
        excluded_features = no_agg.drop(columns=xg_features)
        excluded_features_scaled = ex_scaler.fit_transform(excluded_features)

        ex_pca = PCA(n_components=3, random_state=42)
        pca_data = ex_pca.fit_transform(excluded_features_scaled)
        excluded_pca_df = pd.DataFrame(pca_data, columns=['EX_PC1', 'EX_PC2', 'EX_PC3'])
        training_df = pd.concat([training_df, excluded_pca_df], axis=1)
        joblib.dump(ex_pca, "ex_pca.pkl")
        joblib.dump(ex_scaler, "ex_scaler.pkl")
    else:
        ex_pca = joblib.load("ex_pca.pkl")
        ex_scaler = joblib.load("ex_scaler.pkl")
        mask = ~training_df.columns.str.contains("Mean|Dev|Drawdown", case=False, regex=True)
        no_agg = training_df.loc[:, mask]
        excluded_features = no_agg.drop(columns=xg_features)
        excluded_features_scaled = ex_scaler.transform(excluded_features)

        pca_data = ex_pca.transform(excluded_features_scaled)
        excluded_pca_df = pd.DataFrame(pca_data, columns=['EX_PC1', 'EX_PC2', 'EX_PC3'])
        training_df = pd.concat([training_df, excluded_pca_df], axis=1)

    if file:
        training_df.to_csv("BTCUSDC_15m_processed.csv", index=False)
    return training_df

def cmma(High, Low, Close, period, atr_period = 168):
    atr = talib.ATR(High, Low, Close, atr_period)
    ma = Close.rolling(period).mean()
    ind = (Close - ma) / (atr * period ** 0.5)
    return ind

def custom_pnl_objective(y_true, y_pred):
    y_true[y_true == 0] = -1
    p_buy = 1.0 / (1.0 + np.exp(-y_pred))
    prob_diff = 2.0 * p_buy - 1.0
    loss = -(prob_diff * y_true)

    grad = -((2 * np.exp(-y_pred) * y_true) / ((np.exp(-y_pred) + 1) ** 2))
    hess = -((2 * np.exp(-2 * y_pred) * y_true * (-np.exp(y_pred) + 1)) / (np.exp(-y_pred) + 1) ** 3)
    hess = np.maximum(hess, 1e-6)

    #print(f"y_true:{y_true}, y_pred:{y_pred}, p_buy:{p_buy}, prob_diff:{prob_diff}, grad:{grad}, hess:{hess}")
    #print(f"grad counts:{np.unique(grad, return_counts=True)}, true counts:{np.unique(y_true, return_counts=True)}, pred counts: {np.unique(y_pred, return_counts=True)}")

    return grad.astype(np.float32), hess.astype(np.float32)

def convert_data_to_windows(data, window_size=2):
    final = pd.DataFrame()
    non_windowed = ['Label', 'Weights', 'RSI_PC1', 'RSI_PC2', 'RSI_PC3', 'ATR_PC1', 'ATR_PC2', 'ATR_PC3', 'ADX_PC1', 'ADX_PC2', 'ADX_PC3', 'ADX_PC4', 'ADX_PC5']
    non_windowed = [column for column in non_windowed if column in data.columns]
    for i in range(window_size):
        for feature in data.columns:
            if feature not in non_windowed:
                final[f'{feature}_t{i}'] = data[feature].shift(i)

    for feature in non_windowed:
        final[feature] = data[feature]
    final.dropna(inplace=True)
    return final
    # window_data['Label'] = window_data['Label'].shift(1)
    # window_data['Weights'] = window_data['Weights'].shift(1)

def trading_simulation(labels, closes, starting_money=500, spend_percentage=1):
    money = starting_money
    bitcoin = 0
    close_prices = closes
    buy_order = False
    sell_order = False

    #0 is sell, 2 is buy
    count = 0
    day_count = 0
    last_bought_price = 0
    last_sold_price = 0
    profitable_trade_count = 0
    trade_count = 0
    historical_daily_starting_assets = []
    for label in labels:
        #print(f"close: {close_prices[count]}, label: {label}")
        if count % 96 == 0: #update daily starting assets at the start of each day
            daily_starting_assets = money + (bitcoin * close_prices[count])
            historical_daily_starting_assets.append(daily_starting_assets)
            # if len(historical_daily_starting_assets) >= 2:
            #     if (historical_daily_starting_assets[day_count - 1] * 0.95) > historical_daily_starting_assets[day_count]:
            #         print(f"Stop-loss triggered on Day {day_count}: more than 5% drop in a single day.")
            #         print(f"Current money: {money}, Current bitcoin: {bitcoin}, Current count: {count}")
            #         return
            day_count += 1

        if label == 1:
            sell_order = True
        elif label == 0:
            #print(f"Overconfidence on Day {count}: {overconfidence}")
            buy_order = True

        #print(hold)
        #print(buy_sell)
        if sell_order and label == 0:
            money += bitcoin * close_prices[count]
            last_sold_price = close_prices[count]
            #print(f"Day {day_count}: SOLD {bitcoin} BTC at {last_sold_price} each.")
            bitcoin = 0
            sell_order = False
            if last_sold_price > last_bought_price:
                #print("1")
                profitable_trade_count += 1
                trade_count += 1
            else:
                #print("0")
                trade_count += 1
        elif buy_order and label == 1:
            spend = money * spend_percentage
            amount_to_spend = min(spend, money)
            bitcoin_bought = amount_to_spend / close_prices[count]
            bitcoin += bitcoin_bought
            money -= amount_to_spend
            buy_order = False
            last_bought_price = close_prices[count]
            #print(f"Day {day_count}: BOUGHT {bitcoin_bought} BTC at {last_bought_price} each.")

        #print(f"Money: {money}, Bitcoin: {bitcoin}, Count: {count}")
        count += 1

    total_assets = money + bitcoin * close_prices[count-2]
    profit = total_assets - starting_money
    print("Final money: ", total_assets)
    print("Profit: ", profit)
    #print("Percentage profitable: ", profitable_trade_count / trade_count)
    return profit

def get_next_interval(interval_seconds):
    now = time.time()
    next_interval = ((now // interval_seconds) + 1) * interval_seconds
    return next_interval - now

def notify():
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Notification", "Your code has finished running!")
    root.destroy()

def continuous_sim(models, meta_features, agg_features, lstm_features, hmm_features,
                   min_prob=0.539, max_prob=0.548, overconfidence_mult=100,
                   starting_money=500, spend_percentage=1):
    # while True:
    #     now = datetime.datetime.now()
    #     if now.minute % 15 == 0 and now.second == 1:
    #         break
    #     time.sleep(0.1)

    twenty_days_ms = 60 * 24 * 60 * 60 * 1000
    current_time = int(time.time() * 1000)
    print(f"Current time: {current_time}")
    start_time = current_time - twenty_days_ms
    raw_data = get_historical_data(start_time, current_time)
    eth_data = get_historical_data(start_time, current_time, 'ETHUSDC')
    raw_data.drop(raw_data.index[-1], inplace=True)
    eth_data.drop(eth_data.index[-1], inplace=True)
    eth_df = eth_data.copy()
    recent_df = raw_data.copy()
    # 1735732800000, 1738411200000 -> Jan 2025
    # 1648728000000, 1738324800000 -> March 2023 to now
    recent_df = load_and_preprocess_data("", meta_features, False, data=recent_df, eth_data=eth_df)
    recent_df['Direction'] = (recent_df['Returns'] > 0).astype(int)

    # non_zero = [c for c in recent_df.columns if not (recent_df[c] == 0).all()]
    # imp = IterativeImputer(
    #     estimator=BayesianRidge(),
    #     missing_values=0,
    #     sample_posterior=False,
    #     max_iter=20,
    #     tol=1e-3,
    #     n_nearest_features=10,
    #     initial_strategy='median',
    #     imputation_order='ascending',
    #     random_state=42
    # )
    # recent_df[non_zero] = pd.DataFrame(
    #     imp.fit_transform(recent_df[non_zero]),
    #     index=recent_df.index,
    #     columns=non_zero
    # )

    hmm_scaler = joblib.load('hmm_scaler.pkl')
    hmm = joblib.load('trained_hmm.pkl')
    meta_hmm = hmm_scaler.transform(recent_df[hmm_features].values)
    hmm_df = hmm_walk_forward(hmm, meta_hmm)
    hmm_df = hmm_df.reset_index(drop=True)
    recent_df = recent_df.reset_index(drop=True)
    recent_df = pd.concat([recent_df, hmm_df], axis=1)

    full_features = meta_features.copy()
    full_features.extend(agg_features)
    labels, probas = predict_with_meta_model(models, recent_df[full_features], recent_df[lstm_features])
    label = labels[-1]
    proba = probas[-1]
    price = recent_df['Close'].iloc[-1]
    timestamp = raw_data['Timestamp'].iloc[-1]

    money = starting_money
    bitcoin = 0.0
    buy_order = False
    sell_order = False
    count = 0

    while True:
       # overconfidence = abs(proba - 0.511) * overconfidence_mult
        hold = (min_prob <= proba <= max_prob)

        if label == 1 and not hold:
            sell_order = True
        elif label == 0 and not hold:
            buy_order = True

        if sell_order and label == 0:
            money += bitcoin * price
            bitcoin = 0.0
            sell_order = False
        elif buy_order and label == 1:
            desired = money * spend_percentage
            spend = min(desired, money)
            btc_amt = spend / price
            bitcoin += btc_amt
            money -= spend
            buy_order = False

        count += 1
        print("timestamp: " + str(timestamp))
        print("last close: " + str(price))
        print(f"[{count}] label={label}  proba={proba:.4f}  money={money:.2f}  btc={bitcoin:.6f}")
        while True:
            now = datetime.datetime.now()
            if now.minute % 15 == 0 and now.second == 10:
                break
            time.sleep(0.1)

        row = get_recent_data('2')
        eth_row = get_recent_data('2', 'ETHUSDC')
        row = row.iloc[[0]]
        eth_row = eth_row.iloc[[0]]
        raw_data = pd.concat([raw_data, row], ignore_index=True)
        eth_data = pd.concat([eth_data, eth_row], ignore_index=True)
        recent_df = raw_data.copy()
        eth_df = eth_data.copy()
        # 1735732800000, 1738411200000 -> Jan 2025
        # 1648728000000, 1738324800000 -> March 2023 to now
        recent_df = load_and_preprocess_data("", meta_features, False, data=recent_df, eth_data=eth_df)
        recent_df['Direction'] = (recent_df['Returns'] > 0).astype(int)
        #recent_df = recent_df.fillna(0)

        # recent_df[non_zero] = pd.DataFrame(
        #     imp.transform(recent_df[non_zero]),
        #     index=recent_df.index,
        #     columns=non_zero
        # )

        meta_hmm = hmm_scaler.transform(recent_df[hmm_features].values)
        hmm_df = hmm_walk_forward(hmm, meta_hmm)
        hmm_df = hmm_df.reset_index(drop=True)
        recent_df = recent_df.reset_index(drop=True)
        recent_df = pd.concat([recent_df, hmm_df], axis=1)

        labels, probas = predict_with_meta_model(models, recent_df[full_features], recent_df[lstm_features])
        label = labels[-1]
        proba = probas[-1]
        price = recent_df['Close'].iloc[-1]
        timestamp = raw_data['Timestamp'].iloc[-1]

def calculate_meta_features(data, labels, weights, base_learners, tscv):
    labels = np.array(labels)
    weights = np.array(weights)
    n, m = data.shape[0], len(base_learners)
    meta = np.full((n, m), np.nan)
    aucs = []
    for col, (name, clf) in enumerate(base_learners):
        for train_idx, test_idx in tscv.split(data):
            if name == 'mlp':
                clf.fit(data.iloc[train_idx], labels[train_idx])
                meta[test_idx, col] = clf.predict_proba(data.iloc[test_idx])[:, 1]
            else:
                clf.fit(data.iloc[train_idx], labels[train_idx], sample_weight=weights[train_idx])
                y_pred = clf.predict_proba(data.iloc[test_idx])[:, 1]
                meta[test_idx, col] = y_pred
                aucs.append(roc_auc_score(labels[test_idx], y_pred, sample_weight=weights[test_idx]))

    # lgbm = dict(base_learners)['lgbm']
    # importances = lgbm.feature_importances_
    # feat_names = data.columns
    # feat_imp = pd.Series(importances, index=feat_names)
    # feat_imp_list = list(zip(feat_imp.index.tolist(), feat_imp.values.tolist()))
    # feat_imp_list.sort(key=lambda x: x[1], reverse=True)
    # top50_names = [feat for feat, imp in feat_imp_list[:50]]
    # print(top50_names)
    # corr_matrix = data.corr().abs()
    #
    # def prune_by_corr(importance_series, corr_mat, threshold):
    #     selected = []
    #     for feat in importance_series.index:
    #         if any(corr_mat.loc[feat, sel] > threshold for sel in selected):
    #             continue
    #         selected.append(feat)
    #     return selected
    #
    # for thresh in np.arange(0.85, 0.951, 0.01):
    #     t = round(thresh, 2)
    #     sorted_imp = feat_imp.sort_values(ascending=False)
    #     kept = prune_by_corr(sorted_imp, corr_matrix, t)
    #     print(f"Threshold={t:.2f}: kept {len(kept)} / {len(sorted_imp)} features")
    #     print("Kept features:", kept, "\n")

    print(f"LGBM train‐set AUC: {sum(aucs)/len(aucs)}")
    return pd.DataFrame(meta, index=data.index, columns=[name for name, _ in base_learners])

def calculate_lstm_features(data, weights, window, hidden_units, dropout, learning_rate, rec_drop, noise, ker, rec, tscv):
    preds = np.full(len(data), np.nan)
    features = data.columns.tolist()
    cols = [c for c in data.columns if c != "Direction"]
    for train_idx, test_idx in tscv.split(data):
        train_idx = train_idx[train_idx >= window]
        test_idx = test_idx[test_idx >= window]
        train_df = data.iloc[train_idx].reset_index(drop=True)
        test_df = data.iloc[test_idx].reset_index(drop=True)

        scaler = RobustScaler()
        scaler.fit(train_df[cols])
        train_df[cols] = scaler.transform(train_df[cols])
        test_df[cols] = scaler.transform(test_df[cols])

        X_train, y_train = prep_lstm_data(train_df, features, window, "Direction")
        X_test, _ = prep_lstm_data(test_df, features, window, "Direction")
        w_train = np.array(weights)[train_idx]
        w_train = w_train[window:]

        es = EarlyStopping(monitor="val_auc", patience=10, restore_best_weights=True, mode="max")
        rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, mode="min", min_lr=1e-6)
        model = build_stacked_lstm(timesteps=window, n_feats=len(features), lstm_units=hidden_units, dropout=dropout, lr=learning_rate, noise=noise,
                                   ker=ker, rec=rec, rec_dropout=rec_drop)
        model.fit(X_train, y_train, sample_weight=w_train, epochs=30, batch_size=64, verbose=1, shuffle=False, validation_split=0.1, callbacks=[es, rlrop])
        pred = model.predict(X_test, verbose=1).ravel()

        for i, p in enumerate(pred):
            orig_i = test_idx[i + window]
            preds[orig_i] = p
    return preds

@register_keras_serializable()
def mix_fn(x):
    a, nn_p, lr_p = x
    return a * lr_p + (1.0 - a) * nn_p

def build_meta_model(data_train, labels_train, weights_train, data_test, labels_test, closes_test, lstm_data_train, lstm_data_test, lstm_hidden_units,
                     lstm_dropout, lstm_learning_rate, lstm_window_size, lstm_rec_drop, lstm_noise, lstm_ker, lstm_rec):
    lgbm_params = {
        'boosting_type': 'gbdt',
        'colsample_bytree': 0.1,
        'learning_rate': 0.003093443570013053,
        'max_depth': 15,
        'n_estimators': 1239,
        'num_leaves': 20,
        'reg_alpha': 2.8867915528468985e-08,
        'reg_lambda': 11.404719722698687,
        'subsample': 0.5443075312616172,
        'min_child_samples': 5,
        'min_split_gain': 0.0,
        'scale_pos_weight': 1.1992235739783454,
        'subsample_freq': 10
    }

    lstm_features = lstm_data_train.columns.tolist()
    estimators = [
        ('lgbm', LGBMClassifier(**lgbm_params, n_jobs=-1, metric='auc', random_state=42)),
    ]

    tscv = TimeSeriesSplit(n_splits=3)
    meta_data = calculate_meta_features(data_train, labels_train, weights_train, estimators, tscv)
    mask = ~np.isnan(meta_data).all(axis=1) #drops beginning nan rows
    meta_data = meta_data.loc[mask].reset_index(drop=True)
    meta_labels_train = np.array(labels_train)[mask]
    meta_weights_train = np.array(weights_train)[mask]

    lstm_data = calculate_lstm_features(lstm_data_train, weights_train, lstm_window_size, lstm_hidden_units, lstm_dropout,
                                        lstm_learning_rate, lstm_rec_drop, lstm_noise, lstm_ker, lstm_rec, tscv)
    lstm_data = lstm_data[mask]
    lstm_data = np.nan_to_num(lstm_data, nan=0.5)
    meta_data['LSTM'] = lstm_data

    # corr = meta_data.corr()
    # sns.heatmap(corr, annot=True, fmt=".2f")
    # plt.title("Meta-feature Correlation")
    # plt.show()

    lr = LogisticRegression(max_iter=20000, random_state=42)
    search_space = {
        'C': Real(1e-4, 1000, prior='log-uniform'),
        'penalty': Categorical(['l1', 'l2']),
        'solver': Categorical(['liblinear',  'saga'])
    }

    bayes = BayesSearchCV(
        estimator = lr,
        search_spaces = search_space,
        cv = 5,
        n_iter = 30,
        scoring = 'balanced_accuracy',
        n_jobs = -1,
        verbose = 1,
        refit = True,
        random_state=42
    )
    bayes.fit(meta_data, meta_labels_train, sample_weight=meta_weights_train)
    print("best params:", bayes.best_params_)
    print("best CV AUC:", bayes.best_score_)
    best_stack = bayes.best_estimator_

    data_train_masked = data_train.loc[mask].reset_index(drop=True)
    lr_probas = best_stack.predict_proba(meta_data)[:, 1]
    X_moe = data_train_masked.copy()
    X_moe = pd.concat([X_moe, meta_data], axis=1)

    meta_in = Input(shape=(X_moe.shape[1],), name="meta_input")
    lr_in = Input(shape=(1,), name="lr_input")
    nn_expert = Dense(64, activation="relu")(meta_in)
    nn_expert = BatchNormalization()(nn_expert)
    nn_expert = Dense(32, activation="relu")(nn_expert)
    nn_expert = BatchNormalization()(nn_expert)
    nn_expert = Dense(16, activation="relu")(nn_expert)
    nn_expert = Dense(1, activation="sigmoid", name="nn_expert")(nn_expert)

    gate_in = Concatenate(name="gate_input")([meta_in, lr_in, nn_expert])
    gate = Dense(32, activation="relu")(gate_in)
    gate = BatchNormalization()(gate)
    gate = Dense(16, activation="relu")(gate)
    gate = Dense(1, activation="sigmoid", name="gate")(gate)
    mixture = Lambda(mix_fn, name="mixture")([gate, nn_expert, lr_in])

    es = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, mode="max")
    moe = Model(inputs=[meta_in, lr_in], outputs=mixture)
    moe.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    moe.fit(x=[X_moe, lr_probas], y=meta_labels_train, sample_weight=meta_weights_train, batch_size=256, epochs=20, validation_split=0.1, callbacks=[es])
    moe_probas = moe.predict([X_moe, lr_probas]).ravel()

    X_reliability = data_train[mask].reset_index(drop=True)
    X_reliability['probas'] = moe_probas
    y_reliab = (best_stack.predict(meta_data) == meta_labels_train).astype(int)
    reliability_model = LGBMClassifier(**lgbm_params, n_jobs=-1, metric='auc', random_state=42)
    reliability_model.fit(X_reliability, y_reliab, sample_weight=meta_weights_train)

    cols = [c for c in lstm_features if c != "Direction"] #again, don't want to scale 0s and 1s
    lstm_scaler = RobustScaler()
    lstm_scaler.fit(lstm_data_train[cols])
    scaled_lstm_train = pd.DataFrame()
    scaled_lstm_train[cols] = lstm_scaler.transform(lstm_data_train[cols])
    scaled_lstm_train['Direction'] = lstm_data_train['Direction']
    X_train, y_train = prep_lstm_data(scaled_lstm_train, lstm_features, lstm_window_size, 'Direction')
    w = np.array(weights_train[lstm_window_size:], dtype=np.float32)

    es = EarlyStopping(monitor="val_auc", patience=10, restore_best_weights=True, mode="max")
    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, mode="min", min_lr=1e-6)
    lstm = build_stacked_lstm(timesteps=lstm_window_size, n_feats=len(lstm_features), lstm_units=lstm_hidden_units, dropout=lstm_dropout, lr=lstm_learning_rate,
                            noise=lstm_noise, ker=lstm_ker, rec=lstm_rec, rec_dropout=lstm_rec_drop)
    lstm.fit(X_train, y_train, epochs=30, batch_size=64, verbose=1, sample_weight=w, shuffle=False, validation_split=0.1, callbacks=[es, rlrop])

    lstm_data_test = lstm_data_test.reset_index(drop=True)
    scaled_lstm_test = pd.DataFrame()
    scaled_lstm_test[cols] = lstm_scaler.transform(lstm_data_test[cols])
    scaled_lstm_test['Direction'] = lstm_data_test['Direction']
    X_test, y_test = prep_lstm_data(scaled_lstm_test, lstm_features, lstm_window_size, 'Direction')
    lstm_preds = lstm.predict(X_test, verbose=1).ravel()

    bg = X_train[np.random.choice(len(X_train), 100, replace=False)]
    explainer = shap.GradientExplainer(lstm, bg)
    X_explain = X_test[:200]
    shap_vals = explainer.shap_values(X_explain)
    abs_avg = np.mean(np.abs(shap_vals), axis=(0, 1, 3))
    print("abs_avg shape (should be n_features):", abs_avg.shape)
    feat_idx = np.argsort(abs_avg)[::-1]
    top15 = [lstm_features[i] for i in feat_idx[:15]]
    print("Top 15 features by SHAP attribution:", top15)

    for name, clf in estimators:
        if name == 'mlp':
            clf.fit(data_train, labels_train)
        else:
            clf.fit(data_train, labels_train, sample_weight=weights_train)
    meta_test = pd.DataFrame({name: clf.predict_proba(data_test)[:, 1] for name, clf in estimators}, index=data_test.index)
    lstm_preds = pd.Series(lstm_preds, index=data_test.index[lstm_window_size:])
    meta_test['LSTM'] = lstm_preds
    meta_test['LSTM'].fillna(0.5, inplace=True)

    X_moe_test = data_test.copy()
    X_moe_test = pd.concat([X_moe_test, meta_test], axis=1)
    lr_probas_test = best_stack.predict_proba(meta_test)[:, 1]
    moe_probas_test = moe.predict([X_moe_test, lr_probas_test]).ravel()
    reliability_data_test = data_test.copy()
    reliability_data_test['probas'] = moe_probas_test
    reliability_probs = reliability_model.predict_proba(reliability_data_test)[:, 1]

    # moe_labels = (moe_probas_test >= 0.50300).astype(int)
    # mask = (reliability_probs >= 0.57900).astype(int)
    # preds = np.where(mask, moe_labels, -1)
    # profit = trading_simulation(preds, closes_test)

    thresh_grid = np.linspace(0.4, 0.6, 201)
    best_profit = -np.inf
    best_pair = (None, None)
    for thresh_moe in thresh_grid:
        moe_labels = (moe_probas_test >= thresh_moe).astype(int)
        for thresh_mask in thresh_grid:
            mask = (reliability_probs >= thresh_mask).astype(int)
            preds = np.where(mask, moe_labels, -1)
            profit = trading_simulation(preds, closes_test)
            if profit > best_profit:
                best_profit = profit
                best_pair = (thresh_moe, thresh_mask)
                best_preds = preds

    #Best profit = 268.87617 at thresh_moe = 0.49500, thresh_mask = 0.54500
    #Best profit = 272.15604 at thresh_moe = 0.50300, thresh_mask = 0.57900 - new lgbm data
    #Best profit = 295.27515 at thresh_moe = 0.50000, thresh_mask = 0.58200 - new lstm hypers
    #Best profit = 259.92007 at thresh_moe = 0.50100, thresh_mask = 0.51700 - laptop

    print(f"Best profit = {best_profit:.5f} at " f"thresh_moe = {best_pair[0]:.5f}, " f"thresh_mask = {best_pair[1]:.5f}")
    save_meta_model("meta_model", estimators, best_stack, lstm, lstm_features, lstm_window_size, lstm_scaler, moe, reliability_model)

    return best_stack, best_preds, best_profit


def save_meta_model(path, base_learners, meta_lr, lstm_model, lstm_features, lstm_window_size, lstm_scaler, moe_model, reliability_model):
    os.makedirs(path, exist_ok=True)
    joblib.dump(base_learners, os.path.join(path, "base_learners.pkl"))
    joblib.dump(meta_lr, os.path.join(path, "meta_lr.pkl"))
    lstm_model.save(os.path.join(path, "lstm_model.keras"))
    joblib.dump(lstm_scaler, os.path.join(path, "lstm_scaler.pkl"))
    with open(os.path.join(path, "lstm_config.json"), "w") as f:
        json.dump({
            "features": lstm_features,
            "window_size": lstm_window_size
        }, f)
    moe_model.save(os.path.join(path, "moe_model.keras"))
    joblib.dump(reliability_model, os.path.join(path, "reliability_model.pkl"))

def load_meta_model(path):
    base_learners = joblib.load(os.path.join(path, "base_learners.pkl"))
    meta_lr = joblib.load(os.path.join(path, "meta_lr.pkl"))
    lstm_model = load_model(os.path.join(path, "lstm_model.keras"), compile=False)
    lstm_scaler = joblib.load(os.path.join(path, "lstm_scaler.pkl"))
    with open(os.path.join(path, "lstm_config.json")) as f:
        cfg = json.load(f)
    lstm_features = cfg["features"]
    lstm_window_size = cfg["window_size"]

    moe_model = load_model(os.path.join(path, "moe_model.keras"), compile=False, custom_objects={"mix_fn": mix_fn})
    reliability_model = joblib.load(os.path.join(path, "reliability_model.pkl"))

    return (base_learners, meta_lr, lstm_model, lstm_features, lstm_window_size, lstm_scaler, moe_model, reliability_model)

def predict_with_meta_model(models, data_test, lstm_data_test):
    (base_learners, meta_lr, lstm_model, lstm_features, lstm_window_size, lstm_scaler, moe_model, reliability_model) = models

    meta_test = pd.DataFrame(
      {name: clf.predict_proba(data_test)[:,1]
        for name, clf in base_learners},
      index=data_test.index)

    X = lstm_data_test.copy()
    cols = [c for c in lstm_features if c != "Direction"]
    X[cols] = lstm_scaler.transform(X[cols])
    # prep sliding windows exactly as during training:
    X_seq, _ = prep_lstm_data(X, lstm_features, lstm_window_size, target="Direction")
    lstm_preds = lstm_model.predict(X_seq).ravel()
    # pad front with 0.5 to match length
    pad = pd.Series(0.5, index=data_test.index[:lstm_window_size])
    lstm_series = pd.Series(lstm_preds, index=data_test.index[lstm_window_size:])
    meta_test["LSTM"] = pd.concat([pad, lstm_series]).reindex(data_test.index)

    lr_probas = meta_lr.predict_proba(meta_test)[:,1]
    X_moe_test = pd.concat([
        data_test.reset_index(drop=True),
        meta_test.reset_index(drop=True)
    ], axis=1)
    moe_probas = moe_model.predict([X_moe_test, lr_probas]).ravel()

    reliability_input = data_test.copy()
    reliability_input["probas"] = moe_probas
    mask_probas = reliability_model.predict_proba(reliability_input)[:,1]

    tm, tmask = 0.48500, 0.53300
    moe_labels = (moe_probas >= tm).astype(int)
    mask_labels = (mask_probas >= tmask).astype(int)
    preds = np.where(mask_labels, moe_labels, -1)
    probas = np.where(mask_labels, moe_probas, -1)

    return preds, probas

class HMMEstimator(BaseEstimator): #sklearn wrapper for hmm so I can run bayes search
    def __init__(self, n_components=3, covariance_type="full", n_iter=100, random_state=42, min_covar=1e-3):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.min_covar = min_covar

    def fit(self, X, y=None):
        self._failed = False
        self.model_ = GaussianHMM(n_components=self.n_components, covariance_type=self.covariance_type, n_iter=self.n_iter, random_state=self.random_state, min_covar=self.min_covar)
        try:
            self.model_.fit(X)
        except (ValueError, np.linalg.LinAlgError): #In case covariance fails
            self._failed = True
        return self

    def score(self, X, y=None):
        if self._failed == True:
            return -1e10 #return a terrible log-likelihood in case of value error
        try:
            return self.model_.score(X) / X.shape[0]
        except (ValueError, np.linalg.LinAlgError):
            return -1e10

    def predict(self, X):
        if self._failed == True:
            raise RuntimeError("HMM didn’t fit properly; cannot predict")
        return self.model_.predict(X)

    def _more_tags(self):
        return {"requires_y": False}

def hmm_log_likelihood(estimator, X, y=None): #wrapper for scoring estimator
    return estimator.score(X)

def hmm_walk_forward(model, X):
    log_emlik = model._compute_log_likelihood(X)
    log_pi = np.log(model.startprob_)
    log_A = np.log(model.transmat_)
    n, K = log_emlik.shape
    log_alpha = np.zeros((n, K))
    log_alpha[0] = log_pi + log_emlik[0]

    for t in range(1, n):
        log_alpha[t] = log_emlik[t] + logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0)

    log_norm = logsumexp(log_alpha, axis=1, keepdims=True)
    log_post = log_alpha - log_norm
    posteriors = np.exp(log_post)
    states = posteriors.argmax(axis=1)

    cols = [f"state_{k}_proba" for k in range(K)]
    df = pd.DataFrame(posteriors, columns=cols, index=range(n))
    df["state"] = states
    return df

class ARCHWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, vol='EGARCH', p=1, q=1, dist='ged'):
        self.vol = vol
        self.p = p
        self.q = q
        self.dist = dist

    def fit(self, X, y=None):
        self.model_ = arch_model(X[:, 0], vol=self.vol, p=self.p, q=self.q, dist=self.dist)
        self.res_ = self.model_.fit(disp='off')
        return self

    def predict(self, X):
        fore = self.res_.forecast(horizon=1, reindex=False)
        return fore.variance.values.flatten()

    def score(self, X, y=None):
        return self.res_.loglikelihood

def prep_lstm_data(data, features, window, target):
    sub = data[features + [target]].to_numpy(dtype=np.float32)
    n_samples, n_cols = sub.shape
    n_feats = len(features)
    n_seq = n_samples - window
    X = np.zeros((n_seq, window, n_feats), dtype=np.float32)
    y = np.zeros((n_seq,), dtype=np.float32)
    for i in range(n_seq):
        X[i] = sub[i: i + window, :n_feats]
        y[i] = sub[i + window, n_feats]
    return X, y

def build_lstm(timesteps, n_feats, hidden_units=32):
    m = Sequential([
        Input(shape=(timesteps, n_feats)),
        LSTM(hidden_units, kernel_initializer=GlorotUniform(seed=42), recurrent_initializer=Orthogonal(seed=42), kernel_regularizer=l2(1e-4), recurrent_regularizer=l2(1e-4), return_sequences=True),
        Dropout(0),
        Dense(1, activation="sigmoid", kernel_regularizer=l2(1e-4)),
    ])
    m.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy", AUC(name="auc")])
    return m

def build_stacked_lstm(timesteps, n_feats, lstm_units=[60,24,4], dropout=0.35567097637432504, noise=0.9263082484254979,
                       ker=0.06514259484764427, rec=0.09150447158682373, rec_dropout=0.2, lr=0.0002498531379402823):
    timesteps = int(timesteps)
    n_feats = int(n_feats)
    lstm_units = [int(u) for u in lstm_units]

    m = Sequential()
    m.add(GaussianNoise(noise, input_shape=(timesteps, n_feats)))
    m.add(LSTM(lstm_units[0],
               return_sequences=True,
               input_shape=(timesteps, n_feats),
               kernel_initializer=GlorotUniform(seed=42),
               recurrent_initializer=Orthogonal(seed=42),
               kernel_regularizer=l2(ker),
               recurrent_regularizer=l2(rec),
               recurrent_dropout=rec_dropout))
    m.add(Dropout(dropout))
    for units in lstm_units[1:-1]:
        m.add(LSTM(units,
                   return_sequences=True,
                   kernel_initializer=GlorotUniform(seed=42),
                   recurrent_initializer=Orthogonal(seed=42),
                   kernel_regularizer=l2(ker),
                   recurrent_regularizer=l2(rec),
                   recurrent_dropout=rec_dropout))
        m.add(Dropout(dropout))
    m.add(LSTM(lstm_units[-1],
               return_sequences=False,
               kernel_initializer=GlorotUniform(seed=42),
               recurrent_initializer=Orthogonal(seed=42),
               kernel_regularizer=l2(ker),
               recurrent_regularizer=l2(rec),
               recurrent_dropout=rec_dropout))
    m.add(Dropout(dropout))
    m.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(ker)))
    m.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=["accuracy", AUC(name="auc")])
    return m

def quantize_down(value, tick):
    d = Decimal(str(value))
    q = (d // tick) * tick
    return format(q, f"f")

def continuous_trading(models, meta_features, agg_features, lstm_features, hmm_features,
                   min_prob=0.539, max_prob=0.548, spend_percentage=1):
    # while True:
    #     now = datetime.datetime.now()
    #     if now.minute % 15 == 0 and now.second == 0:
    #         break
    #     time.sleep(0.1)
    filters = get_symbol_filters("BTCUSDC")
    price_tick = Decimal(filters['PRICE_FILTER']['tickSize'])
    qty_step = Decimal(filters['LOT_SIZE']['stepSize'])
    min_notional = Decimal(filters["MIN_NOTIONAL"]["minNotional"])

    twenty_days_ms = 15000 * 15 * 60 * 1000
    current_time = int(time.time() * 1000)
    print(f"Current time: {current_time}")
    logger.info(f"Starting continuous_trading at unix ms {current_time}")
    start_time = current_time - twenty_days_ms

    raw_data = get_historical_data(start_time, current_time)
    eth_data = get_historical_data(start_time, current_time, 'ETHUSDC')
    recent_df = raw_data.copy()
    recent_df = load_and_preprocess_data("", meta_features, False, data=recent_df, eth_data=eth_data)
    recent_df['Direction'] = (recent_df['Returns'] > 0).astype(int)

    hmm_scaler = joblib.load('hmm_scaler.pkl')
    hmm = joblib.load('trained_hmm.pkl')
    meta_hmm = hmm_scaler.transform(recent_df[hmm_features].values)
    hmm_df = hmm_walk_forward(hmm, meta_hmm)
    hmm_df = hmm_df.reset_index(drop=True)
    recent_df = recent_df.reset_index(drop=True)
    recent_df = pd.concat([recent_df, hmm_df], axis=1)

    full_features = meta_features.copy()
    full_features.extend(agg_features)
    labels, probas = predict_with_meta_model(models, recent_df[full_features], recent_df[lstm_features])
    label = labels[-1]
    proba = probas[-1]
    close = recent_df['Close'].iloc[-1]

    buy_order = False
    sell_order = False
    count = 0
    bitcoin, money = safe_get_balances()
    while True:
        print("Latest price: " + str(close))
        logger.info("Latest price: " + str(close))

        hold = (min_prob <= proba <= max_prob)

        if label == 1 and not hold:
            sell_order = True
        elif label == 0 and not hold:
            buy_order = True

        if sell_order and label == 0 and bitcoin != 0:
            sell_price = Decimal(str(close))
            raw_qty = Decimal(str(bitcoin))
            qty_str = quantize_down(raw_qty, qty_step)
            price_str = quantize_down(sell_price, price_tick)
            #^ need this incase str() returns a float in scientific notation
            text = sell_btc_limit(qty_str, price_str)
            deadline = datetime.datetime.now() + datetime.timedelta(minutes=14)
            while datetime.datetime.now() < deadline:
                time.sleep(60 - datetime.datetime.now().second)
                bitcoin, _ = safe_get_balances()
                raw_qty = Decimal(str(bitcoin))
                qty_str = quantize_down(raw_qty, qty_step)

                #if order still active
                if len(get_all_open_orders()) == 1:
                    sell_price -= Decimal(str((0.125 * recent_df['ATR'].iloc[-1])/14))
                    price_str = quantize_down(sell_price, price_tick)
                    cancel_open_orders()
                    text = sell_btc_limit(qty_str, price_str)
                else:
                    break

            bitcoin, _ = safe_get_balances()
            if bitcoin != 0:
                cancel_open_orders()
                text = sell_btc_market(qty_str)

            logger.info(f"Sell order response: {text}")
            logger.info(f"BTC qty: {qty_str}")
            sell_order = False
        elif buy_order and label == 1:
            desired = money * spend_percentage
            spend = min(desired, money)
            buy_price = close
            raw_qty = spend/buy_price
            qty_str = quantize_down(raw_qty, qty_step)
            price_str = quantize_down(close, price_tick)
            text = buy_btc_limit(qty_str, price_str)
            deadline = datetime.datetime.now() + datetime.timedelta(minutes=14)
            while datetime.datetime.now() < deadline:
                time.sleep(60 - datetime.datetime.now().second)
                _, money = safe_get_balances()
                desired = money * spend_percentage
                spend = min(desired, money)

                # if order still active
                if len(get_all_open_orders()) == 1:
                    buy_price += (0.125 * recent_df['ATR'].iloc[-1]) / 14
                    raw_qty = spend / buy_price
                    qty_str = quantize_down(raw_qty, qty_step)
                    price_str = quantize_down(close, price_tick)
                    cancel_open_orders()
                    text = buy_btc_limit(qty_str, price_str)
                else:
                    break

            if len(get_all_open_orders()) == 1:
                cancel_open_orders()
                text = "Buy order cancelled"

            logger.info(f"Buy order response: {text}")
            logger.info(f"USDC qty: {str(spend)}")
            buy_order = False

        count += 1
        bitcoin, money = safe_get_balances()
        print(f"[{count}] label={label}  proba={proba:.4f}  usdc={money:.10f}  btc={bitcoin:.10f}")
        logger.info(f"[{count}] label={label}  proba={proba:.4f}  usdc={money:.10f}  btc={bitcoin:.10f}")

        while True:
            now = datetime.datetime.now()
            if now.minute % 15 == 0 and now.second == 3:
                break
            time.sleep(0.05)

        row = get_recent_data('1')
        eth_row = get_recent_data('1', 'ETHUSDC')
        raw_data = pd.concat([raw_data, row], ignore_index=True)
        eth_data = pd.concat([eth_data, eth_row], ignore_index=True)
        recent_df = raw_data.copy()
        recent_df = load_and_preprocess_data("", meta_features, False, data=recent_df, eth_data=eth_data)
        recent_df['Direction'] = (recent_df['Returns'] > 0).astype(int)

        meta_hmm = hmm_scaler.transform(recent_df[hmm_features].values)
        hmm_df = hmm_walk_forward(hmm, meta_hmm)
        hmm_df = hmm_df.reset_index(drop=True)
        recent_df = recent_df.reset_index(drop=True)
        recent_df = pd.concat([recent_df, hmm_df], axis=1)

        labels, probas = predict_with_meta_model(models, recent_df[full_features], recent_df[lstm_features])
        label = labels[-1]
        proba = probas[-1]
        close = recent_df['Close'].iloc[-1]

hmm_scaler = StandardScaler()
hmm_features = ['RSI_PC2', 'MFV', 'OBV', 'VWAP', 'Range', 'Volume-ATR']
lstm_features = ['RSI_PC2', 'RSI_PC1', 'MFV', 'OBV', 'VWAP', 'Range', 'Volume-ATR', 'Upper_Wick_Ratio', 'WILLR', 'Lower_Wick_Ratio', 'Upper_Wick', 'Lower_Wick', 'Direction']
#meta_features = ['EX_PC1', 'EX_PC2', 'EX_PC3', 'state_1_proba', 'state_3_proba', 'state_4_proba', 'state_7_proba', 'state_8_proba', 'state_9_proba', 'state_12_proba', 'state_13_proba', 'state', 'MFV', 'Upper_Wick_Ratio', 'WILLR', 'Lower_Wick_Ratio', 'Upper_Wick', 'RSI_PC1', 'Closing_Marubozu-OBV', 'Doji-RSI', 'RSI_PC2', 'Closing_Marubozu', 'Short_Candle-RSI', 'Lower_Wick', 'Volume', 'Volume-ATR', 'OBV', 'KC_Upper', 'Range', 'VWAP', 'Long_Candle-OBV', 'Body_Size', 'CCI', 'Doji-OBV', 'Fast_%D', 'Closing_Marubozu-OBV_Dev_16', 'Volume_Dev_16', 'Volume_Mean_48', 'Closing_Marubozu-OBV_Dev_4', 'MFV_Mean_16', 'OBV_Dev_16', 'OBV_Dev_96', 'OBV_Dev_48', 'Doji-OBV_Mean_16', 'Price_Diff_Mean_4', 'Volume_Dev_96', 'Upper_Wick_Ratio_Mean_96', 'Lower_Wick_Mean_4', 'OBV_Dev_4', 'Upper_Wick_Ratio_Mean_4', 'Upper_Wick_Ratio_Mean_16', 'Doji-OBV_Dev_16', 'Label', 'Weights']
meta_features = ['Short_Candle-RSI', 'Volume-ATR', 'Lower_Wick_Ratio', 'Stochastic-RSI', 'Closing_Marubozu-OBV', 'Returns', 'Lower_Wick', 'MFV', 'RSI_PC1', 'Low', 'Body_Ratio', 'RSI_PC3', 'CCI', 'BTC-ETH_Diff', 'Fast_%D', 'ADX_PC4', 'RSI_PC2', 'Body_Size', 'Upper_Wick_Ratio', 'Upper_Wick', 'Closing_Marubozu-ATR', 'Volume', 'ADX_PC2', 'Belt_Hold-RSI', 'Long_Candle-ATR', 'Doji-RSI', 'Doji-ATR', 'Range', 'VWAP', 'ADX_PC3', 'Long_Candle-OBV', 'Short_Candle-ATR', 'ADX_PC1', 'Volume_Change', 'BB-KC', 'VWAP-ATR']
extra_features = ['Returns_Mean_4', 'ATR_Drawdown_4', 'EX_PC2', 'Belt_Hold-ATR_Mean_1152', 'VWAP_Drawdown_16', 'EX_PC3', 'RSI_PC2_Mean_4', 'Volume-ATR_Mean_96', 'Proba_Sig_Mean_48', 'Lower_Wick_Ratio_Mean_4', 'state_9_proba', 'Hour_Sin_Dev_16', 'Volume_Drawdown_4','ADX_PC1_Dev_4', 'CCI_Mean_4', 'Doji-RSI_Mean_16', 'ADX_PC5_Mean_4', 'state_8_proba', 'RSI_PC1_Dev_48', 'state_1_proba', 'state_3_proba', 'BTC-ETH_Diff_Drawdown_16', 'WILLR_Dev_48', 'Volume-ATR_Dev_4', 'state_12_proba', 'Body_Ratio_Dev_48','Short_Candle_Dev_384', 'Hour_Cos_Dev_48', 'Spinning_Top_Dev_48','Belt_Hold-RSI_Drawdown_384', 'BTC-ETH_Diff_Dev_384', 'BB-KC_Drawdown_16', 'Long_Doji-RSI_Mean_48', 'RSI_PC3_Drawdown_4', 'ADX_PC4_Drawdown_48','Aroon_Mean_4', 'Lower_Wick_Mean_4', 'Stochastic-RSI_Dev_4', 'Hikkake_Dev_96','ATR_Drawdown_16', 'MFV_Dev_4', 'Closing_Marubozu-RSI_Dev_1152', 'Closing_Marubozu-OBV_Drawdown_96', 'ATR_Drawdown_384', 'Fast_%K_Mean_16', 'KC_Upper_Dev_4', 'Volume_Change_Drawdown_96', 'Short_Candle-OBV_Mean_384', 'BTC-ETH_Diff_Mean_96', 'TSI_Drawdown_48', 'Short_Candle-RSI_Dev_16', 'Short_Candle-ATR_Drawdown_96', 'Aroon_Mean_96', 'RSI_PC3_Dev_96', 'Body_Size_Drawdown_16', 'ADX_PC2_Dev_96', 'MACDHist_Dev_4', 'Body_Ratio_Mean_4', 'Short_Candle-ATR_Mean_4','RSI_PC3_Drawdown_48', 'TSI_Dev_4', 'Proba_Sig_Dev_96', 'OBV_Dev_4', 'MFV_Dev_48', 'RSI-MACD_Mean_96', 'ATR_PC2_Dev_4','Long_Candle-ATR_Mean_48', 'RSI-MACD_Mean_48', 'Proba_Sig_Dev_384', 'Fast_%K_Drawdown_4', 'Long_Candle-OBV_Mean_4', 'Close_Dev_4', 'Harami_Dev_48', 'BB_Width_Dev_4', 'ADX_PC1_Mean_16', 'Proba_Sig_Mean_16', 'ADX_PC3_Mean_16', 'Belt_Hold_Dev_1152', 'RSI_PC3_Dev_16', 'Chikou_Span_Dev_16', 'A/D_Dev_16', 'BB_Lower_Drawdown_16', 'Long_Candle_Dev_16', 'ADX_PC3_Drawdown_96', 'ADX_PC2_Drawdown_48', 'MACDSignal_Dev_16', 'Fast_%D_Dev_4', 'ADX_PC2_Mean_1152', 'ADX_Dev_48', 'Upper_Wick_Ratio_Mean_4', 'Doji-ATR_Mean_96','A/D_Dev_96','Upper_Wick_Mean_4', 'Volume_Dev_96', 'ADX_PC3_Drawdown_4', 'Closing_Marubozu-ATR_Mean_4', 'RSI_PC2_Dev_1152', 'Hikkake_Mean_48', 'Body_Ratio_Mean_16', 'Lower_Wick_Dev_16', 'BTC-ETH_Diff_Mean_16', 'WILLR_Dev_384', 'BB_Upper_Dev_4', 'Long_Doji-RSI_Mean_1152', 'VWAP_Dev_4','High_Wave_Dev_16', 'Hour_Sin_Dev_4', 'BB-KC_Dev_96', 'Engulfing_Dev_384', 'Long_Candle-RSI_Mean_96', 'BTC-ETH_Diff_Dev_48', 'RSI-MACD_Dev_16', 'Fast_%D_Drawdown_16', 'Upper_Wick_Dev_16', 'CMF_Drawdown_48', 'Lower_Wick_Ratio_Dev_16', 'MACDHist_Mean_1152', 'MACDHist_Drawdown_16', 'R2_Drawdown_48', 'Ichimoku_Overlap_Dev_96', 'Hikkake_Mean_1152', 'Momentum_Dev_4', 'ADX_PC3_Mean_384', 'ADX_PC4_Drawdown_384', 'ADX_PC4_Dev_4', 'Open_Drawdown_4', 'Day_Mean_384', 'ADX_PC4_Dev_16', 'RSI-MACD_Drawdown_4', 'Low_Dev_4', 'OBV_Dev_16','Body_Size_Dev_48', 'CCI_Dev_96', 'CCI_Drawdown_16', 'ADX_PC4_Mean_1152', 'CMF_Drawdown_4', 'RSI_Drawdown_48', 'CMF_Dev_16', 'ADX_Dev_16', 'Momentum_Mean_96', 'ADX_PC1_Mean_1152','ATR_Dev_1152', 'LOW-HIGH_EMA_Dev_4', 'Returns_Dev_96', 'RSI_PC2_Dev_48', 'Short_Candle-RSI_Drawdown_96', 'Volume_Mean_48', 'Volume-ATR_Drawdown_96', 'Volume-ATR_Mean_16', 'MFV_Drawdown_384', 'Closing_Marubozu-ATR_Dev_48', 'TSI_Dev_16', 'RSI_PC2_Drawdown_4', 'KC_Width_Drawdown_96', 'Day_Dev_384', 'ADX_PC2_Dev_384', 'ADX_PC5_Mean_384', 'Belt_Hold-RSI_Drawdown_96', 'Dragonfly_Doji-ATR_Mean_96', 'S1_Dev_4', 'Volume-ATR_Drawdown_48', 'Dragonfly_Doji-RSI_Mean_96', 'Stochastic-RSI_Dev_16', 'Close_Dev_16', 'ADX_PC5_Dev_4', 'BB_Upper_Dev_96', 'ADX_PC5_Dev_96', 'Three_Outside_Dev_96', 'Closing_Marubozu_Dev_48', 'MACD_Dev_4', 'Short_Candle-ATR_Mean_16', 'ADX_PC2_Dev_1152', 'BB-KC_Drawdown_96', 'RSI_PC3_Mean_16', 'ADX_PC5_Drawdown_4', 'Hikkake_Dev_16', 'RSI_PC1_Dev_4', 'High_Dev_4', 'Fast_%D_Dev_96', 'High_Wave_Mean_384', 'ADX_PC2_Mean_384', 'Closing_Marubozu-RSI_Dev_96', 'TSI_Dev_48', 'ADX_Mean_96', 'MACDHist_Drawdown_48', 'ATR_PC2_Drawdown_384', 'Lower_Wick_Ratio_Dev_4', 'ADX_PC5_Mean_1152','MACDHist_Mean_96', 'RSI_PC3_Dev_384', 'ADX_PC5_Dev_384', 'Closing_Marubozu-RSI_Drawdown_384', 'RSI_PC2_Dev_384','BTC-ETH_Diff_Mean_1152', 'TSI_Drawdown_4', 'RSI_Drawdown_4', 'ATR_PC2_Dev_16', 'ADX_Drawdown_16', 'RSI_PC2_Dev_96', 'Belt_Hold-ATR_Mean_96', 'ADX_PC3_Dev_96','Price_Diff_Mean_384', 'RSI_PC3_Mean_48', 'ADX_PC1_Drawdown_16','BB-KC_Dev_4', 'ADX_PC2_Dev_4', 'Aroon_Dev_16', 'RSI_PC1_Dev_1152', 'KC_Width_Dev_4', 'BTC-ETH_Diff_Dev_16', 'ADX_PC5_Dev_1152', 'BB-KC_Dev_48', 'BB_Lower_Drawdown_48', 'Harami_Dev_96', 'Senkou_Span_A_Dev_16', 'ADX_PC3_Dev_4', 'Aroon_Mean_16', 'ATR_PC3_Drawdown_384', 'ADX_PC5_Mean_96', 'RSI_PC3_Drawdown_96', 'Belt_Hold-ATR_Drawdown_96', 'Range_Mean_4', 'ATR_PC1_Drawdown_4', 'ADX_Drawdown_384', 'Day_Mean_1152', 'Spinning_Top_Dev_16', 'Belt_Hold-RSI_Dev_48', 'Body_Size_Drawdown_48', 'Price_Diff_Drawdown_48','BTC-ETH_Diff_Drawdown_4', 'ATR_PC2_Mean_4', 'Closing_Marubozu-OBV_Mean_4', 'Upper_Wick_Ratio_Mean_16', 'ADX_PC2_Dev_16', 'Volume_Change_Mean_4', 'ADX_PC3_Mean_96', 'Fast_%D_Drawdown_4', 'RSI_PC3_Mean_4', 'ADX_PC3_Mean_4', 'ADX_PC1_Dev_16', 'ADX_PC4_Mean_4', 'ADX_PC4_Dev_48', 'Aroon_Dev_48', 'Short_Candle-ATR_Mean_48', 'Closing_Marubozu_Dev_96', 'ADX_PC3_Mean_48', 'BB_Width_Drawdown_96', 'RSI_PC3_Dev_1152', 'Belt_Hold-ATR_Mean_48', 'Fast_%D_Dev_16', 'ADX_PC3_Dev_16', 'ADX_PC1_Dev_96', 'RSI_PC1_Dev_96', 'Hikkake_Dev_384', 'CCI_Drawdown_4','ADX_PC5_Drawdown_96', 'Hikkake_Dev_4', 'Belt_Hold-RSI_Mean_16','Day_Dev_96', 'Belt_Hold-ATR_Mean_16', 'LOW-HIGH_EMA_Drawdown_16', 'VWAP-ATR_Drawdown_96', 'BTC-ETH_Diff_Drawdown_96', 'Volume_Change_Mean_384', 'ADX_PC2_Drawdown_96', 'Sig_Mean_1152', 'ADX_PC3_Dev_384', 'Volatility_Drawdown_96', 'Engulfing_Dev_48', 'Stochastic-RSI_Mean_96', 'KC_Upper_Dev_48', 'BTC-ETH_Diff_Mean_48', 'Closing_Marubozu-ATR_Drawdown_96', 'BTC-ETH_Diff_Mean_384', 'CMF_Drawdown_96', 'Upper_Wick_Ratio_Dev_4', 'ADX_PC4_Drawdown_16', 'Lower_Wick_Ratio_Mean_16', 'VWAP-ATR_Drawdown_48', 'Long_Candle-OBV_Mean_16','CCI_Dev_4', 'BB_Lower_Dev_4', 'ADX_PC3_Dev_48', 'ATR_PC2_Mean_48', 'Senkou_Span_B_Dev_16', 'Closing_Marubozu-RSI_Dev_384', 'TSI_Drawdown_96', 'Short_Candle-OBV_Dev_48', 'Short_Candle-ATR_Mean_96', 'Short_Candle-RSI_Dev_96', 'Range_Drawdown_16', 'Body_Ratio_Drawdown_48', 'Harami_Mean_384', 'CCI_Drawdown_48', 'Proba_Sig_Mean_96', 'Short_Candle-ATR_Mean_384', 'Upper_Wick_Ratio_Dev_16', 'ATR_PC2_Drawdown_4', 'MACDHist_Mean_16', 'ADX_PC4_Drawdown_96', 'Hour_Sin_Mean_4', 'Low_Drawdown_4', 'VWAP_Drawdown_96', 'TSI_Dev_96', 'Hikkake_Dev_48', 'Long_Candle-OBV_Drawdown_96', 'CMF_Drawdown_16', 'Short_Candle-RSI_Dev_4', 'Marubozu-RSI_Dev_384', 'TSI_Mean_4', 'Aroon_Dev_1152', 'RSI_PC1_Drawdown_4', 'Range_Drawdown_48', 'ATR_PC3_Drawdown_4', 'MACDSignal_Drawdown_96', 'BB-KC_Mean_48', 'RSI_Drawdown_16', 'Long_Doji-OBV_Mean_48', 'Volatility_Dev_96', 'ADX_PC4_Dev_96', 'RSI_PC2_Mean_48', 'KC_Lower_Dev_4', 'Lower_Wick_Ratio_Mean_96', 'ADX_PC4_Mean_384','Volume_Change_Mean_96', 'Long_Candle-RSI_Dev_96', 'Momentum_Drawdown_96', 'Chikou_Span_Drawdown_48', 'ADX_PC2_Mean_48', 'Harami_Dev_16', 'Ichimoku_Overlap_Drawdown_48', 'Ichimoku_Overlap_Mean_1152', 'Body_Ratio_Dev_16','ATR_Dev_16','RSI_PC2_Dev_4', 'Body_Ratio_Mean_48', 'ADX_PC4_Drawdown_4', 'RSI-MACD_Drawdown_96', 'MFV_Drawdown_48', 'Stochastic-RSI_Dev_96', 'Dragonfly_Doji-ATR_Mean_384', 'Body_Ratio_Dev_4', 'Returns_Drawdown_16', 'A/D_Drawdown_16', 'OBV_Dev_96', 'CMF_Mean_48', 'Harami_Cross_Dev_1152', 'BB_Width_Drawdown_4', 'RSI-MACD_Dev_4', 'Proba_Sig_Mean_4', 'Returns_Mean_1152', 'Three_Outside_Dev_1152', 'BB-KC_Drawdown_4', 'BB_Upper_Dev_16', 'Long_Candle_Dev_96', 'BB-KC_Mean_16', 'Short_Candle-RSI_Mean_16', 'KC_Upper_Dev_16', 'MACDSignal_Mean_4', 'Belt_Hold-RSI_Dev_16', 'Harami_Cross_Dev_96', 'CCI_Dev_384', 'ATR_PC2_Drawdown_16','LOW-HIGH_EMA_Mean_96', 'RSI_PC3_Drawdown_16', 'Hour_Cos_Mean_4', 'Doji-RSI_Mean_96', 'Volume_Change_Drawdown_384', 'Three_Outside_Dev_384']

#{'filterType': 'LOT_SIZE', 'minQty': '0.00001000', 'maxQty': '9000.00000000', 'stepSize': '0.00001000'}
models = load_meta_model('meta_model')
continuous_trading(models, meta_features, extra_features, lstm_features, hmm_features)
#continuous_sim(models, meta_features, extra_features, lstm_features, hmm_features)
#

# data = load_and_preprocess_data("BTCUSDC_15m.csv", meta_features)
# meta_features.extend(extra_features)
# data = pd.read_csv("BTCUSDC_15m_processed.csv")
# labels = data.pop('Label').tolist()
# labels = [int(x) for x in labels]
# weights = data.pop('Weights').tolist()
#
# data['Direction'] = (data['Returns'] > 0).astype(int)
# closes = data['Close'].tolist()
# n = len(data)
# cut = int(n * 0.2)
# hmm_train = data.iloc[:cut]
# meta_train = data.iloc[cut:]
# meta_labels = labels[cut:]
# meta_weights = weights[cut:]
# closes = closes[cut:]
# # #
# hmm_scaled = hmm_scaler.fit_transform(hmm_train[hmm_features].values)
# joblib.dump(hmm_scaler, 'hmm_scaler.pkl')
# hmm = GaussianHMM(n_components=15, covariance_type="full", n_iter=131, min_covar=0.034885114865499216, random_state=42)
# hmm.fit(hmm_scaled)
# joblib.dump(hmm, 'trained_hmm.pkl')
# # hmm = joblib.load('trained_hmm.pkl')
# #
# meta_hmm = hmm_scaler.transform(meta_train[hmm_features].values)
# hmm_df = hmm_walk_forward(hmm, meta_hmm)
# #
# hmm_df = hmm_df.reset_index(drop=True)
# #hmm_df.drop(['state_2_proba', 'state_5_proba', 'state_6_proba', 'state_10_proba', 'state_11_proba', 'state_14_proba', 'state_0_proba'], axis=1, inplace=True)
# #drop useless (mostly useless) hmm features
# meta_train = meta_train.reset_index(drop=True)
# meta_train = pd.concat([meta_train, hmm_df], axis=1)
# #meta_features.extend(hmm_df.columns.tolist())
# #
# n = len(meta_train)
# cut = int(n * 0.98)
# data_train = meta_train.iloc[:cut]
# labels_train = meta_labels[:cut]
# weights_train = meta_weights[:cut]
# data_test = meta_train.iloc[cut:]
# labels_test = meta_labels[cut:]
# weights_test = meta_weights[cut:]
# closes_test = closes[cut:]

# models = load_meta_model('meta_model')
# preds, probas = predict_with_meta_model(models, data_test[meta_features], data_test[lstm_features])
# profit = trading_simulation(preds, closes_test)

#→ trying hidden1=84, hidden2=55, hidden3=9, lr=0.00060551, dropout=0.09111804, window=40, noise=0.20802087, ker=0.05681326, rec=0.00322820, rec_drop=0.21257794
#307
# stack, pred_labels, profit = build_meta_model(data_train[meta_features], labels_train, weights_train, data_test[meta_features], labels_test, closes_test, data_train[lstm_features], data_test[lstm_features],
#     lstm_hidden_units=[84,55,9], lstm_dropout=0.09111804, lstm_learning_rate=0.00060551, lstm_window_size=40, lstm_rec_drop=0.5,
#     lstm_noise=0.20802087, lstm_ker=0.05681326, lstm_rec=0.00322820)
# print(profit)
#286.2362360730648 at 0.5 rec_drop

#holdout threshold search
# thresh_grid = np.linspace(0.4, 0.6, 201)
# best_profit = -np.inf
# best_pair = (None, None)
# for min_proba in thresh_grid:
#     for max_proba in thresh_grid:
#         profit = trading_simulation(preds, probas, closes_test, min_proba, max_proba)
#         if profit > best_profit:
#             best_profit = profit
#             best_pair = (min_proba, max_proba)
#
# print(best_profit)
# print(best_pair)
#HMM hyperparameter tuning
# X = data[hmm_features].values
# X_scaled = scaler.fit_transform(X)
# tscv = TimeSeriesSplit(n_splits=5)
# hmm = HMMEstimator()
#
# hmm_search_space = {
#     "n_components": Integer(2, 20),
#     "covariance_type": Categorical(["full", "diag", "tied", "spherical"]),
#     "n_iter": Integer(5, 500),
#     "min_covar": Real(1e-8, 1, prior="log-uniform")
# }
#
# hmm_opt = BayesSearchCV(
#     estimator = hmm,
#     search_spaces = hmm_search_space,
#     cv = tscv,
#     n_iter = 30,
#     scoring = hmm_log_likelihood,
#     n_jobs = -1,
#     random_state = 42,
#     verbose = 1
# )
#
# hmm_opt.fit(X_scaled)
# print("HMM best →", hmm_opt.best_params_, "avg-loglike:", hmm_opt.best_score_)
#LGBM hyperparameter tuning
# tscv = TimeSeriesSplit(n_splits=5)
# lgbm = LGBMClassifier(random_state=42, num_threads=4, device='gpu', gpu_device_id=0, metric='auc')
#
# lgbm_search_space = [
#     {
#         "n_estimators": Integer(100, 3000),
#         "max_depth": Integer(1, 15),
#         "learning_rate": Real(1e-4, 0.3, prior="log-uniform"),
#         "num_leaves": Integer(20, 200),
#         "subsample": Real(0.5, 1.0),
#         "colsample_bytree": Real(0.1, 1.0),
#         "reg_alpha": Real(1e-9, 10.0, prior="log-uniform"),
#         "reg_lambda": Real(1e-9, 13.0, prior="log-uniform"),
#         "min_child_samples": Integer(5, 200),
#         "min_split_gain": Real(0.0, 1.0),
#         "subsample_freq": Integer(0, 10),
#         "scale_pos_weight": Real(0.1, 10.0, prior="log-uniform"),
#         "boosting_type": Categorical(["gbdt", "dart"])
#     },
#     {
#         "n_estimators": Integer(100, 3000),
#         "max_depth": Integer(1, 15),
#         "learning_rate": Real(1e-4, 0.3, prior="log-uniform"),
#         "num_leaves": Integer(20, 200),
#         "colsample_bytree": Real(0.1, 1.0),
#         "reg_alpha": Real(1e-9, 10.0, prior="log-uniform"),
#         "reg_lambda": Real(1e-9, 13.0, prior="log-uniform"),
#         "min_child_samples": Integer(5, 200),
#         "min_split_gain": Real(0.0, 1.0),
#         "scale_pos_weight": Real(0.1, 10.0, prior="log-uniform"),
#         "boosting_type": Categorical(["goss"])
#     }
# ]
#
# lgbm_opt = BayesSearchCV(
#     estimator = lgbm,
#     search_spaces = lgbm_search_space,
#     cv = tscv,
#     n_iter = 100,
#     scoring = "roc_auc",
#     n_jobs = -1,
#     random_state = 42,
#     verbose = 1
# )
#
# lgbm_opt.fit(data, labels, sample_weight=weights)
# print("LGBM best: ", lgbm_opt.best_params_, "auc:", lgbm_opt.best_score_)
# best_lgbm = lgbm_opt.best_estimator_
#ARCH hyperparameter tuning
# search_space = {
#     'p': Integer(1, 200),
#     'q': Integer(1, 200)
# }
#
# arch_bayes = BayesSearchCV(
#     estimator=ARCHWrapper(),
#     search_spaces=search_space,
#     n_iter=75,
#     cv=TimeSeriesSplit(5),
#     n_jobs=-1
# )
#
# X = ret.reshape(-1, 1)
# arch_bayes.fit(X, ret)
# print("Best params:", arch_bayes.best_params_)
# print("avg-loglike:", arch_bayes.best_score_)

#LSTM tuning
# space = [
#     Integer(16, 128, name="hidden1"),
#     Integer(8, 64,   name="hidden2"),
#     Integer(4, 32,   name="hidden3"),
#     Real(1e-4, 1e-2, "log-uniform", name="lr"),
#     Real(0.0, 0.5,  name="dropout"),
#     Integer(25, 45, name="window_size"),
#     Real(0.0, 0.5,  name="rec_dropout"),
#     Real(1e-4, 1, name="noise"),
#     Real(1e-4, 1e-1, name="ker"),
#     Real(1e-4, 1e-1, name="rec")
# ]
# def objective(params):
#     hidden1, hidden2, hidden3, lr, dropout, window_size, rec_dropout, noise, ker, rec = params
#     print(f"→ trying hidden1={hidden1}, hidden2={hidden2}, hidden3={hidden3}, "
#           f"lr={lr:.8f}, dropout={dropout:.8f}, window={window_size}, "
#           f"noise={noise:.8f}, ker={ker:.8f}, rec={rec:.8f}, rec_drop={rec_dropout:.8f}")
#
#     stack, preds, profit = build_meta_model(
#         data_train[meta_features], labels_train, weights_train,
#         data_test[meta_features], labels_test, closes_test, data_train[lstm_features], data_test[lstm_features],
#         lstm_hidden_units = [hidden1, hidden2, hidden3],
#         lstm_dropout = dropout,
#         lstm_learning_rate = lr,
#         lstm_window_size = window_size,
#         lstm_rec_drop = rec_dropout,
#         lstm_noise = noise,
#         lstm_ker = ker,
#         lstm_rec = rec
#     )
#     return -profit
#
# result = gp_minimize(
#     objective,
#     space,
#     n_calls=30,
#     random_state=42,
#     verbose=True
# )
#
# print("Best parameters:", result.x)
# print("Best profit:", -result.fun)

#hmm feature search
#hmm_feature_search = ['MFV', 'OBV', 'Stochastic-RSI', 'Upper_Wick', 'RSI_PC1', 'Lower_Wick', 'Lower_Wick_Ratio', 'BTC-ETH_Diff', 'Upper_Wick_Ratio', 'EX_PC2', 'RSI_PC2', 'Closing_Marubozu-OBV', 'Doji-RSI', 'Closing_Marubozu-ATR', 'Short_Candle-RSI', 'Returns', 'Volume-ATR', 'CCI', 'Volume', 'EX_PC3', 'ATR_Drawdown_4', 'Lower_Wick_Ratio_Mean_4', 'Low', 'Body_Ratio', 'TSI_Dev_48', 'Belt_Hold-ATR_Mean_1152', 'Fast_%D', 'Hikkake_Mean_1152', 'Returns_Mean_4', 'OBV_Dev_4', 'RSI_Drawdown_4', 'VWAP', 'Volume_Drawdown_4', 'Range', 'RSI-MACD_Dev_4', 'Day_Mean_384', 'LOW-HIGH_EMA_Mean_96', 'OBV_Dev_96', 'RSI_PC2_Mean_4', 'Closing_Marubozu-ATR_Mean_4', 'VWAP_Drawdown_16', 'Upper_Wick_Mean_4', 'Lower_Wick_Ratio_Dev_4', 'Closing_Marubozu-OBV_Mean_4', 'WILLR_Dev_48', 'Lower_Wick_Mean_4', 'KC_Upper_Dev_4', 'KC_Width_Drawdown_96', 'Day_Mean_1152', 'Volume_Change_Mean_4', 'Doji-ATR_Mean_96']
# X = data[hmm_feature_search].values
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# tscv = TimeSeriesSplit(n_splits=5)
# hmm = HMMEstimator(n_components=15, covariance_type="full", n_iter=131, min_covar=0.0349, random_state=42)
# selector = GeneticSelectionCV(
#         hmm,
#         cv=tscv,
#         verbose=1,
#         scoring=hmm_log_likelihood,
#         max_features=5,
#         n_population=40,
#         crossover_proba=0.5,
#         mutation_proba=0.2,
#         n_generations=20,
#         crossover_independent_proba=0.5,
#         mutation_independent_proba=0.05,
#         tournament_size=3,
#         n_gen_no_change=10,
#         caching=True,
#         n_jobs=5
#     )
#
# selector.fit(X_scaled, np.zeros((X_scaled.shape[0])))
# best_mask = selector.support_
# scores = selector.generation_scores_
# selected_idx = np.where(best_mask)[0]
# selected_feats = [hmm_features[i] for i in selected_idx]
#
# print("best scores: ")
# print(scores)
# print("Selected HMM features:", selected_feats)
#LSTM feature search
# lstm_feature_search = ['MFV', 'Stochastic-RSI', 'Upper_Wick', 'RSI_PC1', 'Lower_Wick', 'Lower_Wick_Ratio', 'BTC-ETH_Diff', 'Upper_Wick_Ratio', 'EX_PC2', 'RSI_PC2', 'Closing_Marubozu-OBV', 'Doji-RSI', 'Closing_Marubozu-ATR', 'Short_Candle-RSI', 'Returns', 'Volume-ATR', 'CCI', 'Volume', 'EX_PC3', 'ATR_Drawdown_4', 'Lower_Wick_Ratio_Mean_4', 'Low', 'Body_Ratio', 'TSI_Dev_48', 'Belt_Hold-ATR_Mean_1152', 'Fast_%D', 'Hikkake_Mean_1152', 'Returns_Mean_4', 'OBV_Dev_4', 'RSI_Drawdown_4', 'VWAP', 'Volume_Drawdown_4', 'Range', 'RSI-MACD_Dev_4', 'Day_Mean_384', 'LOW-HIGH_EMA_Mean_96', 'OBV_Dev_96', 'RSI_PC2_Mean_4', 'Closing_Marubozu-ATR_Mean_4', 'VWAP_Drawdown_16', 'Upper_Wick_Mean_4', 'Lower_Wick_Ratio_Dev_4', 'Closing_Marubozu-OBV_Mean_4', 'WILLR_Dev_48', 'Lower_Wick_Mean_4', 'KC_Upper_Dev_4', 'KC_Width_Drawdown_96', 'Day_Mean_1152', 'Volume_Change_Mean_4', 'Doji-ATR_Mean_96']
#
# mask_space = [
#     Categorical([0,1], name=f"m{i}")
#     for i in range(len(lstm_feature_search))
# ]
#
# def objective(mask):
#     selected = [f for f, m in zip(lstm_feature_search, mask) if m]
#     if "Direction" not in selected:
#         selected.append("Direction")
#     print(selected)
#     if len(selected) == 0:
#         return 1e6
#
#     stack, preds, profit = build_meta_model(data_train[meta_features], labels_train, weights_train,
#                                                   data_test[meta_features], labels_test, closes_test,
#                                                   data_train[selected], data_test[selected],
#                                                   lstm_hidden_units=[84, 55, 9], lstm_dropout=0.09111804,
#                                                   lstm_learning_rate=0.00060551, lstm_window_size=40, lstm_rec_drop=0.5,
#                                                   lstm_noise=0.20802087, lstm_ker=0.05681326, lstm_rec=0.00322820)
#
#     return -profit
#
# res = gp_minimize(
#     objective,
#     mask_space,
#     n_calls=50,
#     random_state=42,
#     verbose=True
# )
#
# best_mask = res.x
# best_features = [f for f,m in zip(meta_features, best_mask) if m]
# best_profit = -res.fun
#
# print("Selected features:", best_features)
# print("Profit:", best_profit)