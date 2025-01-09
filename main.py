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
from skopt.space import Real, Categorical, Integer
from xgboost import plot_importance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import TimeSeriesSplit
import pickle
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
import time

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print("Starting Bitcoin HMM analysis...")

def load_and_preprocess_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, names=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Num Trades', 'Label', 'ATR', 'Price_Change', 'Buy_Threshold', 'Sell_Threshold'])
    df.drop(['ATR', 'Price_Change', 'Buy_Threshold', 'Sell_Threshold'], axis=1, inplace=True)
    df.drop(0, inplace=True)
    #Drop useless row
    df = df.set_index('Timestamp')
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)

    print("Filtering out old data...")
    df = df[(df.index >= "2021-01-01")]

    print("Calculating returns and volatility...")
    df["Returns"] = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(window=24).std()

    #Replaces 0's with nan to avoid infinity being present in Volume_Change
    #Since the label column has 0's present, we need to make sure that they are not replaced
    columns_to_replace = df.columns.difference(['Label'])
    df[columns_to_replace] = df[columns_to_replace].replace(0, pd.NA)
    print("Calculating volume change...")
    df["Volume_Change"] = df["Volume"].pct_change()

    print("Calculating low period EMA")
    df['LOW_EMA'] = talib.EMA(df['Close'], timeperiod=14)

    print("Calculating high period EMA")
    df['HIGH_EMA'] = talib.EMA(df['Close'], timeperiod=50)

    print("Calculating RSI")
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)

    print("Calculating Aroon Oscillator")
    df['Aroon'] = talib.AROONOSC(df['High'], df['Low'], timeperiod=14)

    print("Calculating Bollinger Band width")
    upper, middle, lower = talib.BBANDS(df['Close'], 20)
    df['BB_Width'] = upper - lower

    print("Calculating MACD")
    macd, macdsignal, macdhist = talib.MACDFIX(df['Close'])
    df['MACD'] = macd
    df['MACDSignal'] = macdsignal
    df['MACDHist'] = macdhist

    print("Dropping nan values...")
    df.dropna(inplace=True)

    training_df = df[(df.index >= "2022-01-02")]
    out_of_training_df = df[((df.index >= "2021-01-01") & (df.index <= "2022-01-01"))]

    # Need this because there was a weird error were the data in these columns were not classified as floats, this caused a problem with the pipeline as I'm not using a target encoder
    training_df['Volume'] = pd.to_numeric(training_df['Volume'], errors='coerce')
    training_df['Num Trades'] = pd.to_numeric(training_df['Num Trades'], errors='coerce')
    training_df['Returns'] = pd.to_numeric(training_df['Returns'], errors='coerce')
    out_of_training_df['Volume'] = pd.to_numeric(out_of_training_df['Volume'], errors='coerce')
    out_of_training_df['Num Trades'] = pd.to_numeric(out_of_training_df['Num Trades'], errors='coerce')
    out_of_training_df['Returns'] = pd.to_numeric(out_of_training_df['Returns'], errors='coerce')

    return training_df, out_of_training_df

def train_hmm(data, features, scaler, n_components=3):
    print(f"Training HMM with {n_components} components...")
    X = data[features].values

    print("Normalizing features...")
    X_scaled = scaler.fit_transform(X)

    print("Fitting HMM model...")
    model = hmm.GaussianHMM(n_components=n_components, covariance_type='full', n_iter=50000, random_state=42, verbose=True)
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
    tscv = TimeSeriesSplit(n_splits=3)

    pipeline = ImbPipeline(steps=[
        ('smote', SMOTE(random_state=42)),
        ('xgb', XGBClassifier(random_state=42, eval_metric='mlogloss', early_stopping_rounds=25, verbose=0))
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
    plot_importance(xgboost_model)
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

def get_historical_data():
    unix = int(time.time())
    unix -= 60 * 60 * 100
    print(unix)

    url = 'https://pro-api.coinmarketcap.com/v2/cryptocurrency/ohlcv/historical'
    parameters = {
        'id': '1',
        'time_period': 'hourly',
        'time_start': unix,
        'count': '100',
        'interval': '1h'
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': 'be0057a3-f3af-4dbc-b189-79aa189efc25',
    }

    session = Session()
    session.headers.update(headers)

    try:
      response = session.get(url, params=parameters)
      data = json.loads(response.text)
    except (ConnectionError, Timeout, TooManyRedirects) as e:
      print(e)

    df = pd.DataFrame(columns=["Open", "High", "Low", "Close"])

    btc_close, btc_open, btc_high, btc_low = [], [], [], []
    for i in range(len(data['data']['quotes'])):
        df.loc[len(df)] = {"Open": data['data']['quotes'][i]['quote']['USD']['open'],
                           "High": data['data']['quotes'][i]['quote']['USD']['high'],
                           "Low": data['data']['quotes'][i]['quote']['USD']['low'],
                           "Close": data['data']['quotes'][i]['quote']['USD']['close']}

    return df

def calculate_indicators(df):
    df["Returns"] = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(window=24).std()
    df['LOW_EMA'] = talib.EMA(df['Close'], timeperiod=14)
    df['HIGH_EMA'] = talib.EMA(df['Close'], timeperiod=50)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['Aroon'] = talib.AROONOSC(df['High'], df['Low'], timeperiod=14)

    upper, middle, lower = talib.BBANDS(df['Close'], 20)
    df['BB_Width'] = upper - lower

    macd, macdsignal, macdhist = talib.MACDFIX(df['Close'])
    df['MACD'] = macd
    df['MACDSignal'] = macdsignal
    df['MACDHist'] = macdhist

    return df

print("Starting main execution...")
data_train, data_test = load_and_preprocess_data("btc_1h_data_2018_to_2024-2024-10-10_labeled.csv")
scaler = StandardScaler()
hmm_features = ["Close", "High", "Low", "Open", "LOW_EMA", "HIGH_EMA", "RSI", "Aroon", "BB_Width", "MACD", "MACDSignal", "MACDHist", "Volatility", "Returns"]
xg_features = ["Close", "High", "Low", "Open", "LOW_EMA", "HIGH_EMA", "RSI", "Aroon", "BB_Width", "MACD", "MACDSignal", "MACDHist", "Volatility", "State", "Returns"]

hmm_path = train_hmm(data_train, hmm_features, scaler, 14)
hmm_model = load_hmm(hmm_path)
print("Predicting states...")
states = predict_states(hmm_model, data_train, hmm_features, scaler)
states_out = predict_states(hmm_model, data_test, hmm_features, scaler)
data_train['State'] = states
data_test['State'] = states_out

#Build xgboost model
labels_train = data_train.pop('Label').tolist()
labels_train = [int(x) for x in labels_train]
labels_test = data_test.pop('Label').tolist()
labels_test = [int(x) for x in labels_test]

data_train = data_train[xg_features]
data_test = data_test[xg_features]

#xgboost_path = train_xgboost(data_train, labels_train, data_test, labels_test)
xgboost_path = "xgboost_pipeline.pkl"
xgboost_model = load_xgboost(xgboost_path)
labels_pred = xgboost_model.predict(data_test)
labels_pred = labels_pred.tolist()

# #Filtering lists so that there is only entries where either the real list or the predicted list has a 0 or 2 in them
# #Since buying and selling are the more important predictions in an actual algo trader I want to see only the buy/sell accuracy
buy_sell_label, buy_sell_label_pred = zip(*[(x, y) for x, y in zip(labels_test, labels_pred) if (x in [0, 2] or y in [0, 2])])
buy_sell_label = list(buy_sell_label)
buy_sell_label_pred = list(buy_sell_label_pred)
buy_sell_accuracy = accuracy_score(buy_sell_label, buy_sell_label_pred)
print(f"buy_sell accuracy: {buy_sell_accuracy}")

recent_df = get_historical_data()
recent_df = calculate_indicators(recent_df)
recent_df = recent_df.dropna()
recent_df['State'] = predict_states(hmm_model, recent_df, hmm_features, scaler)

labels_recent_pred = xgboost_model.predict(recent_df[xg_features])
labels_recent_pred = labels_recent_pred.tolist()
recent_df['Label'] = labels_recent_pred
recent_df['PriceDiff'] = recent_df['Close'].diff()

print(recent_df[['Close', 'Label', 'PriceDiff']])

