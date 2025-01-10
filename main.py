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

def load_and_preprocess_data(filepath, window_size=2):
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

    df.dropna(inplace=True)

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
    tscv = TimeSeriesSplit(n_splits=5)

    pipeline = ImbPipeline(steps=[
        ('smote', SMOTE(random_state=42)),
        ('xgb', XGBClassifier(random_state=42, eval_metric='mlogloss', early_stopping_rounds=25))
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

    #xgboost_step = opt.best_estimator_.steps[1]
    #xgboost_model = xgboost_step[1]
    #plot_importance(xgboost_model)
    #plt.show()

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

def calculate_indicators(df, window_size=2):
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

def convert_data_to_windows(data, window_size=50):
    rows = []

    for i in range(len(data) - window_size):
        row = {}

        for feature in data.columns:
            if feature != 'Label':
                for t in range(window_size):
                    row[f"{feature}_t-{window_size-t}"] = data[feature].iloc[i + t]

        row['Label'] = data['Label'].iloc[i + window_size]

        rows.append(row)

    window_data = pd.DataFrame(rows)

    return window_data


print("Starting main execution...")
window_size = 2
data_train, data_eval, data_test = load_and_preprocess_data("btc_1h_data_2018_to_2024-2024-10-10_labeled.csv", window_size)
scaler = StandardScaler()
hmm_features = ["Close", "High", "Low", "Open", "LOW_EMA", "HIGH_EMA", "RSI", "Aroon", "BB_Width", "MACD", "MACDSignal", "MACDHist", "Volatility", "Returns"]
xg_features = ["Close", "High", "Low", "Open", "LOW_EMA", "HIGH_EMA", "RSI", "Aroon", "BB_Width", "MACD", "MACDSignal", "MACDHist", "Volatility", "State", "Returns", "Label"]
agg_features = ["Close_Mean", "High_Mean", "Low_Mean", "Open_Mean", "LOW_EMA_Mean", "HIGH_EMA_Mean", "RSI_Mean", "Aroon_Mean", "BB_Width_Mean", "MACD_Mean", "MACDSignal_Mean", "MACDHist_Mean", "Volatility_Mean", "Returns_Mean",
                "Close_Dev", "High_Dev", "Low_Dev", "Open_Dev", "LOW_EMA_Dev", "HIGH_EMA_Dev", "RSI_Dev", "Aroon_Dev", "BB_Width_Dev", "MACD_Dev", "MACDSignal_Dev", "MACDHist_Dev", "Volatility_Dev", "Returns_Dev",
                "Close_Drawdown", "High_Drawdown", "Low_Drawdown", "Open_Drawdown", "LOW_EMA_Drawdown", "HIGH_EMA_Drawdown", "RSI_Drawdown", "Aroon_Drawdown", "BB_Width_Drawdown", "MACD_Drawdown", "MACDSignal_Drawdown", "MACDHist_Drawdown", "Volatility_Drawdown", "Returns_Drawdown"]

hmm_path = train_hmm(data_train, hmm_features, scaler, 8)
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

xg_labels_train = xg_data_train.pop('Label').tolist()
xg_labels_train = [int(x) for x in xg_labels_train]
xg_labels_eval = xg_data_eval.pop('Label').tolist()
xg_labels_eval = [int(x) for x in xg_labels_eval]
xg_labels_test = xg_data_test.pop('Label').tolist()
xg_labels_test = [int(x) for x in xg_labels_test]

#xgboost_path = train_xgboost(xg_data_train, xg_labels_train, xg_data_eval, xg_labels_eval)
xgboost_path = "xgboost_pipeline.pkl"
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

recent_df = get_historical_data()
recent_df = calculate_indicators(recent_df)
recent_df = recent_df.dropna()
recent_df['State'] = predict_states(hmm_model, recent_df, hmm_features, scaler)
print(recent_df['Close'])

#Need this to make recent_df compatible with convert_data_to_windows method
recent_df['Label'] = 0
recent_df_windowed = convert_data_to_windows(recent_df[xg_features], window_size)

#recent_df = recent_df.drop(xg_features, axis=1)
recent_df = recent_df.drop(index=recent_df.index[:window_size])
recent_df = recent_df.reset_index(drop=True)
recent_df_windowed = recent_df_windowed.reset_index(drop=True)
recent_df = pd.concat([recent_df_windowed, recent_df[agg_features]], axis=1)
recent_df.drop(['Label'], axis=1, inplace=True)

labels_recent_pred = xgboost_model.predict(recent_df)
labels_recent_pred = labels_recent_pred.tolist()
recent_df['Label'] = labels_recent_pred
recent_df['PriceDiff'] = recent_df['Close_t-1'].diff()

print(recent_df[['Close_t-1', 'Label', 'PriceDiff']])

