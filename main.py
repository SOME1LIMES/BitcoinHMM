import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import talib
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from itertools import combinations
import random
from sklearn.cluster import KMeans
import statistics

pd.set_option('display.max_columns', None)
print("Starting Bitcoin HMM analysis...")

def load_and_preprocess_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, names=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Num Trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
    df.drop(columns=['Close time', 'Quote asset volume', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'], inplace=True)
    #drop useless row
    df = df.drop(0)
    #df = df.drop(columns=['Timestamp'])
    df = df.set_index('Timestamp')
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)

    print("Filtering out old data...")
    df = df[(df.index >= "2024-02-28")]

    print("Calculating returns and volatility...")
    df["Returns"] = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(window=24).std()

    # replaces 0's with nan to avoid infinity being present in Volume_Change
    df = df.replace(0, pd.NA)
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

    print(f"Data preprocessed, df shape:{df.shape}")
    return df

def train_hmm(data, n_components=3):
    print(f"Training HMM with {n_components} components...")
    features = ["EMA", "RSI", "Aroon", "Volatility"]
    X = data[features].values

    print("Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Fitting HMM model...")
    model = hmm.GaussianHMM(n_components=n_components, covariance_type='full', n_iter=50000, random_state=42, verbose=True)
    model.fit(X_scaled)

    print("HMM training complete")
    return model, scaler

def train_hmm_ensemble(data, features, scaler, n_components=3, sample_num=30):
    models = []
    samples = np.array_split(data, sample_num)

    for i in range(sample_num):
        # Prepare bootstrap sample
        sample_data = samples[i]
        X = sample_data[features].values
        X_scaled = scaler.fit_transform(X)

        # Train HMM model
        model = hmm.GaussianHMM(n_components=n_components, covariance_type='full', n_iter=300, init_params='stmc',
                                random_state=i)
        model.fit(X_scaled)
        models.append(model)

    print(f"Ensemble of {sample_num} HMM models trained.")
    return models


def predict_ensemble_states(models, data, features, scaler):
    X = data[features].values
    X_scaled = scaler.transform(X)
    all_predictions = []

    # Predict states using each model
    for model in models:
        states = model.predict(X_scaled)
        all_predictions.append(states)

    # Aggregate the predictions using voting
    all_predictions = np.array(all_predictions)
    final_states = []

    for t in range(all_predictions.shape[1]):
        # Get the predictions for the current time step from all models
        state_votes = all_predictions[:, t]
        # Find the state that occurs most frequently (voting)
        final_state = np.bincount(state_votes).argmax()
        final_states.append(final_state)

    final_states = np.array(final_states)
    print(f"Final states predicted using ensemble voting. Unique states: {np.unique(final_states)}")
    return final_states

def analyze_states(data, features, states, n_components=3):
    print("Analyzing states...")
    df_analysis = data.copy()
    df_analysis['State'] = states

    for state in range(n_components):
        print(f"\nAnalyzing State {state}:")
        state_data = df_analysis[df_analysis['State'] == state]
        print(state_data[features].describe())
        print(f"Number of periods in State {state}: {len(state_data)}")

def plot_results(data, states, n_components):
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
    plt.savefig('plots/' + str(timestamp))

print("Starting main execution...")
data = load_and_preprocess_data("btc_15m_data_2018_to_2024-2024-10-10.csv")

scaler = StandardScaler()
features = ["Close", "High", "Low", "Open", "LOW_EMA", "HIGH_EMA", "RSI", "Aroon", "BB_Width", "MACD", "MACDSignal", "MACDHist", "Volume", "Volatility"]
models = train_hmm_ensemble(data, features, scaler, 10, 60)

# print("Training HMM model...")
# model, scaler = train_hmm(data)
print("Predicting states...")
states = predict_ensemble_states(models, data, features, scaler)

print("Analyzing states...")
analyze_states(data, features, states, 10)
#
print("Plotting results...")
plot_results(data, states, 10)
#
# print("Printing transition matrix...")
# print(model.transmat_)
# print("Start probabilities:")
# print(model.startprob_)
#
# features = ["EMA", "RSI", "Aroon", "Close"]
# X = data[features].values
#
# #print("\nPrinting means and covariances of each state")
# #for i in range(model.n_components):
#  #   print(f"State {i}:")
#   #  print("Mean:", model.means_[i])
#    # print("Covariance:", model.covars_[i])
#
# print("Bitcoin HMM analysis complete")
