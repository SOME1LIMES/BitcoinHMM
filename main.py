import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import talib
from datetime import datetime

print("Starting Bitcoin HMM analysis...")

def load_and_preprocess_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, names=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    #drop useless row
    df = df.drop(0)
    df = df.astype(float)
    df['Unix'] = df['Timestamp']
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.set_index('Timestamp')

    print("Resampling to hourly values...")
    df = df.resample('h').mean()

    print("Calculating returns and volatility...")
    df["Returns"] = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(window=48).std()

    # replaces 0's with nan to avoid infinity being present in Volume_Change
    df = df.replace(0, pd.NA)
    #print("Calculating volume change...")
    #df["Volume_Change"] = df["Volume"].pct_change()

    print("Filter out older data")
    df = df[(df['Unix'] >= 1514782800) & (df['Unix'] <= 1546232400)]

    print("Calculating EMA")
    df['EMA'] = talib.EMA(df['Close'], timeperiod=30)

    print("Calculating RSI")
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)

    print("Calculating Aroon Oscillator")
    df['Aroon'] = talib.AROONOSC(df['High'], df['Low'], timeperiod=14)

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

def predict_states(model, data, scaler):
    print("Predicting states...")
    features = ["EMA", "RSI", "Aroon", "Volatility"]
    X = data[features].values
    X_scaled = scaler.transform(X)
    states = model.predict(X_scaled)
    print(f"States precicted. Unique states: {np.unique(states)}")
    return states

def analyze_states(model, data, states):
    print("Analyzing states...")
    df_analysis = data.copy()
    df_analysis['State'] = states

    for state in range(model.n_components):
        print(f"\nAnalyzing State {state}:")
        state_data = df_analysis[df_analysis['State'] == state]
        print(state_data[["EMA", "RSI", "Aroon", "Volatility"]].describe())
        print(f"Number of periods in State {state}: {len(state_data)}")

def plot_results(model, data, states):
    print("Plotting results...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,10), sharex=True)

    ax1.plot(data.index, data['Close'])
    ax1.set_title('Bitcoin Price and HMM States')
    ax1.set_ylabel('Price')

    for state in range(model.n_components):
        mask = (states == state)
        ax1.fill_between(data.index, data['Close'].min(), data['Close'].max(), where=mask, alpha=0.3, label=f'State {state}')

    ax1.legend()

    ax2.plot(data.index, data['Returns'])
    ax2.set_title('Bitcoin Returns')
    ax2.set_ylabel('Returns')
    ax2.set_xlabel('Date')

    plt.tight_layout()
    print("Showing plot...")
    plt.show()

print("Starting main execution...")
data = load_and_preprocess_data("btcusd_1-min_data.csv")

print("Training HMM model...")
model, scaler = train_hmm(data)

print("Predicting states...")
states = predict_states(model, data, scaler)

print("Analyzing states...")
analyze_states(model, data, states)

print("Plotting results...")
plot_results(model, data, states)

print("Printing transition matrix...")
print(model.transmat_)

print("\nPrinting means and covariances of each state")
for i in range(model.n_components):
    print(f"State {i}:")
    print("Mean:", model.means_[i])
    print("Covariance:", model.covars_[i])

print("Bitcoin HMM analysis complete")
