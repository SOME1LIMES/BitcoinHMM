import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import talib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

print("Starting Bitcoin HMM analysis...")

def load_and_preprocess_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, names=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Num Trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
    df.drop(columns=['Close time', 'Quote asset volume', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'], inplace=True)
    #drop useless row
    df = df.drop(0)
    df = df.set_index('Timestamp')
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)

    print("Filtering out old data...")
    df = df[(df.index >= "2020-01-01")]

    print("Calculating returns and volatility...")
    df["Returns"] = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(window=24).std()

    # replaces 0's with nan to avoid infinity being present in Volume_Change
    df = df.replace(0, pd.NA)
    print("Calculating volume change...")
    df["Volume_Change"] = df["Volume"].pct_change()

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

def train_hmm(data, test_size=0.1, n_components=5):
    train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)

    print(f"Training HMM with {n_components} components...")
    features = ["Close"]
    X_train = train_data[features].values
    X_test = test_data[features].values

    print("Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Fitting HMM model...")
    model = hmm.GaussianHMM(n_components=n_components, covariance_type='full', n_iter=1000, random_state=42, verbose=True, tol=0.0001)
    model.fit(X_train_scaled)

    train_states = model.predict(X_train_scaled)
    test_states = model.predict(X_test_scaled)

    shifted_states = np.roll(test_states, -1)[:-1]
    original_states = test_states[:-1]

    # Calculate the accuracy as the fraction of times the states stay the same
    accuracy = np.mean(original_states == shifted_states)

    print(f"Testing State Prediction Consistency Score: {accuracy}")
    return model, scaler, train_states, train_data

# def predict_states(model, data, scaler):
#     print("Predicting states...")
#     features = ["EMA", "RSI", "Aroon", "Close"]
#     X = data[features].values
#     X_scaled = scaler.transform(X)
#     states = model.predict(X_scaled)
#     print(f"States precicted. Unique states: {np.unique(states)}")
#     return states

def analyze_states(model, data, states):
    print("Analyzing states...")
    df_analysis = data.copy()
    df_analysis['State'] = states

    for state in range(model.n_components):
        print(f"\nAnalyzing State {state}:")
        state_data = df_analysis[df_analysis['State'] == state]
        print(state_data[["Close"]].describe())
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
    print("Saving plot...")
    plt.savefig('my_plot.png')

print("Starting main execution...")
data = load_and_preprocess_data("btc_1h_data_2018_to_2024-2024-10-10.csv")

print("Training HMM model...")
model, scaler, states, train_data = train_hmm(data)

print("Analyzing states...")
analyze_states(model, train_data, states)

print("Plotting results...")
plot_results(model, train_data, states)

print("Printing transition matrix...")
print(model.transmat_)
print("Start probabilities:")
print(model.startprob_)

features = ["EMA", "RSI", "Aroon", "Close"]
X = train_data[features].values

#print("\nPrinting means and covariances of each state")
#for i in range(model.n_components):
 #   print(f"State {i}:")
  #  print("Mean:", model.means_[i])
   # print("Covariance:", model.covars_[i])

print("Bitcoin HMM analysis complete")
