import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import talib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from itertools import combinations
import random
from sklearn.cluster import KMeans

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
    df = df[(df.index >= "2024-01-01")]

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

#need this function because the pandas sample function takes random rows whereas I need consecutive rows
def generate_consecutive_samples(df, sample_length=100):
    max_start_index = len(df) - sample_length
    start_index = np.random.randint(0, max_start_index + 1)
    sample_df = df.iloc[start_index:start_index + sample_length]
    return sample_df

def bootstrap_train_hmm(data, features, scaler, n_components=3, sample_num=30):
    def hamming_distance(seq1, seq2):
        return np.sum(seq1 != seq2) / len(seq1)

    features = features
    scaler = scaler
    state_predictions = []
    models = []
    samples = np.array_split(data, sample_num)

    #create test data
    test_sample = samples[len(samples)-1]

    #create test data
    X_test = test_sample[features].values
    X_scaled_test = scaler.fit_transform(X_test)
    #the range is len(samples)-2 so that the test_sample is not included
    for i in range(len(samples)-2):
        #Create sample, get feature data and scale data
        X = samples[i][features].values
        X_scaled = scaler.fit_transform(X)
        #Apply K-Means clustering to initialize HMM means
        kmeans = KMeans(n_clusters=n_components, random_state=42)
        kmeans.fit(X_scaled)
        initial_means = kmeans.cluster_centers_
        #Initialize and train the HMM model
        model = hmm.GaussianHMM(n_components=n_components, covariance_type='full', n_iter=300, init_params='st', random_state=42)
        models.append(model)
        model.means_ = initial_means  # Set initial means from K-Means
        model.covars_ = np.array([np.cov(X_scaled.T) + 1e-4 * np.eye(X_scaled.shape[1]) for _ in range(n_components)])
        model.fit(X_scaled)
        #predict states with each bootstrap model
        states = model.predict(X_scaled_test)
        states = states.tolist()
        state_predictions.append(states)

    total_hamming = 0
    current_state_prediction = state_predictions[0]
    hamming_distances = []
    used = []
    i = 0
    j = 0
    #this whole while loop is meant to find the lowest average hamming distance from state predictions, we can then use this to find the best bootstrap model
    while current_state_prediction not in used:
        if i != j: #To make sure we're not calculating the hamming distance between the same state predictions
            total_hamming += hamming_distance(current_state_prediction, state_predictions[i])
        i += 1
        if i >= len(state_predictions):
            j += 1
            if j >= len(state_predictions):
                break
            used.append(current_state_prediction)
            average_hamming_distance = total_hamming / i
            hamming_distances.append(average_hamming_distance)
            current_state_prediction = state_predictions[j]
            total_hamming = 0
            i = 0

    #average_hamming_distance = total_hamming / pair_count
    #Here I'm finding the model with the tenth highest accuracy in hopes that it will have a more even distribution of states
    sorted_indices = np.argsort(hamming_distances)
    sorted_hamming_distances = [hamming_distances[i] for i in sorted_indices]
    print(hamming_distances)
    print(sorted_hamming_distances)
    low_hamming_distance = sorted_hamming_distances[9]
    decent_model = models[sorted_indices[9]]
    print(low_hamming_distance)
    return decent_model

def predict_states(model, data, features, scaler):
    print("Predicting states...")
    features = features
    X = data[features].values
    X_scaled = scaler.transform(X)
    states = model.predict(X_scaled)
    print(f"States precicted. Unique states: {np.unique(states)}")
    return states

def analyze_states(model, data, features, states):
    print("Analyzing states...")
    df_analysis = data.copy()
    df_analysis['State'] = states

    for state in range(model.n_components):
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
    plt.savefig('my_plot.png')

print("Starting main execution...")
data = load_and_preprocess_data("btc_15m_data_2018_to_2024-2024-10-10.csv")

scaler = StandardScaler()
features = ["Close", "High", "Low", "Open", "LOW_EMA", "HIGH_EMA", "RSI", "Aroon", "BB_Width", "MACD", "MACDSignal", "MACDHist", "Volume", "Volatility"]
best_model = bootstrap_train_hmm(data, features, scaler, 4, 40)

# print("Training HMM model...")
# model, scaler = train_hmm(data)
print("Predicting states...")
states = predict_states(best_model, data, features, scaler)

print("Analyzing states...")
analyze_states(best_model, data, features, states)
#
print("Plotting results...")
plot_results(data, states, 4)
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
