import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import talib
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

pd.set_option('display.max_columns', None)
print("Starting Bitcoin HMM analysis...")

def load_and_preprocess_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, names=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Num Trades', 'Label'])
    #Drop useless row
    df = df.drop(0)
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

    return training_df, out_of_training_df

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
        #Prepare bootstrap sample
        sample_data = samples[i]
        X = sample_data[features].values
        X_scaled = scaler.fit_transform(X)

        #Train HMM model
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

    #Predict states using each model
    for model in models:
        states = model.predict(X_scaled)
        all_predictions.append(states)

    #Aggregate the predictions using voting
    all_predictions = np.array(all_predictions)
    final_states = []

    for i in range(all_predictions.shape[1]):
        #Get the predictions for the current time step from all models
        state_votes = all_predictions[:, i]
        #Find the state that occurs most frequently
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

def calculate_transition_matrix(states, n_components):
    #Initialize the transition matrix with zeros
    transition_matrix = np.zeros((n_components, n_components))

    #Iterate through the state sequence to count transitions
    for i in range(len(states) - 1):
        current_state = states[i]
        next_state = states[i + 1]
        transition_matrix[current_state, next_state] += 1

    #Normalize the rows to convert counts to probabilities
    for i in range(n_components):
        row_sum = np.sum(transition_matrix[i])
        if row_sum > 0:
            transition_matrix[i] /= row_sum

    return transition_matrix

def transition_matrix_simulation(data_len, trans_matrix, starting_state):
    possible_states = [i for i in range(len(trans_matrix))]
    predicted_states = []
    for i in range(data_len):
        starting_state = np.random.choice(possible_states, p=trans_matrix[starting_state])
        predicted_states.append(starting_state)
    predicted_states = [int(item) for item in predicted_states]
    return predicted_states

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

print("Starting main execution...")
data, data_out = load_and_preprocess_data("btc_15m_data_2018_to_2024-2024-10-10_labeled.csv")
scaler = StandardScaler()
hmm_features = ["Close", "High", "Low", "Open", "LOW_EMA", "HIGH_EMA", "RSI", "Aroon", "BB_Width", "MACD", "MACDSignal", "MACDHist", "Volume", "Volatility"]
rf_features = ["Close", "High", "Low", "Open", "LOW_EMA", "HIGH_EMA", "RSI", "Aroon", "BB_Width", "MACD", "MACDSignal", "MACDHist", "Volume", "Volatility", "State"]

models = train_hmm_ensemble(data, hmm_features, scaler, 13, 13)
print("Predicting states...")
states = predict_ensemble_states(models, data, hmm_features, scaler)
states_out = predict_ensemble_states(models, data_out, hmm_features, scaler)
data['State'] = states
data_out['State'] = states_out

#Resample dataset so that it is balanced
# data_hold = data[data['Label'] == 1]
# data_buy = data[data['Label'] == 2]
# data_sell = data[data['Label'] == 0]
#
# max_len = max(len(data_buy), len(data_sell))
# data_hold_downsampled = resample(data_hold, replace=False, n_samples=max_len, random_state=42)
# data_balanced = pd.concat([data_hold_downsampled, data_buy, data_sell])

#Build random forest model
labels = data.pop('Label').tolist()
labels = [int(x) for x in labels]
data_train, data_test, label_train, label_test = train_test_split(data[rf_features], labels, test_size=0.2)

rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=300)
rf.fit(data_train, label_train)
label_pred = rf.predict(data_test)

#Need to do this for plotting results as data_test has completely random rows (thus completely random timestamps). This leads to a cluttered graph. This is also useful for testing data that is out of the training data
data_visual = data_out[len(data_out)-2000:]
label_visual = data_visual.pop('Label').tolist()
label_visual = [int(x) for x in label_visual]
data_visual = data_visual[rf_features]
label_visual_pred = rf.predict(data_visual)
label_visual_pred = label_visual_pred.tolist()

#Testing accuracy
accuracy = accuracy_score(label_test, label_pred)
print(f"test accuracy: {accuracy}")
accuracy = accuracy_score(label_visual, label_visual_pred)
print(f"visual accuracy: {accuracy}")
print(label_visual)
print(label_visual_pred)

#Filtering lists so that there is only entries where either the real list or the predicted list has a 0 or 2 in them
#Since buying and selling are the more important predictions in an actual algo trader I want to see only the buy/sell accuracy
buy_sell_label, buy_sell_label_pred = zip(*[(x, y) for x, y in zip(label_visual, label_visual_pred) if (x in [0, 2] or y in [0, 2])])
buy_sell_label = list(buy_sell_label)
buy_sell_label_pred = list(buy_sell_label_pred)
buy_sell_accuracy = accuracy_score(buy_sell_label, buy_sell_label_pred)
print(f"buy_sell accuracy: {buy_sell_accuracy}")
print(buy_sell_label)
print(buy_sell_label_pred)

print("Plotting predicted states...")
plot_results_hmm(data, states, 13)

print("Plotting real and predicted labels...")
plot_results_rf(data_visual, np.array(label_visual), np.array(label_visual_pred))

#print(label_test)
#print(label_pred.tolist())

# print("Analyzing states...")
# analyze_states(data_test, features, states, 18)
#
# print("Calculating transition matrix...")
# trans_matrix = calculate_transition_matrix(states, 18)
# trans_df = pd.DataFrame(trans_matrix)
# print(trans_df)
#


#differences_count = sum(1 for a, b in zip(states[len(states)-1000:], trans_states) if a != b)
#print(f"Differences between actual predicted states and transition matrix estimation: {differences_count}")

