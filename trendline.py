import pandas as pd
import numpy as np
import talib
from catboost import CatBoostClassifier

def walkforward_model(close: np.array, trades: pd.DataFrame, data_x: pd.DataFrame, data_y: pd.Series, train_size: int, step_size: int):
    signal = np.zeros(len(close))
    prob_signal = np.zeros(len(close))
    next_train = train_size
    trade_i = 0
    in_trade = False
    tp_price = None
    sl_price = None
    hp_i = None
    model = None
    for i in range(len(close)):
        if i == next_train:
            start_i = i - train_size
            train_indices = trades[(trades['entry_i'] > start_i) & (trades['exit_i'] < i)].index
            x_train = data_x.loc[train_indices]
            y_train = data_y.loc[train_indices]
            print('training', i, 'N cases', len(train_indices))
            model = CatBoostClassifier(random_state=42)
            model.fit(x_train.to_numpy(), y_train.to_numpy())
            next_train += step_size
        if in_trade:
            if close[i] >= tp_price or close[i] <= sl_price or i >= hp_i:
                signal[i] = 0
                prob_signal[i] = 0
                in_trade = False
            else:
                signal[i] = signal[i - 1]
                prob_signal[i] = prob_signal[i - 1]
        if trade_i < len(trades) and i == trades['entry_i'].iloc[trade_i]:
            if model is not None:
                prob = model.predict_proba(data_x.iloc[trade_i].to_numpy().reshape(1, -1))[0][1]
                prob_signal[i] = prob
                trades.loc[trade_i, 'model_prob'] = prob
                if prob > 0.5:
                    signal[i] = 1
                in_trade = True
                trade = trades.iloc[trade_i]
                tp_price = trade['tp']
                sl_price = trade['sl']
                hp_i = trade['hp_i']
            trade_i += 1
    return signal, prob_signal


def trendline_breakout_dataset(ohlcv: pd.DataFrame, lookback: int, hold_period: int = 12, tp_mult: float = 3.0, sl_mult: float = 3.0, atr_lookback: int = 168):
    assert (atr_lookback >= lookback)
    close = np.log(ohlcv['Close'].to_numpy())
    atr_arr = talib.ATR(np.log(ohlcv['High']), np.log(ohlcv['Low']), np.log(ohlcv['Close']), atr_lookback)
    vol_arr = (ohlcv['Volume'] / ohlcv['Volume'].rolling(atr_lookback).median()).to_numpy()
    adx_arr = talib.ADX(ohlcv['High'], ohlcv['Low'], ohlcv['Close'], lookback)

    trades = pd.DataFrame()
    trade_i = 0

    in_trade = False
    tp_price = None
    sl_price = None
    hp_i = None
    for i in range(atr_lookback, len(ohlcv)):
        window = close[i - lookback: i]
        s_coefs, r_coefs = fit_trendlines_single(window)
        r_val = r_coefs[1] + lookback * r_coefs[0]

        if not in_trade and close[i] > r_val:
            tp_price = close[i] + atr_arr[i] * tp_mult
            sl_price = close[i] - atr_arr[i] * sl_mult
            hp_i = i + hold_period
            in_trade = True
            trades.loc[trade_i, 'entry_i'] = i
            trades.loc[trade_i, 'entry_p'] = close[i]
            trades.loc[trade_i, 'atr'] = atr_arr[i]
            trades.loc[trade_i, 'sl'] = sl_price
            trades.loc[trade_i, 'tp'] = tp_price
            trades.loc[trade_i, 'hp_i'] = i + hold_period
            trades.loc[trade_i, 'slope'] = r_coefs[0]
            trades.loc[trade_i, 'intercept'] = r_coefs[1]

            trades.loc[trade_i, 'resist_s'] = r_coefs[0] / atr_arr[i]
            line_vals = (r_coefs[1] + np.arange(lookback) * r_coefs[0])
            err = np.sum(line_vals - window) / lookback
            err /= atr_arr[i]
            trades.loc[trade_i, 'tl_err'] = err

            diff = line_vals - window
            trades.loc[trade_i, 'max_dist'] = diff.max() / atr_arr[i]
            trades.loc[trade_i, 'vol'] = vol_arr[i]
            trades.loc[trade_i, 'adx'] = adx_arr[i]

        if in_trade:
            if close[i] >= tp_price or close[i] <= sl_price or i >= hp_i:
                trades.loc[trade_i, 'exit_i'] = i
                trades.loc[trade_i, 'exit_p'] = close[i]

                in_trade = False
                trade_i += 1

    trades['return'] = trades['exit_p'] - trades['entry_p']
    data_x = trades[['resist_s', 'tl_err', 'vol', 'max_dist', 'adx']]
    data_y = pd.Series(0, index=trades.index)
    data_y.loc[trades['return'] > 0] = 1

    return trades, data_x, data_y


def fit_trendlines_single(data: np.array):
    x = np.arange(len(data))
    coefs = np.polyfit(x, data, 1)
    line_points = coefs[0] * x + coefs[1]
    upper_pivot = (data - line_points).argmax()
    lower_pivot = (data - line_points).argmin()
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], data)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)
    return (support_coefs, resist_coefs)


def optimize_slope(support: bool, pivot: int, init_slope: float, y: np.array):
    slope_unit = (y.max() - y.min()) / len(y)

    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step

    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)
    assert (best_err >= 0.0)

    get_derivative = True
    derivative = None
    while curr_step > min_step:

        if get_derivative:
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err;

            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err

            if test_err < 0.0:
                raise Exception("Derivative failed. Check your data. ")
            get_derivative = False

        if derivative > 0.0:
            test_slope = best_slope - slope_unit * curr_step
        else:
            test_slope = best_slope + slope_unit * curr_step

        test_err = check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err:
            curr_step *= 0.5
        else:
            best_err = test_err
            best_slope = test_slope
            get_derivative = True

    return (best_slope, -best_slope * pivot + y[pivot])


def check_trend_line(support: bool, pivot: int, slope: float, y: np.array):
    intercept = -slope * pivot + y[pivot]
    line_vals = slope * np.arange(len(y)) + intercept
    diffs = line_vals - y

    if support and diffs.max() > 1e-5:
        return -1.0
    elif not support and diffs.min() < -1e-5:
        return -1.0

    err = (diffs ** 2.0).sum()
    return err;
