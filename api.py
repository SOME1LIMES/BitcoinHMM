import time
import json
import requests
from requests.exceptions import ConnectionError, RequestException
import urllib.parse
import hmac
import hashlib
import pandas as pd
import datetime
from decimal import Decimal

api_key = '3VogoEDvzJYbzG8xF1z7VDHCZMwvHpnsG9lCh04oQ6wKfl4EQ7PyYizYOpnSDh75'
api_sec = 'n40V3JFTVxlrOZ3KNevHGk7MjtsQDQYpgavMdVGWYlggyN80C4ZgOOpCNSxHs3JH'
def get_binanceus_signature(data, secret):
    postdata = urllib.parse.urlencode(data)
    message = postdata.encode()
    byte_key = bytes(secret, 'UTF-8')
    mac = hmac.new(byte_key, message, hashlib.sha256).hexdigest()
    return mac

def get_exchange_info(symbol="BTCUSDC"):
    url = "https://api.binance.us/api/v3/exchangeInfo"
    params = {"symbol": symbol}
    response = requests.get(url, params=params)
    return response.json()

def get_symbol_filters(symbol="BTCUSDC"):
    info = get_exchange_info(symbol)
    f = info['symbols'][0]['filters']
    return {x['filterType']: x for x in f}

def buy_btc_market(quoteOrderQty):
    data = {
        "symbol": 'BTCUSDC',
        "side": 'BUY',
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

def sell_btc_market(qty):
    data = {
        "symbol": 'BTCUSDC',
        "side": 'SELL',
        "type": 'MARKET',
        "quantity": qty,
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

def buy_btc_limit(qty, price):
    data = {
        "symbol": 'BTCUSDC',
        "side": 'BUY',
        "type": 'LIMIT',
        "timeInForce": 'GTC',
        "quantity": qty,
        "price": price,
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

def sell_btc_limit(qty, price):
    data = {
        "symbol": 'BTCUSDC',
        "side": 'SELL',
        "type": 'LIMIT',
        "timeInForce": 'GTC',
        "quantity": qty,
        "price": price,
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

def cancel_open_orders():
    data = {
        "symbol": 'BTCUSDC',
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
    req = requests.delete(('https://api.binance.us/api/v3/openOrders'), headers=headers, params=payload)
    return req.text

def get_all_open_orders():
    data = {
        "symbol": 'BTCUSDC',
        "timestamp": int(round(time.time() * 1000))
    }
    headers = {
        'X-MBX-APIKEY': api_key
    }
    signature = get_binanceus_signature(data, api_sec)
    data["signature"] = signature
    req = requests.get(('https://api.binance.us/api/v3/openOrders'), headers=headers, params=data)
    return req.json()

def get_historical_data(start_time, end_time, symbol='BTCUSDC'):
    url = 'https://api.binance.us/api/v3/klines'
    headers = {
        'X-MBX-APIKEY': api_key,
    }
    parameters = {
        'symbol': symbol,
        'interval': '15m',
        'startTime': str(start_time),
        'limit': '1000'
    }

    flag = True
    dataframes = []
    session = requests.Session()
    while(flag):
        session.headers.update(headers)

        try:
            response = session.get(url, params=parameters)
            data = json.loads(response.text)
        except (ConnectionError, requests.Timeout, requests.TooManyRedirects) as e:
            print(e)

        df = pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])

        for i in range(len(data)):
            df.loc[len(df)] = {"Timestamp": int(data[i][6]),
                               "Open": float(data[i][1]),
                               "High": float(data[i][2]),
                               "Low": float(data[i][3]),
                               "Close": float(data[i][4]),
                               "Volume": float(data[i][5])}
            close_time = int(data[i][6])

            if close_time >= end_time:
                flag = False
        dataframes.append(df)

        start_time += (1000*15*60000) #add time in milliseconds
        parameters['startTime'] = start_time #update time

    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def get_recent_data(count='1000', symbol='BTCUSDC'):
    url = 'https://api.binance.us/api/v3/klines'
    headers = {
        'X-MBX-APIKEY': api_key,
    }
    parameters = {
        'symbol': symbol,
        'interval': '15m',
        'limit': count
    }

    session = requests.Session()
    session.headers.update(headers)

    try:
        response = session.get(url, params=parameters)
        data = json.loads(response.text)
    except (ConnectionError, requests.Timeout, requests.TooManyRedirects) as e:
        print(e)

    df = pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])

    for i in range(len(data)):
        df.loc[len(df)] = {"Timestamp": int(data[i][6]),
                            "Open": float(data[i][1]),
                            "High": float(data[i][2]),
                            "Low": float(data[i][3]),
                            "Close": float(data[i][4]),
                            "Volume": float(data[i][5])}

    return df

def get_balances():
    headers = {
        'X-MBX-APIKEY': api_key,
    }
    parameters = {
        'timestamp': int(time.time() * 1000)
    }
    signature = get_binanceus_signature(parameters, api_sec)
    parameters['signature'] = signature
    req = requests.get(('https://api.binance.us/api/v3/account'), headers=headers, params=parameters)

    req = req.json()
    balances = req['balances']

    btc_entry = next(item for item in balances if item['asset'] == 'BTC')
    btc_free = float(btc_entry['free'])
    usdc_entry = next(item for item in balances if item['asset'] == 'USDC')
    usdc_free = float(usdc_entry['free'])
    return btc_free, usdc_free

#in case of network errors, get_balances is crucial for the trading loop to function correctly
def safe_get_balances(max_retries=7, backoff_factor=0.5):
    for attempt in range(1, max_retries + 1):
        try:
            return get_balances()
        except RequestException as e:
            if attempt == max_retries:
                break
            wait = backoff_factor * (2 ** (attempt - 1))
            print(f"[Balance fetch failed: {e!r}] retry {attempt}/{max_retries} in {wait:.1f}s…")
            time.sleep(wait)

    raise RuntimeError(f"Unable to fetch balances after {max_retries} attempts")

def qdown_dec(value, step):
    d = Decimal(str(value))
    return (d // step) * step

def str_dec(d):
    return format(d, "f")

def passes_filters(qty, price, min_qty, min_notional):
    if qty <= 0:
        return False
    if qty < min_qty:
        return False
    if (qty * price) < min_notional:
        return False
    return True

def place_limit_buy_checked(spend_usdc, price_dec, qty_step, min_qty, min_notional, price_tick):
    raw_qty = Decimal(str(spend_usdc)) / price_dec
    qty = qdown_dec(raw_qty, qty_step)
    price_rounded = qdown_dec(price_dec, price_tick)
    if not passes_filters(qty, price_rounded, min_qty, min_notional):
        return None, "SKIPPED: qty/notional below limits"
    resp_txt = buy_btc_limit(str_dec(qty), str_dec(price_rounded))
    return json.loads(resp_txt), resp_txt

def place_limit_sell_checked(btc_qty, price_dec, qty_step, min_qty, min_notional, price_tick):
    qty = qdown_dec(btc_qty, qty_step)
    price_rounded = qdown_dec(price_dec, price_tick)
    if not passes_filters(qty, price_rounded, min_qty, min_notional):
        return None, "SKIPPED: qty/notional below limits"
    resp_txt = sell_btc_limit(str_dec(qty), str_dec(price_rounded))
    return json.loads(resp_txt), resp_txt

def get_order(symbol, order_id):
    data = {"symbol": symbol, "orderId": order_id, "timestamp": int(time.time()*1000)}
    sig = get_binanceus_signature(data, api_sec)
    headers = {'X-MBX-APIKEY': api_key}
    data["signature"] = sig
    r = requests.get("https://api.binance.us/api/v3/order", headers=headers, params=data)
    return r.json()