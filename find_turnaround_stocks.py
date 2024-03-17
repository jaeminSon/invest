import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import ssl


def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data


def turning_around(data, long_window=200, short_window=20):

    new_data = {}
    new_data['close'] = data
    new_data['Upper'] = data.rolling(window=short_window).mean(
    ) + data.rolling(window=short_window).std()
    new_data['Lower'] = data.rolling(window=long_window).mean(
    ) - data.rolling(window=long_window).std()
    new_data = pd.DataFrame(new_data)
    new_data = new_data.dropna()

    data_short_term = new_data.iloc[-short_window:]
    rows_bullish = data_short_term[(
        data_short_term['close'] > data_short_term['Upper'])]
    bullish_enough = len(rows_bullish) > int(short_window * 0.1)

    data_long_term = new_data.iloc[-long_window:]
    rows_bearish = data_long_term[(
        data_long_term['close'] < data_long_term['Lower'])]
    bearish_enough = len(rows_bearish) > int(long_window * 0.5)

    return bullish_enough and bearish_enough


def get_period(history_days):
    now = datetime.now()
    start_date = now - timedelta(days=history_days)
    end_date = datetime.now().strftime('%Y-%m-%d')
    return start_date, end_date


def get_tickers(category):
    if category.lower() == "s&p500":
        ssl._create_default_https_context = ssl._create_unverified_context
        tickers = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        return tickers.Symbol.to_list()


def get_prev_stocks(path_txt="./watchlist.txt"):
    with open(path_txt) as f:
        lines = f.readlines()

    prev_stocks = []
    for line in lines:
        for ticker in line.split(":")[1].split(","):
            prev_stocks.append(ticker.replace(" ", ""))

    return prev_stocks


if __name__ == "__main__":

    history_days = 720

    start_date, end_date = get_period(history_days=history_days)
    tickers = get_tickers(category="s&p500")
    sp500_symbols = yf.download(
        tickers, start=start_date, end=end_date, interval="1d", auto_adjust=True)["Close"]

    turnaround_stocks = []
    for symbol in sp500_symbols:
        if not sp500_symbols[symbol].empty:
            if turning_around(sp500_symbols[symbol]):
                turnaround_stocks.append(symbol)

    prev_stocks = get_prev_stocks()
    new_turnaround_stocks = set(turnaround_stocks) - set(prev_stocks)

    new_turnaround_stocks_str = ", ".join(new_turnaround_stocks)
    print(
        f"======= New Turnaround Stocks=======: {new_turnaround_stocks_str}")
