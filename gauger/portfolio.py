import numpy as np
import pandas as pd
from typing import List

import scipy

from .yahoo_finance import download, get_ticker, period


def kelly(win_rate: float, net_profit: float, net_loss: float) -> float:
    assert 0 <= win_rate <= 1
    assert net_loss > 0

    if net_profit < 0:
        return 0

    bet = (1.0 * win_rate / (net_loss + 1e-6)) - (
        1.0 * (1 - win_rate) / (net_profit + 1e-6)
    )

    return np.clip(bet, 0, 1)


def kelly_cube(n_pts_per_axis: int = 101):
    linspace = np.linspace(0, 1.0, num=n_pts_per_axis)

    # cube[p_win][profit][loss]
    cube = np.zeros((n_pts_per_axis, n_pts_per_axis, n_pts_per_axis))
    for i in range(len(linspace)):
        for j in range(len(linspace)):
            for k in range(len(linspace)):
                ratio = (
                    kelly(linspace[i], linspace[j], linspace[k])
                    if linspace[k] > 1e-6
                    else 0
                )
                cube[i][j][k] = ratio if ratio > 0 else 0

    return cube


def cov_mat_from_pct_ch(pct_ch: pd.DataFrame) -> np.ndarray:
    return np.array(pct_ch.cov())


def compute_budge(total_budget: int):
    """
    Args:
        total_budget: dollars
    Return:
        list of money for each weight
    """
    pf = read_portfolio(path_portfolio)
    return {e[0]: e[1] * total_budget for e in pf["weights"]}


def price_ratio(price: pd.Series, window: int) -> pd.Series:
    return (price / price.rolling(window=window).mean()).dropna()


def win_rate(price: pd.Series, window: int) -> float:
    p_ratio = price_ratio(price, window)

    p = density_function(list(p_ratio))

    return integral(p, p_ratio.iloc[-1])


def win_rates_by_group(
    group: str,
    start_date: str,
    end_date: str = None,
    key: str = "Close",
):
    if end_date is None:
        start_date, end_date = period(start_date)

    tickers = get_ticker(group)
    df = download(tickers, start_date, end_date)

    win_rates = {}
    for ticker in tickers:
        window2win_rate = {}
        for window in [20, 50, 100, 200]:
            df_mean = df[key][ticker].to_frame("price_ratio")
            window2win_rate[window] = win_rate(df_mean["price_ratio"], window)
        win_rates[ticker] = window2win_rate

    return win_rates


def bet_ratios_by_group(
    group: str,
    start_date: str,
    end_date: str = None,
    key: str = "Close",
    profit: float = 0.1,
    loss: float = 0.5,
):
    win_rates = win_rates_by_group(group, start_date, end_date, key)

    bet_ratios = {}
    for ticker in win_rates.keys():
        window2tuple = {}
        for window in [20, 50, 100, 200]:
            p_win = win_rates[ticker][window]

            bet_ratio = kelly(
                p_win,
                profit,
                loss,
            )
            if bet_ratio > 1e-5:
                window2tuple[window] = {
                    "p_win": np.round(p_win, 2),
                    "bet": np.round(bet_ratio, 2),
                }

        bet_ratios[ticker] = window2tuple

    return bet_ratios


def density_function(x: List[float]):
    return scipy.stats.gaussian_kde(x)


def integral(p: callable, start: float, end=2, n_samples=1000):
    x = np.linspace(start, end, n_samples)
    return np.sum([(x[i] - x[i - 1]) * p(x[i - 1]) for i in range(1, len(x))])


def warnings():
    list_warning = [
        "Never trade at premarket or postmarket",
        "Bet size should be 1/64, 1/32, ..., 1/2",
    ]
    return "".join([f"({i+1}) {c}.\n" for i, c in enumerate(list_warning)])
