from typing import List, Tuple

from .yahoo_finance import download_by_group
from .portfolio import win_rate


def sort_by_p_ratio(
    group: str,
    start_date: str,
    end_date: str = None,
    window: int = 100,
) -> List[Tuple[str, float]]:
    df = download_by_group(group, start_date, end_date)
    ticker_winrate = [
        (ticker, win_rate(df["Close"][ticker], window))
        for ticker in df["Close"].columns
        if len(df["Close"][ticker].dropna()) > window
    ]
    return sorted(ticker_winrate, key=lambda el: el[1], reverse=True)


def find_undervalued(
    group: str,
    winrate_threshold: float,
    start_date: str,
    end_date: str = None,
    window: int = 100,
):
    sorted_ticker_winrate = sort_by_p_ratio(group, start_date, end_date, window)
    return [(t, w) for t, w in sorted_ticker_winrate if w >= winrate_threshold]
