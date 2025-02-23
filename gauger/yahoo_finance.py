import os
import json
import ssl
from typing import Optional, Tuple, List, Dict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

DIR_TAGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tickers")


def get_ticker(group: str) -> List:
    name = group.lower()
    if name == "s&p500":
        ssl._create_default_https_context = ssl._create_unverified_context
        tickers = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )[0]
        return tickers.Symbol.to_list()
    else:
        return [i[0] for i in json.load(open(os.path.join(DIR_TAGS, f"{name}.json")))]


def ticker2name(category: str) -> Dict:
    name = category.lower()
    return {i[0]: i[1] for i in json.load(open(os.path.join(DIR_TAGS, f"{name}.json")))}


def download_by_group(
    group: str, start_date: str, end_date: Optional[str] = None
) -> pd.DataFrame:
    if end_date is None:
        start_date, end_date = period(start_date)
    tickers = get_ticker(group)
    return download(tickers, start_date, end_date)


def download(symbol, start_date, end_date) -> pd.DataFrame:
    stock_data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)

    if isinstance(symbol, list) and len(symbol) == 1:
        stock_data.columns = pd.MultiIndex.from_tuples(
            [(c, symbol[0]) for c in stock_data.columns]
        )

    return stock_data


def period(start_date: Optional[str] = None, date_back: int = None) -> Tuple[str, str]:
    assert not (
        start_date and date_back
    ), "Both start_date and date_back cannot be set at the same time."
    if date_back:
        now = datetime.now()
        start_date = (now - timedelta(days=date_back)).strftime("%Y-%m-%d")
        end_date = now.strftime("%Y-%m-%d")
        return start_date, end_date
    elif start_date:
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
        except:
            raise ValueError(
                f"Failed to process {start_date}. start_date should be YYYY-MM-DD format."
            )
        end_date = datetime.now().strftime("%Y-%m-%d")
        return start_date, end_date


def percent_change(yf_df: pd.DataFrame, key="Close", periods=1) -> pd.DataFrame:
    return (
        yf_df[key]
        .rolling(window=periods)
        .mean()
        .dropna()
        .pct_change(periods=periods)
        .fillna(0)
        .dropna(axis=1, how="all")
        .dropna(axis=0)
    )


def yf_return(yf_df: pd.DataFrame, key="Close") -> pd.DataFrame:
    pct_ch = percent_change(yf_df, key)
    return (pct_ch + 1).cumprod()


def expected_return_from_pct_ch(pct_ch: pd.DataFrame) -> np.ndarray:
    return np.array(pct_ch).mean(axis=0)
