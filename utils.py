from typing import Tuple, List, Optional
from datetime import datetime, timedelta
import ssl
import json

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


#####################
### yahoo finance ###
#####################
def download(symbol, start_date, end_date) -> pd.DataFrame:
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data


def period(
    start_date: Optional[str] = None, history_days: Optional[str] = None
) -> Tuple[str, str]:
    assert not (
        start_date and history_days
    ), "Both start_date and history_days cannot be set at the same time."
    if history_days:
        now = datetime.now()
        start_date = (now - timedelta(days=history_days)).strftime("%Y-%m-%d")
        end_date = now.strftime("%Y-%m-%d")
        return start_date, end_date
    elif start_date:
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
        except:
            raise ValueError(f"Failed to process {start_date}")
        end_date = datetime.now().strftime("%Y-%m-%d")
        return start_date, end_date


def ticker(category: str) -> List:
    if category.lower() == "s&p500":
        ssl._create_default_https_context = ssl._create_unverified_context
        tickers = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )[0]
        return tickers.Symbol.to_list()
    elif category.lower() == "sectors":
        return [i[0] for i in json.load(open("sectors.json"))]
    elif category.lower() == "assets":
        return [i[0] for i in json.load(open("assets.json"))]
    else:
        raise ValueError("Unknown category.")


def name(category: str) -> List:
    if category.lower() == "sectors":
        return [i[1] for i in json.load(open("sectors.json"))]
    elif category.lower() == "assets":
        return [i[1] for i in json.load(open("assets.json"))]
    else:
        raise ValueError("Unknown category.")


#################
### portfolio ###
#################
def kelly(win_rate: float, net_profit: float, net_loss: float) -> float:
    assert 0 <= win_rate <= 1
    return (1.0 * win_rate / (net_loss + 1e-6)) - (
        1.0 * (1 - win_rate) / (net_profit + 1e-6)
    )


def draw_kelly_2d(loss: int = 1, path_savefile="kelly_criterion.png") -> None:
    """
    Usage:
    =======================================================
    for i in [0.33, 0.66, 1]:
        draw_kelly_2d(i, f"kelly_criterion_loss_{i}.png")
    =======================================================
    """
    import matplotlib.pyplot as plt
    import numpy as np

    win_rates = np.linspace(0, 1.0, num=101)
    profits = np.linspace(0, 1.0, num=101)

    data = [[0] * len(profits) for _ in range(len(win_rates))]
    for i in range(len(win_rates)):
        for j in range(len(profits)):
            ratio = kelly(win_rates[i], profits[j], loss)
            data[i][j] = ratio if ratio > 0 else 0

    plt.close()
    plt.imshow(np.array(data), cmap="viridis", interpolation="nearest")
    plt.colorbar()
    plt.title("Kelly criterion")
    plt.xlabel("Net profit")
    plt.ylabel("Win rate")
    plt.xticks(
        range(0, len(profits), len(profits) // 5),
        [profits[i] for i in range(0, len(profits), len(profits) // 5)],
    )
    plt.yticks(
        range(0, len(win_rates), len(win_rates) // 5),
        [win_rates[i] for i in range(0, len(win_rates), len(win_rates) // 5)],
    )
    plt.savefig(path_savefile)


############
### plot ###
############
def plot_return(start_date: str, path_savefile: str = "return.png") -> None:
    """
    Usage:
    =====================================
    plot_return(start_date="2024-01-01")
    =====================================
    """
    start_date, end_date = period(start_date)

    tickers = ticker("sectors")
    names = name("sectors")
    ticker2name = {t: " ".join(n.split()[2:]) for t, n in zip(tickers, names)}

    df = download(tickers, start_date, end_date)
    df.columns = pd.MultiIndex.from_tuples(
        [(c1, ticker2name[c2]) for c1, c2 in df.columns]
    )

    plt.close()
    pct_ch = df["Close"].pct_change()
    (pct_ch + 1).cumprod().plot(figsize=(16, 12))

    plt.savefig(path_savefile)
