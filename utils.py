from typing import Tuple, List, Dict, Iterable, Optional
from datetime import datetime, timedelta
import os
import ssl
import shutil
import json

import numpy as np
import scipy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


#####################
### yahoo finance ###
#####################
def download_SandP(start_date, end_date: Optional[str] = None) -> pd.DataFrame:
    if end_date is None:
        start_date, end_date = period(start_date)
    tickers = get_ticker("s&p500")
    return download(tickers, start_date, end_date)


def download_assets(start_date, end_date: Optional[str] = None) -> pd.DataFrame:
    if end_date is None:
        start_date, end_date = period(start_date)
    tickers = get_ticker("assets")
    return download(tickers, start_date, end_date)


def download_sectors(start_date, end_date: Optional[str] = None) -> pd.DataFrame:
    if end_date is None:
        start_date, end_date = period(start_date)
    tickers = get_ticker("sectors")
    return download(tickers, start_date, end_date)


def download_portfolio(start_date, end_date: Optional[str] = None) -> pd.DataFrame:
    pf = read_portfolio()
    tickers = [e[0] for e in pf["weights"] if e[0] != "cash"]

    if end_date is None:
        start_date, end_date = period(start_date)

    df = download(tickers, start_date, end_date)

    return df


def download(symbol, start_date, end_date) -> pd.DataFrame:
    stock_data = yf.download(symbol, start=start_date, end=end_date)
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
            raise ValueError(f"Failed to process {start_date}")
        end_date = datetime.now().strftime("%Y-%m-%d")
        return start_date, end_date


def benchmark_return(benchmark: str, start_date, end_date) -> pd.Series:
    try:
        benchmark_ticker = find_ticker(benchmark)
    except:
        benchmark_ticker = benchmark

    df_benchmark = download([benchmark_ticker], start_date, end_date)
    return (df_benchmark["Close"].pct_change().fillna(0) + 1).cumprod().dropna()


def portfolio_return(test_date) -> pd.Series:
    portfolio = read_portfolio()
    portfolio_return = portfolio["cash"] + sum(e[1] for e in portfolio["weights"])
    return pd.Series([portfolio_return], index=[test_date], name="Close")


###########
### I/O ###
###########
def get_ticker(category: str) -> List:
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


def ticker2name(category: str) -> Dict:
    if category.lower() == "sectors":
        return {i[0]: i[1] for i in json.load(open("sectors.json"))}
    elif category.lower() == "assets":
        return {i[0]: i[1] for i in json.load(open("assets.json"))}
    else:
        raise ValueError("Unknown category.")


def portfolio_start_date() -> Dict:
    return json.load(open("portfolio.json"))["date_generated"]


def find_ticker(keyword):
    if keyword == "s&p":
        return "SPY"
    elif keyword == "nasdaq":
        return "QQQ"
    elif keyword == "dow":
        return "DIA"
    elif keyword == "gold":
        return "GLD"
    else:
        raise ValueError(f"Unknown keyword {keyword}")


def write_portfolio(
    date_generated,
    tickers: List[str],
    weight: np.ndarray,
    cash_amount: float,
    path_savefile: str = "portfolio.json",
) -> None:
    assert len(tickers) == len(
        weight
    ), "tickers and weight should have the same length."

    t2n = ticker2name("assets")

    data = {
        "date_generated": date_generated
        if isinstance(date_generated, str)
        else date_generated.strftime("%Y-%m-%d"),
        "cash": cash_amount,
        "weights": [(t, w, t2n[t]) for t, w in zip(tickers, weight)],
    }

    json.dump(data, open(path_savefile, "w"), indent=4)


def read_portfolio(path_portfolio: str = "portfolio.json"):
    return json.load(open(path_portfolio))


def update_tickers(stocks: Iterable[str]) -> None:
    alread_included = [e.rstrip() for e in open("./watchlist.txt").readlines()]

    with open("./watchlist.txt", "a") as f:
        for stock in stocks:
            if stock not in alread_included:
                f.write(f"{stock}\n")


def write_new_portfolio(
    rebalacing_period,
    path_savefile: str = "portfolio.json",
    dir_prev_portfolio="prev_portfolio",
):
    start_date, end_date = period(date_back=rebalacing_period)

    if os.path.exists(path_savefile):
        os.makedirs(dir_prev_portfolio, exist_ok=True)
        shutil.copy(
            path_savefile,
            os.path.join(
                dir_prev_portfolio, f"portfolio_{start_date}_to_{end_date}.json"
            ),
        )
    else:
        create_portfolio_file(end_date, path_savefile)

    generate_portfolio_via_rebalancing(
        end_date=datetime.strptime(end_date, "%Y-%m-%d"),
        path_savefile=path_savefile,
    )


#######################
### stock selection ###
#######################
def select_stock(
    yf_df: pd.DataFrame, window: int, top_k: int, key="Close"
) -> List[str]:
    pct_ch = (
        yf_df[key][yf_df[key].columns]
        .rolling(window=window)
        .mean()
        .pct_change(periods=window)
        .dropna(axis=1, how="all")
    )

    # sharpe is a series that maps ticker to Sharpe value
    sharpe = ((pct_ch.iloc[-1] - pct_ch.mean()) / np.sqrt(pct_ch.var())).dropna()
    sharpe.sort_values(ascending=False, inplace=True)

    return list(sharpe.index[:top_k])


def select_stock_varying_windows(yf_df: pd.DataFrame, top_k: int) -> List[str]:
    windows = [3, 5, 10, 20]
    return {w: select_stock(yf_df, w, top_k) for w in windows}


#################
### portfolio ###
#################
def kelly(win_rate: float, net_profit: float, net_loss: float) -> float:
    assert 0 <= win_rate <= 1
    assert net_loss > 0

    if net_profit < 0:
        return 0

    bet = (1.0 * win_rate / (net_loss + 1e-6)) - (
        1.0 * (1 - win_rate) / (net_profit + 1e-6)
    )

    return np.clip(bet, 0, 1)


def all_cash(portfolio: Dict):
    return all(e[1] < 1e-6 for e in portfolio["weights"])


def sell_signal(end_date, yf_df: pd.DataFrame, key="Close"):
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    if yf_df is None:
        df = download_assets(start_date=asset_open_date, end_date=end_date)
    else:
        df = yf_df[
            (yf_df.index >= end_date - timedelta(days=200)) & (yf_df.index <= end_date)
        ]

    df_ma100 = df[key].rolling(window=100).mean().dropna()
    for ticker in df[key].columns:
        if (
            sum(df_ma100[ticker].diff(periods=1)[-5:] <= 0) >= 4
            and sum(df_ma100[ticker].diff(periods=1).diff(periods=1)[-5:] <= 0) >= 4
        ):
            return True

    return False


def buy_signal(end_date, yf_df: pd.DataFrame, key="Close"):
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    if yf_df is None:
        df = download_assets(start_date=asset_open_date, end_date=end_date)
    else:
        df = yf_df[
            (yf_df.index >= end_date - timedelta(days=365)) & (yf_df.index <= end_date)
        ]

    df_ma100 = df[key].rolling(window=100).mean().dropna()
    for ticker in df[key].columns:
        if (
            sum(df_ma100[ticker].diff(periods=1)[-5:] >= 0) >= 3
            and sum(df_ma100[ticker].diff(periods=1).diff(periods=1)[-5:] >= 0) >= 3
        ):
            return True

    return False


def sell(portfolio: Dict, transaction_fee_rate) -> Dict:
    tickers = get_ticker("assets")
    optimal_w = [0] * len(tickers)

    cash_amount = portfolio["cash"]
    t2w = {e[0]: e[1] for e in portfolio["weights"]}
    total = sum(t2w.values()) + cash_amount

    cash = cash_amount + (total - cash_amount) / (1 + transaction_fee_rate)

    return tickers, optimal_w, cash


def compute_win_rate(yf_df: pd.DataFrame, ticker: str) -> float:
    # empirical wining rate of the previous period
    list_prob_win = []
    for window in [10, 20, 50, 100]:
        pct_ch = percent_change(yf_df, periods=window, fillna_zero=False)
        pct_ch_sorted = sorted(np.array(pct_ch[ticker]))
        prob_win = np.sum(pct_ch_sorted <= pct_ch[ticker].iloc[-1]) / len(pct_ch_sorted)
        list_prob_win.append(prob_win)
    return np.mean(list_prob_win)


def compute_confidence(yf_df: pd.DataFrame, ticker: str, key="Close"):
    list_confidence = []
    for window in [10, 20, 50, 100]:
        df_ma = yf_df[key].rolling(window=window).mean().dropna()
        confidence = sum(
            np.array(df_ma[ticker].diff(periods=1)[-window:] >= 0)
            * linear_weight(window)
        )
        list_confidence.append(confidence)
    return np.mean(list_confidence)


def linear_weight(length):
    return np.arange(length) / sum(np.arange(length))


def set_target_weights(
    end_date,
    asset_open_date="2010-04-01",
    key="Close",
    yf_df: pd.DataFrame = None,
) -> Dict:
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    if yf_df is None:
        df = download_assets(start_date=asset_open_date, end_date=end_date)
    else:
        df = yf_df[(yf_df.index >= asset_open_date) & (yf_df.index <= end_date)]

    weights = {}
    for ticker in df[key].columns:
        prob_win = compute_win_rate(df, ticker)

        # reversed optimal weight == 2*(1-prob_win)
        bet_ratio = 1 - kelly(
            win_rate=prob_win,
            net_profit=1,
            net_loss=1,
        )

        confidence = compute_confidence(df, ticker)

        weights[ticker] = bet_ratio * confidence

    sum_weights_assets = sum(weights.values())
    if sum_weights_assets > 1:
        weights = {t: 1.0 * w / sum_weights_assets for t, w in weights.items()}
        weights["cash"] = 0
    else:
        weights["cash"] = 1 - sum_weights_assets

    return weights


def generate_initial_portfolio_backtest(end_date, yf_df: pd.DataFrame = None) -> None:
    target_weights = set_target_weights(end_date, yf_df=yf_df)
    tickers = [t for t in target_weights.keys() if t != "cash"]
    write_portfolio(
        end_date,
        tickers,
        [target_weights[t] for t in tickers],
        target_weights["cash"],
        path_savefile="portfolio.json",
    )


def create_portfolio_file(date, path_savefile="portfolio.json"):
    tickers = get_ticker("assets")
    write_portfolio(
        date,
        tickers,
        weight=[0 for t in tickers],
        cash_amount=1,
        path_savefile=path_savefile,
    )


def generate_portfolio_via_rebalancing(
    end_date,
    yf_df: pd.DataFrame = None,
    transaction_fee_rate: float = 0.005,
    path_savefile: str = "portfolio.json",
) -> None:
    portfolio = read_portfolio(path_savefile)

    target_weights = set_target_weights(end_date, yf_df=yf_df)

    (
        tickers,
        optimal_w,
        new_cash_amount,
    ) = optimize_asset_portfolio_via_rebalancing(
        portfolio, target_weights, transaction_fee_rate
    )

    write_portfolio(end_date, tickers, optimal_w, new_cash_amount, path_savefile)


def optimize_asset_portfolio_via_rebalancing(
    portfolio: Dict,
    target_weights: Iterable[float],
    transaction_fee_rate: float,
) -> Tuple:
    assert abs(sum(target_weights.values()) - 1) < 1e-6

    cash_amount = portfolio["cash"]
    t2w = {e[0]: e[1] for e in portfolio["weights"]}
    total = sum(t2w.values()) + cash_amount

    optimal_w = []
    for ticker in t2w.keys():
        new_weight = target_weights[ticker] * total
        if new_weight > t2w[ticker]:
            new_weight = t2w[ticker] + (new_weight - t2w[ticker]) / (
                1 + transaction_fee_rate
            )  # compensate for transaction fee by reducing weight
        optimal_w.append(new_weight)

    new_cash_amount = total * target_weights["cash"]
    if new_cash_amount > cash_amount:
        new_cash_amount = cash_amount + (new_cash_amount - cash_amount) / (
            1 + transaction_fee_rate
        )

    return t2w.keys(), optimal_w, new_cash_amount


def cov_mat_from_pct_ch(pct_ch: pd.DataFrame) -> np.ndarray:
    return np.array(pct_ch.cov())


def compute_return_volatility(
    pct_ch: pd.DataFrame,
) -> Tuple[List, np.ndarray, np.ndarray]:
    tickers = list(pct_ch.columns)
    r = expected_return_from_pct_ch(pct_ch)
    cov = cov_mat_from_pct_ch(pct_ch)

    # check the order of columns
    assert all((pct_ch[t].mean() - r[i]) < 1e-6 for i, t in enumerate(tickers))
    assert all((pct_ch[t].var() - cov[i, i]) < 1e-6 for i, t in enumerate(tickers))

    return tickers, r, cov


def random_portfolio(
    ret: np.ndarray, cov_mat: np.ndarray, N=10_000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(0)

    weights = np.random.random((N, len(ret)))
    weights /= np.sum(weights, axis=1, keepdims=True)

    returns = np.array([w.T @ ret for w in weights])

    volatilities = np.sqrt([w.T @ cov_mat @ w for w in weights])

    return weights, returns, volatilities


def optimize_asset_portfolio_via_sampling(
    df: pd.DataFrame, key="Close", return_samples: bool = False
) -> Tuple:
    pct_ch = percent_change(df, key)
    pct_ch_filted = pct_ch
    # pct_ch_filted = filter_assets(pct_ch)

    tickers, r, cov = compute_return_volatility(pct_ch_filted)

    random_w, random_r, random_v = random_portfolio(r, cov)

    random_sharpe = random_r / random_v
    index_opt = np.argmax(random_sharpe)
    optimal_w = random_w[index_opt]
    optimal_v = random_v[index_opt]
    optimal_r = random_r[index_opt]
    assert abs(optimal_w.T @ r - optimal_r) < 1e-6
    assert abs(np.sqrt(optimal_w.T @ cov @ optimal_w) - optimal_v) < 1e-6

    if return_samples:
        return (
            tickers,
            optimal_w,
            optimal_r,
            optimal_v,
            random_r,
            random_v,
            random_sharpe,
        )
    else:
        return tickers, optimal_w, optimal_r, optimal_v


def compute_budge(total_budget: int, path_portfolio: str = "portfolio.json"):
    """
    Args:
        total_budget: dollars
    Return:
        list of money for each weight
    """
    pf = read_portfolio(path_portfolio)
    return [e + [e[1] * total_budget] for e in pf["weights"]]


def need_to_update_portfolio(date: datetime.date, rebalacing_period: int):
    portfolio = read_portfolio()
    stale = (
        date - datetime.strptime(portfolio["date_generated"], "%Y-%m-%d")
    ).days >= rebalacing_period
    return stale or all_cash(portfolio)


###############
### Utility ###
###############
def percent_change(
    yf_df: pd.DataFrame, key="Close", periods=1, fillna_zero=True
) -> pd.DataFrame:
    if fillna_zero:
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
    else:
        return (
            yf_df[key]
            .rolling(window=periods)
            .mean()
            .dropna()
            .pct_change(periods=periods)
            .dropna(axis=1, how="all")
            .dropna(axis=0)
        )


def yf_return(yf_df: pd.DataFrame, key="Close") -> pd.DataFrame:
    pct_ch = percent_change(yf_df, key)
    return (pct_ch + 1).cumprod()


def expected_return(yf_df: pd.DataFrame, key="Close") -> np.ndarray:
    pct_ch = percent_change(yf_df, key)
    return expected_return_from_pct_ch(pct_ch)


def expected_return_from_pct_ch(pct_ch: pd.DataFrame) -> np.ndarray:
    return np.array(pct_ch).mean(axis=0)


def simulate_market(yf_df: pd.DataFrame, eval_date, key="Close"):
    """Update portfolio based on market results on eval_date"""
    dday = yf_df.iloc[np.where(yf_df.index == eval_date)[0][0]]
    prev_day = yf_df.iloc[np.where(yf_df.index == eval_date)[0][0] - 1]

    portfolio = read_portfolio()
    t2w = {e[0]: e[1] for e in portfolio["weights"]}
    for ticker in t2w.keys():
        t2w[ticker] *= dday[key][ticker] / prev_day[key][ticker]

    write_portfolio(
        portfolio["date_generated"],
        list(t2w.keys()),
        list(t2w.values()),
        portfolio["cash"],
    )


############
### plot ###
############
def plot_kelly_2d(path_savefile="kelly_criterion.png") -> None:
    win_rates = np.linspace(0, 1.0, num=101)
    profits = np.linspace(0, 1.0, num=101)

    losses = np.linspace(0.1, 1.0, num=10)
    all_data = []
    for loss in losses:
        data = [[0] * len(profits) for _ in range(len(win_rates))]
        for i in range(len(win_rates)):
            for j in range(len(profits)):
                ratio = kelly(win_rates[i], profits[j], loss)
                data[i][j] = ratio if ratio > 0 else 0
        all_data.append(data)

    plt.close()
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(2):
        for j in range(5):
            draw = axes[i, j].imshow(
                np.array(all_data[5 * i + j]),
                cmap="viridis",
                interpolation="nearest",
                vmin=0,
                vmax=1,
            )
            axes[i, j].set_title(f"Loss {losses[5 * i + j]:.1}")
            axes[i, j].set_xlabel("Net profit")
            if j == 0:
                axes[i, j].set_ylabel("Win rate")
            axes[i, j].set_xticks(
                range(0, len(profits), len(profits) // 5),
                [profits[i] for i in range(0, len(profits), len(profits) // 5)],
            )
            axes[i, j].set_yticks(
                range(0, len(win_rates), len(win_rates) // 5),
                [win_rates[i] for i in range(0, len(win_rates), len(win_rates) // 5)],
            )

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(draw, cax=cbar_ax)

    plt.savefig(path_savefile)


def plot_return(df: pd.DataFrame, path_savefile: str) -> None:
    df_return = yf_return(df)
    plt.close()
    df_return.plot(figsize=(16, 12))

    plt.savefig(path_savefile)


def plot_return_leverage_with_ma(
    start_date, path_savefile: str = "return_leverage_with_ma.png"
) -> None:
    tickers = ["SPXL", "TQQQ", "SOXL"]
    start_date, end_date = period(start_date)
    df = download(tickers, start_date, end_date)

    plt.close()
    plt.figure(figsize=(20, 15))
    df_return = yf_return(df)
    # df_ma5 = df["Close"].rolling(window=5).mean().dropna()
    # df_ma5 /= df_ma5.iloc[0]
    # df_ma10 = df["Close"].rolling(window=10).mean().dropna()
    # df_ma10 /= df_ma10.iloc[0]
    df_ma20 = df["Close"].rolling(window=20).mean().dropna()
    df_ma20 /= df_ma20.iloc[0]
    df_ma50 = df["Close"].rolling(window=50).mean().dropna()
    df_ma50 /= df_ma50.iloc[0]
    df_ma100 = df["Close"].rolling(window=100).mean().dropna()
    df_ma100 /= df_ma100.iloc[0]
    df_ma200 = df["Close"].rolling(window=200).mean().dropna()
    df_ma200 /= df_ma200.iloc[0]

    for ticker in tickers:
        plt.plot(df_return[ticker].index, list(df_return[ticker]), label=ticker)
        # plt.plot(df_ma5[ticker].index, list(df_ma5[ticker]), label=ticker + "_ma5")
        # plt.plot(df_ma10[ticker].index, list(df_ma10[ticker]), label=ticker + "_ma10")
        # plt.plot(df_ma20[ticker].index, list(df_ma20[ticker]), label=ticker + "_ma20")
        # plt.plot(df_ma50[ticker].index, list(df_ma50[ticker]), label=ticker + "_ma50")
        plt.plot(
            df_ma100[ticker].index, list(df_ma100[ticker]), label=ticker + "_ma100"
        )
        # plt.plot(df_ma200[ticker].index, list(df_ma200[ticker]), label=ticker + "_ma200")

        peaks = scipy.signal.find_peaks(df_ma100[ticker], width=1, rel_height=0.01)[0]
        bottomes = scipy.signal.find_peaks(-df_ma100[ticker], width=1, rel_height=0.01)[
            0
        ]
        plt.scatter(
            [df_ma100[ticker].index[p] for p in peaks],
            [df_ma100[ticker].iloc[p] for p in peaks],
            color="r",
            marker="v",
            s=30,
        )
        plt.scatter(
            [df_ma100[ticker].index[p] for p in bottomes],
            [df_ma100[ticker].iloc[p] for p in bottomes],
            color="b",
            marker="v",
            s=30,
        )

    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()

    plt.savefig(path_savefile)


def plot_return_leverage(
    start_date, path_savefile: str = "return_leverage.png"
) -> None:
    tickers = ["SPY", "SPXL", "TQQQ", "QQQ", "SOXX", "SOXL"]
    start_date, end_date = period(start_date)
    df = download(tickers, start_date, end_date)

    plot_return(df, path_savefile)


def plot_return_by_sector(start_date, path_savefile: str = "return_sector.png") -> None:
    df = download_sectors(start_date)
    t2n = {t: " ".join(n.split()[2:]) for t, n in ticker2name("sectors").items()}
    df.columns = pd.MultiIndex.from_tuples([(c1, t2n[c2]) for c1, c2 in df.columns])

    plot_return(df, path_savefile)


def plot_return_by_asset(start_date, path_savefile: str = "return_asset.png") -> None:
    df = download_assets(start_date)

    pct_ch = df["Close"].pct_change(periods=1).dropna(axis=1, how="all")
    df_return = (pct_ch + 1).cumprod()
    df_return = df_return[
        [col for col in df_return.columns if df_return[col].iloc[-1] > 4]
    ]
    t2n = {t: n for t, n in ticker2name("assets").items()}
    df_return.columns = [t2n[c] for c in df_return.columns]

    plt.close()
    df_return.plot(figsize=(16, 12))
    plt.savefig(path_savefile)


def plot_return_index(start_date, path_savefile: str = "return_index.png") -> None:
    start_date, end_date = period(start_date)
    tickers = [
        "SPY",
        "DIA",
        "QQQ",
        "IWM",
        "GLD",
        "IBIT",
        "IEF",
        "TLT",
        "USO",
        "DBA",
        "DBB",
    ]
    names = [
        "S&P",
        "Dow",
        "Nasdaq",
        "Russell",
        "Gold",
        "Bitcoin",
        "Bond (7-10Y)",
        "Bond (20+Y)",
        "Oil",
        "Agriculutre",
        "Base Metals",
    ]
    t2n = {t: n for t, n in zip(tickers, names)}

    df = download(tickers, start_date, end_date)
    df.columns = pd.MultiIndex.from_tuples([(c1, t2n[c2]) for c1, c2 in df.columns])

    plot_return(df, path_savefile)


def plot_return_portfolio_stocks(
    start_date, path_savefile: str = "return_portfolio_stocks.png"
) -> None:
    pf = read_portfolio()
    t2n = {e[0]: e[2] for e in pf["weights"]}

    df = download_portfolio(start_date)

    df.columns = pd.MultiIndex.from_tuples([(c1, t2n[c2]) for c1, c2 in df.columns])

    plot_return(df, path_savefile)


def plot_matrix(start_date, target: str, matrix_data: str):
    start_date, end_date = period(start_date)

    if target == "assets":
        tickers = get_ticker("assets")
        names = name("assets")
        t2n = ticker2name("assets")
    elif target == "sectors":
        tickers = get_ticker("sectors")
        names = name("sectors")
        t2n = {t: " ".join(n.split()[2:]) for t, n in ticker2name("sectors").items()}
    else:
        raise ValueError(f"Unknown target {target}.")

    df = download(tickers, start_date, end_date)
    df.columns = pd.MultiIndex.from_tuples([(c1, t2n[c2]) for c1, c2 in df.columns])

    if matrix_data == "correlation":
        data = df["Close"].corr()
    else:
        raise ValueError(f"Unknown matrix_data {matrix_data}")

    plt.close()
    # assets have many elements that figsize should be big
    sns.heatmap(data, cmap="seismic", annot=True)
    plt.savefig(f"{matrix_data}_{target}.png", bbox_inches="tight")


def plot_correlation(start_date, target: str = "assets"):
    plot_matrix(start_date, target, "correlation")


def plot_covariance(start_date, target: str = "assets"):
    plot_matrix(start_date, target, "cov")


def plot_return_measure(start_date, path_savefile: str = "return_measure.png"):
    df = download_sectors(start_date)

    x = np.array(percent_change(df))
    log_x_1 = np.log(x + 1)

    plt.close()
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    for i in range(2):
        for j in range(5):
            axes[2 * i, j].hist(
                x[:, 5 * i + j], bins=50, color="skyblue", edgecolor="black"
            )
            axes[2 * i + 1, j].hist(
                log_x_1[:, 5 * i + j], bins=50, color="lightblue", edgecolor="black"
            )
            axes[2 * i, j].set_title(f"x (sector {5 * i + j})")
            axes[2 * i + 1, j].set_title(f"log(x+1) (sector {5 * i + j})")

    plt.savefig(f"measure_estimate.png")


def plot_soaring_stocks(top_k=7):
    start_date, end_date = period(date_back=365)
    df = download_SandP(start_date)
    # df = download_sectors(start_date)
    window2stocks = select_stock_varying_windows(df, top_k)

    update_tickers(set(sum(window2stocks.values(), [])))

    for window in window2stocks:
        plt.close()

        pct_ch = df["Close"][window2stocks[window]].pct_change().fillna(0).dropna()
        df_return = (pct_ch + 1).cumprod()

        df_return.plot(figsize=(16, 12))

        plt.savefig(f"SandP_{window}_days_best_{top_k}.png")


def optimality(returns: np.ndarray, volatilities: np.ndarray) -> np.ndarray:
    covered = np.array([False] * len(returns))
    for i in range(len(returns)):
        if not covered[i]:
            covered[(returns <= returns[i]) & (volatilities >= volatilities[i])] = True
            covered[i] = False

    return covered


def plot_portfolio_via_sampling(
    start_date: str, key="Close", path_savefile: str = "portfolio_via_sampling.png"
) -> None:
    df = download_assets(start_date)

    (
        tickers,
        weight,
        optimal_r,
        optimal_v,
        random_r,
        random_v,
        random_sharpe,
    ) = optimize_asset_portfolio_via_sampling(df, key, return_samples=True)

    df_random = pd.DataFrame(
        {
            "return": random_r,
            "volatility": random_v,
            "Sharpe": random_sharpe,
        }
    )

    plt.close()
    sns.scatterplot(
        df_random, x="volatility", y="return", hue="Sharpe", s=3, legend=False
    )
    # draw frontiers
    covered = optimality(random_r, random_v)
    plt.scatter(
        random_v[~covered], random_r[~covered], color="orange", marker=".", s=10
    )
    # draw tangent point
    plt.scatter(
        [optimal_v],
        [optimal_r],
        color="r",
        marker="*",
        s=30,
        label="Tangent",
    )
    # plot Nasdaq, Dow, Gold
    plt.scatter(
        [np.sqrt(df[key]["TQQQ"].pct_change().var())],
        [df[key]["TQQQ"].pct_change().mean()],
        color="blue",
        marker="o",
        s=10,
        label="TQQQ",
    )
    plt.scatter(
        [np.sqrt(df[key]["SPXL"].pct_change().var())],
        [df[key]["SPXL"].pct_change().mean()],
        color="green",
        marker="o",
        s=10,
        label="SPXL",
    )
    plt.scatter(
        [np.sqrt(df[key]["SOXL"].pct_change().var())],
        [df[key]["SOXL"].pct_change().mean()],
        color="yellow",
        marker="o",
        s=10,
        label="SOXL",
    )

    plt.legend()
    plt.savefig(path_savefile, bbox_inches="tight")


def append_returns(prev_r: Optional[pd.Series], next_r: pd.Series) -> pd.Series:
    if prev_r is None:
        return next_r
    else:
        return pd.concat([prev_r, next_r])


def set_dates_backtest(current_date: datetime, rebalacing_period: int) -> Tuple:
    # optimize porfolio with data between [opt_s, opt_e] and test on [test_s, test_e]
    opt_s = current_date - timedelta(days=rebalacing_period)
    opt_e = current_date
    test_s = opt_e
    test_e = test_s + timedelta(days=rebalacing_period)
    return opt_s, opt_e, test_s, test_e


def get_benchmark_start_date(dict_benchmark):
    df_benchmark = pd.DataFrame(dict_benchmark)
    df_benchmark.dropna(inplace=True)
    return df_benchmark.index[0].to_pydatetime()


def get_benchmark_data_backtest(start_date, end_date):
    sandp = benchmark_return(
        "s&p",
        start_date,
        end_date,
    )

    df_benchmark = download(["SPXL", "TQQQ", "SOXL"], start_date, end_date)
    equal_rate = (
        (df_benchmark["Close"].pct_change().fillna(0) + 1)
        .cumprod()
        .mean(axis=1)
        .dropna()
    )
    spxl = (df_benchmark["Close"]["SPXL"].pct_change().fillna(0) + 1).cumprod().dropna()
    tqqq = (df_benchmark["Close"]["TQQQ"].pct_change().fillna(0) + 1).cumprod().dropna()
    soxl = (df_benchmark["Close"]["SOXL"].pct_change().fillna(0) + 1).cumprod().dropna()

    return {
        "s&p": sandp,
        "SPXL": spxl,
        "TQQQ": tqqq,
        "SOXL": soxl,
        "equal_rate": equal_rate,
    }


def plot_rebalancing_backtest(
    start_date,
    end_date,
    rebalacing_periods: List[int] = [10, 20, 50, 100, 200],
    asset_open_date="2010-04-01",
    path_savefile: str = "backtest.png",
) -> None:
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    dict_benchmark = get_benchmark_data_backtest(start_date, end_date)
    start_date = get_benchmark_start_date(dict_benchmark)

    yf_df = download_assets(start_date=asset_open_date, end_date=end_date)

    period2series = {}
    for rebalacing_period in rebalacing_periods:
        generate_initial_portfolio_backtest(start_date, yf_df=yf_df)

        portfolio_returns = pd.Series([1], index=[start_date], name="Close")

        test_date = start_date + timedelta(days=1)
        while test_date <= end_date:
            if sum(test_date == yf_df["Close"].index) > 0:
                if need_to_update_portfolio(test_date, rebalacing_period):
                    generate_portfolio_via_rebalancing(
                        end_date=test_date - timedelta(days=1),
                        yf_df=yf_df[yf_df.index <= test_date - timedelta(days=1)],
                    )

                simulate_market(yf_df, test_date)

                pf_r = portfolio_return(test_date)
                portfolio_returns = append_returns(portfolio_returns, pf_r)

            test_date += timedelta(days=1)

        period2series[rebalacing_period] = portfolio_returns

        os.system("rm portfolio.json")

    plt.close()
    data = {f"portfolio_update_{p}": s for p, s in period2series.items()}
    data.update(dict_benchmark)
    df_return = pd.DataFrame(data)
    df_return.plot(figsize=(16, 8))
    plt.savefig(path_savefile)
