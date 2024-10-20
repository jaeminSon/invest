from typing import Tuple, List, Dict, Iterable, Optional
from datetime import datetime, timedelta
import os
import ssl
import shutil
import json

import numpy as np
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
    tickers = ticker("s&p500")
    return download(tickers, start_date, end_date)


def download_assets(start_date, end_date: Optional[str] = None) -> pd.DataFrame:
    if end_date is None:
        start_date, end_date = period(start_date)
    tickers = ticker("assets")
    return download(tickers, start_date, end_date)


def download_sectors(start_date, end_date: Optional[str] = None) -> pd.DataFrame:
    if end_date is None:
        start_date, end_date = period(start_date)
    tickers = ticker("sectors")
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


# def portfolio_return(
#     start_date, end_date, has_predecessor: bool, yf_df: pd.DataFrame = None
# ) -> pd.Series:
# if has_predecessor:
#     start_date -= timedelta(days=10)

# if yf_df is None:
#     df = download_portfolio(start_date, end_date)
# else:
#     df = yf_df[(yf_df.index >= start_date) & (yf_df.index <= end_date)]

# portfolio = read_portfolio()
# portfolio_return = pd.Series(0, index=df["Close"].index, name="Close")
# for ticker, weight, name in portfolio["weights"]:
#     portfolio_return += (df["Close"][ticker] / df["Close"][ticker].iloc[0]).fillna(
#         1
#     ) * weight

# if "cash" in portfolio:
#     portfolio_return += portfolio["cash"]

# return portfolio_return


###########
### I/O ###
###########
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
    asset_open_date="2012-01-01",
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

    yf_df = download_assets(start_date=asset_open_date, end_date=end_date)

    generate_portfolio_via_rebalancing(
        end_date=datetime.strptime(end_date, "%Y-%m-%d"),
        rebalacing_period=rebalacing_period,
        yf_df=yf_df,
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
            (yf_df.index >= end_date - timedelta(days=50)) & (yf_df.index <= end_date)
        ]

    df_ma5 = df["Close"].iloc[-30:].rolling(window=5).mean().dropna()
    df_ma10 = df["Close"].iloc[-30:].rolling(window=10).mean().dropna()
    df_ma20 = df["Close"].iloc[-30:].rolling(window=20).mean().dropna()

    return any(
        df_ma5[ticker].diff(periods=1).iloc[-1] < 0
        and df_ma10[ticker].diff(periods=1).iloc[-1] < 0
        and df_ma20[ticker].diff(periods=1).iloc[-1] < 0
        for ticker in df[key].columns
    )


def buy_signal(end_date, yf_df: pd.DataFrame, key="Close"):
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    if yf_df is None:
        df = download_assets(start_date=asset_open_date, end_date=end_date)
    else:
        df = yf_df[
            (yf_df.index >= end_date - timedelta(days=50)) & (yf_df.index <= end_date)
        ]

    df_ma5 = df["Close"].iloc[-30:].rolling(window=5).mean().dropna()
    df_ma10 = df["Close"].iloc[-30:].rolling(window=10).mean().dropna()
    df_ma20 = df["Close"].iloc[-30:].rolling(window=20).mean().dropna()

    return any(
        df_ma5[ticker].diff(periods=1).iloc[-1] > 0
        and df_ma10[ticker].diff(periods=1).iloc[-1] > 0
        for ticker in df[key].columns
    )


def sell(portfolio: Dict, transaction_fee_rate, key="Close") -> Dict:
    tickers = ticker("assets")
    optimal_w = [0] * len(tickers)

    cash_amount = portfolio["cash"]
    t2w = {e[0]: e[1] for e in portfolio["weights"]}
    total = sum(t2w.values()) + cash_amount

    cash = cash_amount + (total - cash_amount) / (1 + transaction_fee_rate)

    return tickers, optimal_w, cash


def set_target_weights(
    start_date,
    end_date,
    asset_open_date="2012-01-01",
    key="Close",
    yf_df: pd.DataFrame = None,
) -> Dict:
    # weights = {t:1./len(ticker("assets")) for t in ticker("assets")}
    # weights["cash"] = 0
    # return weights
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    if yf_df is None:
        df = download_assets(start_date=asset_open_date, end_date=end_date)
    else:
        df = yf_df[(yf_df.index >= asset_open_date) & (yf_df.index <= end_date)]

    pct_ch = percent_change(df, periods=(end_date - start_date).days, fillna_zero=False)

    weights = {}
    for ticker in df[key].columns:
        # empirical wining rate of the previous period
        pct_ch_sorted = sorted(np.array(pct_ch[ticker]))
        prob_win = np.sum(pct_ch_sorted <= pct_ch[ticker].iloc[-1]) / len(pct_ch_sorted)

        # reverse optimal weight
        weights[ticker] = 1 - kelly(win_rate=prob_win, net_profit=1, net_loss=1)

    sum_weights_assets = sum(weights.values())
    if sum_weights_assets > 1:
        weights = {t: 1.0 * w / sum_weights_assets for t, w in weights.items()}
        weights["cash"] = 0
    else:
        weights["cash"] = 1 - sum_weights_assets

    return weights


def generate_initial_portfolio_backtest(
    start_date, end_date, yf_df: pd.DataFrame = None
) -> None:
    target_weights = set_target_weights(start_date, end_date, yf_df=yf_df)
    tickers = [t for t in target_weights.keys() if t != "cash"]
    write_portfolio(
        end_date,
        tickers,
        [target_weights[t] for t in tickers],
        target_weights["cash"],
        "portfolio.json",
    )


def create_portfolio_file(date, path_savefile="portfolio.json"):
    tickers = ticker("assets")
    write_portfolio(
        date,
        tickers,
        weight=[0 for t in tickers],
        cash_amount=1,
        path_savefile=path_savefile,
    )


def generate_portfolio_via_rebalancing(
    end_date,
    rebalacing_period: int,
    yf_df: pd.DataFrame = None,
    transaction_fee_rate: float = 0.005,
    path_savefile: str = "portfolio.json",
) -> None:
    # print(read_portfolio()["cash"] + sum(e[1] for e in read_portfolio()["weights"]))
    portfolio = read_portfolio(path_savefile)

    # update_portfolio = False
    # if all_cash(portfolio):
    # update_portfolio = buy_signal(end_date, yf_df)
    # else:
    #     if sell_signal(end_date, yf_df):
    #         tickers, optimal_w, new_cash_amount = sell(portfolio, transaction_fee_rate)
    #         write_portfolio(
    #             end_date, tickers, optimal_w, new_cash_amount, path_savefile
    #         )
    #     else:
    #         update_portfolio = (
    #             end_date - datetime.strptime(portfolio["date_generated"], "%Y-%m-%d")
    #         ).days >= rebalacing_period

    update_portfolio = (
        end_date - datetime.strptime(portfolio["date_generated"], "%Y-%m-%d")
    ).days >= rebalacing_period
    update_portfolio = True
    if update_portfolio:
        start_date = end_date - timedelta(days=rebalacing_period)
        target_weights = set_target_weights(start_date, end_date, yf_df=yf_df)
        tickers, optimal_w, new_cash_amount = optimize_asset_portfolio_via_rebalancing(
            portfolio, target_weights, transaction_fee_rate
        )

        write_portfolio(end_date, tickers, optimal_w, new_cash_amount, path_savefile)


def optimize_asset_portfolio_via_rebalancing(
    portfolio: Dict,
    target_weights: Iterable[float],
    transaction_fee_rate: float,
    key="Close",
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


def optimize_asset_portfolio_via_equal_weight(
    df: pd.DataFrame, key="Close", return_samples: bool = False
) -> Tuple:
    df_return = yf_return(df)
    tickers = df_return.columns
    optimal_w = np.array([1.0 / df_return[t].iloc[-1] for t in tickers])
    optimal_w /= optimal_w.sum()
    return tickers, optimal_w, -1, -1


def compute_budge(total_budget: int, path_portfolio: str = "portfolio.json"):
    """
    Args:
        total_budget: dollars
    Return:
        list of money for each weight
    """
    pf = read_portfolio(path_portfolio)
    return [e + [e[1] * total_budget] for e in pf["weights"]]


###############
### Utility ###
###############
def percent_change(
    yf_df: pd.DataFrame, key="Close", periods=1, fillna_zero=True
) -> pd.DataFrame:
    if fillna_zero:
        return (
            yf_df[key]
            .pct_change(periods=periods)
            .fillna(0)
            .dropna(axis=1, how="all")
            .dropna(axis=0)
        )
    else:
        return (
            yf_df[key]
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
    df_ma5 = df["Close"].rolling(window=5).mean().dropna()
    df_ma5 /= df_ma5.iloc[0]
    df_ma10 = df["Close"].rolling(window=10).mean().dropna()
    df_ma10 /= df_ma10.iloc[0]
    df_ma20 = df["Close"].rolling(window=20).mean().dropna()
    df_ma20 /= df_ma20.iloc[0]

    for ticker in tickers:
        plt.plot(df_return[ticker].index, list(df_return[ticker]), label=ticker)
        plt.plot(df_ma5[ticker].index, list(df_ma5[ticker]), label=ticker + "_ma5")
        plt.plot(df_ma10[ticker].index, list(df_ma10[ticker]), label=ticker + "_ma10")
        plt.plot(df_ma20[ticker].index, list(df_ma20[ticker]), label=ticker + "_ma20")

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
        tickers = ticker("assets")
        names = name("assets")
        t2n = ticker2name("assets")
    elif target == "sectors":
        tickers = ticker("sectors")
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


def append_returns(prev_r: Optional[pd.Series], next_r: pd.Series) -> pd.Series:
    if prev_r is None:
        return next_r
    else:
        return pd.concat([prev_r, next_r])
        # prev_last_r = prev_r.iloc[-1]
        # prev_last_date = prev_r.index[-1]
        # return pd.concat(
        #     [
        #         prev_r[:-1],
        #         (next_r[prev_last_date:] * prev_last_r / next_r[prev_last_date]),
        #     ]
        # )


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
    asset_open_date="2012-01-01",
    path_savefile: str = "backtest.png",
) -> None:
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    dict_benchmark = get_benchmark_data_backtest(start_date, end_date)
    start_date = get_benchmark_start_date(dict_benchmark)

    yf_df = download_assets(start_date=asset_open_date, end_date=end_date)

    period2series = {}
    for rebalacing_period in rebalacing_periods:
        generate_initial_portfolio_backtest(
            start_date - timedelta(days=rebalacing_period), start_date, yf_df=yf_df
        )

        portfolio_returns = pd.Series([1], index=[start_date], name="Close")
        # while current_date + timedelta(days=2 * rebalacing_period) <= end_date:
        #     opt_s, opt_e, test_s, test_e = set_dates_backtest(
        #         current_date, rebalacing_period
        #     )

        test_date = start_date + timedelta(days=1)
        while test_date <= end_date:
            if sum(test_date == yf_df["Close"].index) > 0:
                generate_portfolio_via_rebalancing(
                    end_date=test_date - timedelta(days=1),
                    rebalacing_period=rebalacing_period,
                    yf_df=yf_df[yf_df.index <= test_date - timedelta(days=1)],
                )

                # pf_r = portfolio_return(
                #     test_date,
                #     test_date + timedelta(days=1),
                #     portfolio_returns is not None,
                #     yf_df=yf_df,
                # )

                simulate_market(yf_df, test_date)

                pf_r = portfolio_return(test_date)
                portfolio_returns = append_returns(portfolio_returns, pf_r)

            test_date += timedelta(days=1)

            # current_date += timedelta(days=rebalacing_period)

        period2series[rebalacing_period] = portfolio_returns

        os.system("rm portfolio.json")

    plt.close()
    data = {f"portfolio_update_{p}": s for p, s in period2series.items()}
    data.update(dict_benchmark)
    df_return = pd.DataFrame(data)
    df_return.plot(figsize=(16, 8))
    plt.savefig(path_savefile)
