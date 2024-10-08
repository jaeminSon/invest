from typing import Tuple, List, Dict, Iterable, Optional
from datetime import datetime, timedelta
import ssl
import json

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


#####################
### yahoo finance ###
#####################
def download_SandP(start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    if end_date is None:
        start_date, end_date = period(start_date)
    tickers = ticker("s&p500")
    return download(tickers, start_date, end_date)


def download_assets(start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    if end_date is None:
        start_date, end_date = period(start_date)
    tickers = ticker("assets")
    return download(tickers, start_date, end_date)


def download_sectors(start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    if end_date is None:
        start_date, end_date = period(start_date)
    tickers = ticker("sectors")
    return download(tickers, start_date, end_date)


def download_portfolio(start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    pf = read_portfolio()
    tickers = [e[0] for e in pf["weights"] if e[0] != "cash"]

    if end_date is None:
        start_date, end_date = period(start_date)

    df = download(tickers, start_date, end_date)

    return df


def download(symbol, start_date, end_date) -> pd.DataFrame:
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data


def period(
    start_date: Optional[str] = None, date_back: Optional[str] = None
) -> Tuple[str, str]:
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


def benchmark_return(benchmark: str, start_date: str, end_date: str) -> pd.Series:
    try:
        benchmark_ticker = find_ticker(benchmark)
    except:
        benchmark_ticker = benchmark

    df_benchmark = download([benchmark_ticker], start_date, end_date)
    return (df_benchmark["Close"].pct_change().fillna(0) + 1).cumprod().dropna()


def portfolio_return(start_date: str, end_date: str) -> pd.Series:
    portfolio = read_portfolio()
    df = download_portfolio(start_date, end_date)

    portfolio_return = pd.Series(0, index=df["Close"].index, name="Close")
    for ticker, weight, name in portfolio["weights"]:
        portfolio_return += (
            df["Close"][ticker].pct_change().fillna(0) + 1
        ).cumprod().dropna() * weight

    if "cash" in portfolio:
        portfolio_return += portfolio["cash"]

    return portfolio_return


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
    tickers: List[str],
    weight: np.ndarray,
    expected_r: float,
    expected_v: float,
    path_savefile: str,
) -> None:
    assert len(tickers) == len(
        weight
    ), "tickers and weight should have the same length."

    t2n = ticker2name("assets")

    data = {
        "expected_return": expected_r,
        "expected_volatility": expected_v,
        "weights": [(t, w, t2n[t]) for t, w in zip(tickers, weight)],
    }

    json.dump(data, open(path_savefile, "w"), indent=4)


def write_portfolio_rebalancing(
    tickers: List[str],
    weight: np.ndarray,
    cash_amount: float,
    path_savefile: str,
) -> None:
    assert len(tickers) == len(
        weight
    ), "tickers and weight should have the same length."

    t2n = ticker2name("assets")

    data = {
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
        .pct_change(periods=window, fill_method=None)
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
def generate_portfolio_via_rebalancing(
    start_date: str,
    end_date: str,
    target_weights: List[float],
    transaction_fee_rate: float,
    path_savefile: str = "portfolio.json",
) -> None:
    portfolio = read_portfolio()
    cash_amount = portfolio["cash"]

    df = download_assets(start_date=start_date, end_date=end_date)

    tickers, optimal_w, new_cash_amount = optimize_asset_portfolio_via_rebalancing(
        df, cash_amount, target_weights, transaction_fee_rate
    )
    write_portfolio_rebalancing(tickers, optimal_w, new_cash_amount, path_savefile)


def generate_portfolio_via_equal_weight(
    start_date: str,
    end_date: str,
    path_savefile: str = "portfolio.json",
) -> None:
    df = download_assets(start_date=start_date, end_date=end_date)
    (
        tickers,
        optimal_w,
        optimal_r,
        optimal_v,
    ) = optimize_asset_portfolio_via_equal_weight(df)
    write_portfolio(tickers, optimal_w, optimal_r, optimal_v, path_savefile)


def generate_portfolio_via_sampling(
    start_date: str,
    end_date: str,
    path_savefile: str = "portfolio.json",
) -> None:
    df = download_assets(start_date=start_date, end_date=end_date)
    tickers, optimal_w, optimal_r, optimal_v = optimize_asset_portfolio_via_sampling(df)
    write_portfolio(tickers, optimal_w, optimal_r, optimal_v, path_savefile)


def generate_portfolio_via_efficient_frontier(
    start_date: str,
    end_date: str,
    path_savefile: str = "portfolio.json",
) -> None:
    df = download_assets(start_date=start_date, end_date=end_date)
    (
        tickers,
        optimal_w,
        optimal_r,
        optimal_v,
    ) = optimize_asset_portfolio_via_efficient_frontier(df)
    write_portfolio(tickers, optimal_w, optimal_r, optimal_v, path_savefile)


def kelly(win_rate: float, net_profit: float, net_loss: float) -> float:
    assert 0 <= win_rate <= 1
    return (1.0 * win_rate / (net_loss + 1e-6)) - (
        1.0 * (1 - win_rate) / (net_profit + 1e-6)
    )


def return_tolerance(percentile: float = 0.03):
    """
    Assume +percentile and -percentile/(1+percentile) alternate.
    Then, compute mean return in that case, which is positive.
    """
    return (percentile - percentile / (1 + percentile)) / 2


def filter_assets(pct_ch: pd.DataFrame, key="Close"):
    # expected return better than SPY
    mean_const = pct_ch.mean() > pct_ch["SPY"].mean()
    # volatility lower than SPY with minimum expected return
    std_const = pct_ch.std() < pct_ch["SPY"].std()
    min_mean_const = pct_ch.mean() > return_tolerance()

    return pct_ch.loc[:, mean_const | (std_const & min_mean_const)]


def efficient_frontier(
    ret: np.ndarray, cov_mat: np.ndarray, N=1_000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert len(ret) == len(
        cov_mat
    ), "Please make sure the returns matches the shape of the covariance matrix."

    inv_cov = np.linalg.inv(cov_mat)

    n = len(ret)
    a = np.ones(n).T @ inv_cov @ ret
    b = ret.T @ inv_cov @ ret
    c = np.ones(n).T @ inv_cov @ np.ones(n)
    d = b * c - a**2

    returns = np.linspace(0.0, 0.01, N)
    volatilities = np.zeros(N)
    weights = np.zeros((N, len(ret)))

    for i in range(N):
        w = 1 / d * (c * inv_cov @ ret - a * inv_cov @ np.ones(n)) * returns[
            i
        ] + 1 / d * (b * inv_cov @ np.ones(n) - a * inv_cov @ ret)
        volatilities[i] = np.sqrt(w.T @ cov_mat @ w)
        weights[i, :] = w

    return weights, returns, volatilities


def random_portfolio(
    ret: np.ndarray, cov_mat: np.ndarray, N=100_000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(0)

    weights = np.random.random((N, len(ret)))
    weights /= np.sum(weights, axis=1, keepdims=True)

    returns = np.array([w.T @ ret for w in weights])

    volatilities = np.sqrt([w.T @ cov_mat @ w for w in weights])

    return weights, returns, volatilities


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


def optimize_asset_portfolio_via_rebalancing(
    df: pd.DataFrame,
    cash_amount: float,
    target_weights: Iterable[float],
    transaction_fee_rate: float,
    key="Close",
) -> Tuple:
    df_return = df[key].iloc[-1] / df[key].iloc[0]

    tickers = list(df_return.index)
    t2r = {ticker: df_return[ticker] * target_weights[ticker] for ticker in tickers}
    total = sum(t2r.values()) + cash_amount

    optimal_w = []
    for ticker in tickers:
        target = target_weights[ticker] * total
        new_weight = target / df_return[ticker]
        new_weight /= (
            1 + transaction_fee_rate
        )  # compensate for transaction fee by reducing weight
        optimal_w.append(new_weight)

    curr_cash_amount = total * target_weights["cash"]

    return tickers, optimal_w, curr_cash_amount


def optimize_asset_portfolio_via_efficient_frontier(
    df: pd.DataFrame, key="Close", return_frontiers: bool = False
) -> Tuple:
    pct_ch = percent_change(df, key)
    tickers, r, cov = compute_return_volatility(pct_ch)

    weights, returns, volatilities = efficient_frontier(r, cov)

    sharpe = returns / volatilities
    index_opt = np.argmax(sharpe)
    optimal_w = weights[index_opt]
    optimal_r = returns[index_opt]
    optimal_v = volatilities[index_opt]

    if return_frontiers:
        return tickers, optimal_w, optimal_r, optimal_v, returns, volatilities, sharpe
    else:
        return tickers, optimal_w, optimal_r, optimal_v


def optimize_asset_portfolio_via_equal_weight(
    df: pd.DataFrame, key="Close", return_samples: bool = False
) -> Tuple:
    df_return = yf_return(df)
    tickers = df_return.columns
    optimal_w = np.array([1.0 / df_return[t].iloc[-1] for t in tickers])
    optimal_w /= optimal_w.sum()
    return tickers, optimal_w, -1, -1


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


def optimality(returns: np.ndarray, volatilities: np.ndarray) -> np.ndarray:
    covered = np.array([False] * len(returns))
    for i in range(len(returns)):
        if not covered[i]:
            covered[(returns <= returns[i]) & (volatilities >= volatilities[i])] = True
            covered[i] = False

    return covered


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
def percent_change(yf_df: pd.DataFrame, key="Close", periods=1) -> pd.DataFrame:
    return (
        yf_df[key]
        .pct_change(periods=periods)
        .fillna(0)
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


def cov_mat(yf_df: pd.DataFrame, key="Close") -> np.ndarray:
    """
    Args:
        yf_df: downloaded yahoo finance dataframe (T, N)
        key: key for the yahoo finance dataframe
    Returns:
        N*N numpy array
    """
    pct_ch = percent_change(yf_df, key)
    return cov_mat_from_pct_ch(pct_ch)


def cov_mat_from_pct_ch(pct_ch: pd.DataFrame) -> np.ndarray:
    return np.array(pct_ch.cov())


def inverse_cov_mat(cov_mat: np.ndarray, eps: float = 1e-2) -> np.ndarray:
    w, v = np.linalg.eig(cov_mat)
    assert np.where(w >= 0, True, False).sum() == len(
        w
    ), "Please ensure the covariance matrix is positive semi-definite."

    weighted_w = w / np.sum(w)
    w_hat = np.where(weighted_w >= eps, w, 0)
    noise_free_w = w_hat * (np.sum(w) / np.sum(w_hat))

    inv_mat = v @ np.diag(np.where(noise_free_w != 0, 1 / noise_free_w, 0)) @ v.T

    return inv_mat


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
    start_date: str, path_savefile: str = "return_leverage_with_ma.png"
) -> None:
    tickers = ["SPXL", "TQQQ", "SOXL"]
    start_date, end_date = period(start_date)
    df = download(tickers, start_date, end_date)

    plt.close()
    plt.figure(figsize=(16, 12))
    df_return = yf_return(df)
    df_ma10 = df["Close"].rolling(window=10).mean().dropna()
    df_ma10 /= df_ma10.iloc[0]
    df_ma20 = df["Close"].rolling(window=20).mean().dropna()
    df_ma20 /= df_ma20.iloc[0]

    for ticker in tickers:
        plt.plot(df_return[ticker].index, list(df_return[ticker]), label=ticker)
        plt.plot(df_ma10[ticker].index, list(df_ma10[ticker]), label=ticker + "_ma10")
        plt.plot(df_ma20[ticker].index, list(df_ma20[ticker]), label=ticker + "_ma20")

    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()

    plt.savefig(path_savefile)


def plot_return_leverage(
    start_date: str, path_savefile: str = "return_leverage.png"
) -> None:
    tickers = ["SPY", "SPXL", "TQQQ", "QQQ", "SOXX", "SOXL"]
    start_date, end_date = period(start_date)
    df = download(tickers, start_date, end_date)

    plot_return(df, path_savefile)


def plot_return_by_sector(
    start_date: str, path_savefile: str = "return_sector.png"
) -> None:
    df = download_sectors(start_date)
    t2n = {t: " ".join(n.split()[2:]) for t, n in ticker2name("sectors").items()}
    df.columns = pd.MultiIndex.from_tuples([(c1, t2n[c2]) for c1, c2 in df.columns])

    plot_return(df, path_savefile)


def plot_return_by_asset(
    start_date: str, path_savefile: str = "return_asset.png"
) -> None:
    df = download_assets(start_date)

    pct_ch = (
        df["Close"].pct_change(periods=1, fill_method=None).dropna(axis=1, how="all")
    )
    df_return = (pct_ch + 1).cumprod()
    df_return = df_return[
        [col for col in df_return.columns if df_return[col].iloc[-1] > 4]
    ]
    t2n = {t: n for t, n in ticker2name("assets").items()}
    df_return.columns = [t2n[c] for c in df_return.columns]

    plt.close()
    df_return.plot(figsize=(16, 12))
    plt.savefig(path_savefile)


def plot_return_index(start_date: str, path_savefile: str = "return_index.png") -> None:
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
    start_date: str, path_savefile: str = "return_portfolio_stocks.png"
) -> None:
    pf = read_portfolio()
    t2n = {e[0]: e[2] for e in pf["weights"]}

    df = download_portfolio(start_date)

    df.columns = pd.MultiIndex.from_tuples([(c1, t2n[c2]) for c1, c2 in df.columns])

    plot_return(df, path_savefile)


def plot_portfolio_via_efficient_frontier(
    start_date: str,
    key="Close",
    path_savefile: str = "portfolio_via_efficient_frontier.png",
) -> None:
    df = download_assets(start_date)

    (
        tickers,
        weight,
        optimal_r,
        optimal_v,
        returns,
        valitilities,
        sharpe,
    ) = optimize_asset_portfolio_via_efficient_frontier(df, key, return_frontiers=True)

    df_random = pd.DataFrame(
        {
            "return": returns,
            "volatility": valitilities,
            "Sharpe": sharpe,
        }
    )

    plt.close()
    sns.scatterplot(
        df_random, x="volatility", y="return", hue="Sharpe", s=3, legend=False
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
    # plot S&P, Nasdaq, Dow, Gold
    plt.scatter(
        [np.sqrt(df[key][find_ticker("s&p")].pct_change().fillna(0).var())],
        [df[key][find_ticker("s&p")].pct_change().fillna(0).mean()],
        color="black",
        marker="o",
        s=10,
        label="s&p",
    )
    plt.scatter(
        [np.sqrt(df[key][find_ticker("nasdaq")].pct_change().fillna(0).var())],
        [df[key][find_ticker("nasdaq")].pct_change().fillna(0).mean()],
        color="blue",
        marker="o",
        s=10,
        label="nasdaq",
    )
    plt.scatter(
        [np.sqrt(df[key][find_ticker("dow")].pct_change().fillna(0).var())],
        [df[key][find_ticker("dow")].pct_change().fillna(0).mean()],
        color="green",
        marker="o",
        s=10,
        label="dow",
    )
    plt.scatter(
        [np.sqrt(df[key][find_ticker("gold")].pct_change().fillna(0).var())],
        [df[key][find_ticker("gold")].pct_change().fillna(0).mean()],
        color="yellow",
        marker="o",
        s=10,
        label="gold",
    )

    plt.legend()
    plt.savefig(path_savefile, bbox_inches="tight")


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
    # plot S&P, Nasdaq, Dow, Gold
    plt.scatter(
        [np.sqrt(df[key][find_ticker("s&p")].pct_change().fillna(0).var())],
        [df[key][find_ticker("s&p")].pct_change().fillna(0).mean()],
        color="black",
        marker="o",
        s=10,
        label="s&p",
    )
    plt.scatter(
        [np.sqrt(df[key][find_ticker("nasdaq")].pct_change().fillna(0).var())],
        [df[key][find_ticker("nasdaq")].pct_change().fillna(0).mean()],
        color="blue",
        marker="o",
        s=10,
        label="nasdaq",
    )
    plt.scatter(
        [np.sqrt(df[key][find_ticker("dow")].pct_change().fillna(0).var())],
        [df[key][find_ticker("dow")].pct_change().fillna(0).mean()],
        color="green",
        marker="o",
        s=10,
        label="dow",
    )
    plt.scatter(
        [np.sqrt(df[key][find_ticker("gold")].pct_change().fillna(0).var())],
        [df[key][find_ticker("gold")].pct_change().fillna(0).mean()],
        color="yellow",
        marker="o",
        s=10,
        label="gold",
    )
    # # plot all assets in the portfolio
    # for i in range(len(r)):
    #     plt.scatter(
    #         [np.sqrt(cov[i, i])],
    #         [r[i]],
    #         color="red",
    #         s=10,
    #     )

    plt.legend()
    plt.savefig(path_savefile, bbox_inches="tight")


def plot_matrix(start_date: str, target: str, matrix_data: str):
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
    elif matrix_data == "cov":
        data = cov_mat(df)
    else:
        raise ValueError(f"Unknown matrix_data {matrix_data}")

    plt.close()
    # assets have many elements that figsize should be big
    if target == "assets":
        plt.figure(figsize=(60, 48))
    sns.heatmap(data, cmap="seismic", annot=True)
    plt.savefig(f"{matrix_data}_{target}.png", bbox_inches="tight")


def plot_correlation(start_date: str, target: str = "assets"):
    plot_matrix(start_date, target, "correlation")


def plot_covariance(start_date: str, target: str = "assets"):
    plot_matrix(start_date, target, "cov")


def plot_return_measure(start_date: str, path_savefile: str = "return_measure.png"):
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


def plot_backtest(start_date: str, end_date: str, benchmark: str = "s&p") -> None:
    portfolio_returns = portfolio_return(start_date, end_date)
    benchmark_returns = benchmark_return(benchmark, start_date, end_date)

    plt.close()
    df_return = pd.DataFrame(
        {"portfolio": portfolio_returns, benchmark: benchmark_returns}
    )
    df_return.plot(figsize=(8, 6))
    plt.savefig("backtest.png")


def plot_periodic_update_backtest(
    start_date: str,
    end_date: str,
    method: str,
    update_periods: List[int] = [50, 100, 200],  # [10, 20, 50, 100, 200],
    benchmark: str = "s&p",
    transaction_fee_rate: float = 0.01,
) -> None:
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    benchmark_returns = benchmark_return(
        benchmark,
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

    # period2series = {}
    # for update_period in update_periods:
    #     tickers = ticker("assets")
    #     prev_weights = {t: 1.0 / len(tickers) for t in tickers}

    #     portfolio_returns = None
    #     current_date = start_date - timedelta(days=update_period)
    #     while current_date + timedelta(days=2*update_period) <= end_date:
    #         # optimize porfolio with data between [opt_s, opt_e] and test on [test_s, test_e]
    #         opt_s = current_date
    #         opt_e = current_date + timedelta(days=update_period)
    #         test_s = opt_e
    #         test_e = test_s + timedelta(days=update_period)

    #         if method == "sampling":
    #             generate_portfolio_via_sampling(
    #                 start_date=opt_s,
    #                 end_date=opt_e,
    #             )
    #         elif method == "efficient_frontier":
    #             generate_portfolio_via_efficient_frontier(
    #                 start_date=opt_s,
    #                 end_date=opt_e,
    #             )
    #         elif method == "equal_weight":
    #             generate_portfolio_via_equal_weight(
    #                 start_date=opt_s,
    #                 end_date=opt_e,
    #             )

    #         curr_weights = {e[0]: e[1] for e in read_portfolio()["weights"]}
    #         change_ratio = sum(
    #             [
    #                 abs(curr_weights[asset] - prev_weights[asset])
    #                 for asset in curr_weights
    #             ]
    #         )
    #         prev_weights = {e[0]: e[1] for e in read_portfolio()["weights"]}

    #         pf_r = portfolio_return(test_s, test_e)
    #         if portfolio_returns is None:
    #             portfolio_returns = pf_r
    #         else:
    #             portfolio_returns = pd.concat(
    #                 [
    #                     portfolio_returns,
    #                     pf_r
    #                     * portfolio_returns.iloc[-1]
    #                     * (1 - transaction_fee_rate * change_ratio),
    #                 ]
    #             )

    #         current_date += timedelta(days=update_period)

    #     period2series[update_period] = portfolio_returns

    plt.close()
    # data = {f"portfolio_update_{p}": s for p, s in period2series.items()}
    data = {}
    data.update(
        {
            benchmark: benchmark_returns,
            "SPXL": spxl,
            "TQQQ": tqqq,
            "SOXL": soxl,
            "equal_rate": equal_rate,
        }
    )
    df_return = pd.DataFrame(data)
    df_return.plot(figsize=(16, 8))
    plt.savefig(f"backtest_update_{method}.png")


def plot_rebalancing_backtest(
    start_date: str,
    end_date: str,
    update_periods: List[int] = [10, 20, 50, 100, 200],
    benchmark: str = "s&p",
    transaction_fee_rate: float = 0.005,
) -> None:
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    tickers = ticker("assets")

    benchmark_returns = benchmark_return(
        benchmark,
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

    target_weights = {
        "cash": 1.0 / 4,
        "SOXL": 1.0 / 4,
        "SPXL": 1.0 / 4,
        "TQQQ": 1.0 / 4,
    }
    period2series = {}
    for update_period in update_periods:
        portfolio_returns = None
        current_date = start_date - timedelta(days=update_period)
        while current_date + timedelta(days=2 * update_period) <= end_date:
            # optimize porfolio with data between [opt_s, opt_e] and test on [test_s, test_e]
            opt_s = current_date
            opt_e = current_date + timedelta(days=update_period)
            test_s = opt_e
            test_e = test_s + timedelta(days=update_period)

            generate_portfolio_via_rebalancing(
                start_date=opt_s,
                end_date=opt_e,
                target_weights=target_weights,
                transaction_fee_rate=transaction_fee_rate,
            )

            pf_r = portfolio_return(test_s, test_e)
            if portfolio_returns is None:
                portfolio_returns = pf_r
            else:
                portfolio_returns = pd.concat(
                    [portfolio_returns, pf_r * portfolio_returns.iloc[-1]]
                )

            current_date += timedelta(days=update_period)

        period2series[update_period] = portfolio_returns

    plt.close()
    data = {f"portfolio_update_{p}": s for p, s in period2series.items()}
    data.update(
        {
            benchmark: benchmark_returns,
            "SPXL": spxl,
            "TQQQ": tqqq,
            "SOXL": soxl,
            "equal_rate": equal_rate,
        }
    )
    df_return = pd.DataFrame(data)
    df_return.plot(figsize=(16, 8))
    plt.savefig(f"backtest_update_balance.png")
