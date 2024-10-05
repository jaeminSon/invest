from typing import Tuple, List, Dict, Optional
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
def download_assets(start_date: str) -> pd.DataFrame:
    start_date, end_date = period(start_date)
    tickers = ticker("assets")
    return download(tickers, start_date, end_date)


def download_sectors(start_date: str) -> pd.DataFrame:
    start_date, end_date = period(start_date)
    tickers = ticker("sectors")
    return download(tickers, start_date, end_date)


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
    path_savefile: str = "portfolio.json",
):
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


def read_portfolio(path_portfolio: str = "portfolio.json"):
    return json.load(open(path_portfolio))


#################
### portfolio ###
#################
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
    ret: np.ndarray, cov_mat: np.ndarray, N=1000
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

    returns = np.linspace(0.0, 0.005, N)
    volatilities = np.zeros(N)
    weight_arr = np.zeros((N, len(ret)))

    for i in range(N):
        w = 1 / d * (c * inv_cov @ ret - a * inv_cov @ np.ones(n)) * returns[
            i
        ] + 1 / d * (b * inv_cov @ np.ones(n) - a * inv_cov @ ret)
        volatilities[i] = np.sqrt(w.T @ cov_mat @ w)
        weight_arr[i, :] = w

    return weight_arr, returns, volatilities


def random_portfolio(
    ret: np.ndarray, cov_mat: np.ndarray, N=100_000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(0)

    weights = np.random.random((N, len(ret)))
    weights /= np.sum(weights, axis=1, keepdims=True)

    returns = np.array([w.T @ ret for w in weights])

    volatilities = np.sqrt([w.T @ cov_mat @ w for w in weights])

    return weights, returns, volatilities


def optimize_asset_portfolio(
    df: pd.DataFrame, key="Close", return_samples: bool = False
):
    pct_ch = percent_change(df, key)
    pct_ch_filted = filter_assets(pct_ch)

    tickers = list(pct_ch_filted.columns)
    r = expected_return_from_pct_ch(pct_ch_filted)
    cov = cov_mat_from_pct_ch(pct_ch_filted)
    assert all((pct_ch_filted[t].mean() - r[i]) < 1e-6 for i, t in enumerate(tickers))
    assert all(
        (pct_ch_filted[t].var() - cov[i, i]) < 1e-6 for i, t in enumerate(tickers)
    )

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
def percent_change(yf_df: pd.DataFrame, key="Close") -> pd.DataFrame:
    return yf_df[key].pct_change().dropna()


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
    datas = []
    for loss in losses:
        data = [[0] * len(profits) for _ in range(len(win_rates))]
        for i in range(len(win_rates)):
            for j in range(len(profits)):
                ratio = kelly(win_rates[i], profits[j], loss)
                data[i][j] = ratio if ratio > 0 else 0
        datas.append(data)

    plt.close()
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(2):
        for j in range(5):
            draw = axes[i, j].imshow(
                np.array(datas[5 * i + j]),
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


def plot_return_by_sector(
    start_date: str, path_savefile: str = "return_sector.png"
) -> None:
    df = download_sectors(start_date)
    t2n = {t: " ".join(n.split()[2:]) for t, n in ticker2name("sectors").items()}
    df.columns = pd.MultiIndex.from_tuples([(c1, t2n[c2]) for c1, c2 in df.columns])

    plot_return(df, path_savefile)


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


def plot_return_portfolio(
    start_date: str, path_savefile: str = "return_portfolio.png"
) -> None:
    pf = read_portfolio()
    tickers = [e[0] for e in pf["weights"]]
    t2n = {e[0]: e[2] for e in pf["weights"]}

    start_date, end_date = period(start_date)
    df = download(tickers, start_date, end_date)
    df.columns = pd.MultiIndex.from_tuples([(c1, t2n[c2]) for c1, c2 in df.columns])

    plot_return(df, path_savefile)


def plot_portfolio(
    start_date: str, key="Close", path_savefile: str = "portfolio.png"
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
    ) = optimize_asset_portfolio(df, key, return_samples=True)

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
        [np.sqrt(df[key][find_ticker("s&p")].pct_change().var())],
        [df[key][find_ticker("s&p")].pct_change().mean()],
        color="black",
        marker="o",
        s=10,
        label="s&p",
    )
    plt.scatter(
        [np.sqrt(df[key][find_ticker("nasdaq")].pct_change().var())],
        [df[key][find_ticker("nasdaq")].pct_change().mean()],
        color="blue",
        marker="o",
        s=10,
        label="nasdaq",
    )
    plt.scatter(
        [np.sqrt(df[key][find_ticker("dow")].pct_change().var())],
        [df[key][find_ticker("dow")].pct_change().mean()],
        color="green",
        marker="o",
        s=10,
        label="dow",
    )
    plt.scatter(
        [np.sqrt(df[key][find_ticker("gold")].pct_change().var())],
        [df[key][find_ticker("gold")].pct_change().mean()],
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
