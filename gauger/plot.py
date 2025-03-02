import os
from fractions import Fraction
from typing import Dict, List

import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .yahoo_finance import (
    download,
    download_by_group,
    yf_return,
    period,
    ticker2name,
    get_ticker,
)
from .fred import get_fred_series
from .portfolio import (
    kelly_cube,
    divide_by_rolling_ma,
    density_function,
    bet_ratios_martingale_from_pdf,
)


DIR_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def plot_kelly(var_fix: str, path_savefile: str) -> None:
    assert var_fix in ["p_win", "profit", "loss"], f"Unknown variable {var_fix}."

    path_kelly_cube = os.path.join(DIR_DATA, "kelly_cube.npy")

    if os.path.exists(path_kelly_cube):
        cube = np.load(path_kelly_cube)
    else:
        cube = kelly_cube()
        np.save(path_kelly_cube, cube)

    assert (
        len(set(cube.shape)) == 1
    ), f"kelly cube does not have equal length {cube.shape}."

    length = cube.shape[0]
    linspace = np.linspace(0, 1.0, num=length)
    fix_interval = length // 10

    if var_fix == "p_win":
        all_data = [cube[fix_interval * (i + 1) - 1, ...] for i in range(10)]
        ylabel, xlabel = "profit", "loss"
    elif var_fix == "profit":
        all_data = [cube[:, fix_interval * (i + 1) - 1, :] for i in range(10)]
        ylabel, xlabel = "p_win", "loss"
    elif var_fix == "loss":
        all_data = [cube[..., fix_interval * (i + 1) - 1] for i in range(10)]
        ylabel, xlabel = "p_win", "profit"

    plt.close()
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(2):
        for j in range(5):
            index_data = 5 * i + j
            draw = axes[i, j].imshow(
                np.array(all_data[index_data]),
                cmap="viridis",
                interpolation="nearest",
                vmin=0,
                vmax=1,
            )

            val_var_fix = linspace[fix_interval * (index_data + 1)]
            axes[i, j].set_title(f"{var_fix} {val_var_fix:.1}")
            axes[i, j].set_xlabel(xlabel)
            if j == 0:
                axes[i, j].set_ylabel(ylabel)
            axes[i, j].set_xticks(
                range(0, length, length // 5),
                [linspace[i] for i in range(0, length, length // 5)],
            )
            axes[i, j].set_yticks(
                range(0, length, length // 5),
                [linspace[i] for i in range(0, length, length // 5)],
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


def ticker2name_for_plot(group: str) -> Dict[str, str]:
    if group == "sectors":
        return {t: " ".join(n.split()[2:]) for t, n in ticker2name("sectors").items()}
    else:
        return ticker2name(group)


def plot_return_by_group(
    group: str, start_date: str, path_savefile: str = "return_sector.png"
) -> None:
    df = download_by_group(group, start_date)

    t2n = ticker2name_for_plot(group)

    df.columns = pd.MultiIndex.from_tuples([(c1, t2n[c2]) for c1, c2 in df.columns])

    plot_return(df, path_savefile)


def plot_matrix(
    group: str, start_date, datatype: str, path_savefile: str, key: str = "Close"
) -> None:
    start_date, end_date = period(start_date)

    tickers = get_ticker(group)

    t2n = ticker2name_for_plot(group)

    df = download(tickers, start_date, end_date)
    df.columns = pd.MultiIndex.from_tuples([(c1, t2n[c2]) for c1, c2 in df.columns])

    if datatype == "correlation":
        data = df[key].corr()
    else:
        raise ValueError(f"Unknown datatype {datatype}")

    plt.close()
    # assets have many elements that figsize should be big
    sns.heatmap(data, cmap="seismic", annot=True)
    plt.savefig(path_savefile, bbox_inches="tight")


def plot_correlation(
    group: str, start_date: str, path_savefile: str, key="Close"
) -> None:
    plot_matrix(group, start_date, "correlation", path_savefile, key)


def plot_sort_by_return(
    group: str, start_date: str, path_savefile: str = "return_etf.png"
) -> None:
    df = download_by_group(group, start_date)

    df_return = yf_return(df)

    t2n = ticker2name_for_plot(group)

    df_return.columns = [t2n[c] for c in df_return.columns]

    plt.close()
    plt.figure(figsize=(20, 15))
    list_ticker_return = sorted(
        list(zip(df_return.iloc[-1].index, df_return.iloc[-1].values)),
        key=lambda x: x[1],
    )
    x, y = zip(*list_ticker_return)
    plt.bar(x, y, color="skyblue")
    plt.axhline(y=1, color="red", linestyle="--", linewidth=1)
    plt.ylabel("Return")
    plt.xticks(rotation=90)
    plt.savefig(path_savefile, bbox_inches="tight")


def plot_mean_std(
    df: pd.DataFrame | List[pd.DataFrame],
    column_name: str | List[str],
    path_savefile: str,
    width_plot: float = 2,
):
    def add_std(df: pd.DataFrame):
        assert "mean" in df.columns
        std = np.std(df["mean"])
        df["+1s"] = df["mean"] + std
        df["+2s"] = df["mean"] + 2 * std
        df["-1s"] = df["mean"] - std
        df["-2s"] = df["mean"] - 2 * std
        return df.dropna()

    def config_plots(column_name: str, width_plot: float):
        return {
            "color": {
                column_name: "blue",
                "mean": "yellow",
                "+1s": "orange",
                "+2s": "red",
                "-1s": "green",
                "-2s": "lime",
            },
            "style": {
                column_name: "-",
                "mean": "--",
                "+1s": "--",
                "+2s": "--",
                "-1s": "--",
                "-2s": "--",
            },
            "linewidth": {
                column_name: width_plot,
                "mean": 1,
                "+1s": 1,
                "+2s": 1,
                "-1s": 1,
                "-2s": 1,
            },
        }

    if isinstance(df, pd.DataFrame):
        df = add_std(df)
        plt.close()
        fig, ax = plt.subplots(figsize=(16, 8))
        plot_kwargs = config_plots(column_name, width_plot)
        for column in df.columns:
            df[column].plot(
                ax=ax, label=column, **{k: plot_kwargs[k][column] for k in plot_kwargs}
            )
        plt.legend(loc="best")
        plt.savefig(path_savefile)
    elif isinstance(df, list) and isinstance(column_name, list):
        assert len(df) == len(column_name) == 2, "Currently Price and Volume only."
        plt.close()
        fig, axes = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [5, 1]}, figsize=(20, 18)
        )
        for i in range(2):
            df_std_added = add_std(df[i])
            plot_kwargs = config_plots(column_name[i], width_plot)
            for column in df_std_added.columns:
                df_std_added[column].plot(
                    ax=axes[i],
                    label=column,
                    **{k: plot_kwargs[k][column] for k in plot_kwargs},
                )
            axes[i].legend(loc="best")
        plt.tight_layout()
        plt.savefig(path_savefile)


def plot_A_divided_by_B(
    A: pd.Series, B: pd.Series, column_name: str, ma_window: int, path_savefile: str
) -> None:
    series = pd.concat([A, B], axis=1).dropna()
    df = (series[0] / series[1]).to_frame(name=column_name)
    df = (df.pct_change().fillna(0) + 1).cumprod().dropna()
    df["mean"] = df.rolling(window=ma_window).mean()

    plot_mean_std(df, column_name, path_savefile)


def plot_sandp_divided_by_m2(
    start_date: str,
    end_date: str = None,
    path_savefile: str = "SP500_divided_by_M2.png",
):
    if end_date is None:
        start_date, end_date = period(start_date)

    sandp = get_fred_series("s&p500", start_date, end_date)
    m2 = get_fred_series("m2", start_date, end_date)

    plot_A_divided_by_B(sandp, m2, "S&P / M2", 12, path_savefile)


def plot_sandp_divided_by_gdp(
    start_date: str,
    end_date: str = None,
    path_savefile: str = "SP500_divided_by_gdp.png",
):
    if end_date is None:
        start_date, end_date = period(start_date)

    sandp = get_fred_series("s&p500", start_date, end_date)
    gdp = get_fred_series("gdp", start_date, end_date)

    plot_A_divided_by_B(sandp, gdp, "S&P / GDP", 4, path_savefile)


def plot_nasdaq_divided_by_m2(
    start_date: str,
    end_date: str = None,
    path_savefile: str = "Nasdaq_divided_by_M2.png",
):
    if end_date is None:
        start_date, end_date = period(start_date)

    sandp = get_fred_series("nasdaq", start_date, end_date)
    m2 = get_fred_series("m2", start_date, end_date)

    plot_A_divided_by_B(sandp, m2, "Nasdaq / M2", 12, path_savefile)


def plot_nasdaq_divided_by_gdp(
    start_date: str,
    end_date: str = None,
    path_savefile: str = "Nasdaq_divided_by_gdp.png",
):
    if end_date is None:
        start_date, end_date = period(start_date)
    nasdaq = get_fred_series("nasdaq", start_date, end_date)
    gdp = get_fred_series("gdp", start_date, end_date)

    plot_A_divided_by_B(nasdaq, gdp, "Nasdaq / GDP", 4, path_savefile)


def plot_SP500_JOLT(
    start_date: str,
    end_date: str = None,
    path_savefile: str = "SP500_JOLT.png",
):
    if end_date is None:
        start_date, end_date = period(start_date)

    jolt = get_fred_series("jolt", start_date, end_date).to_frame("JOLT")
    jolt.index.name = "Date"
    sp500 = download("SPY", start_date, end_date)
    df = jolt.merge(sp500["Close"]["SPY"].to_frame(), on="Date")
    df.dropna(inplace=True)

    corr = np.corrcoef(df["SPY"], df["JOLT"])[0, 1]
    print(f"[All] Correlation between SP500 and JOLT: {corr}")

    # before chatgpt debut
    corr = np.corrcoef(
        df["SPY"][df["SPY"].index <= "2022-11-30"],
        df["JOLT"][df["JOLT"].index <= "2022-11-30"],
    )[0, 1]
    print(f"[Before chatgpt] Correlation between SP500 and JOLT: {corr}")

    # after chatgpt debut
    corr = np.corrcoef(
        df["SPY"][df["SPY"].index >= "2022-11-30"],
        df["JOLT"][df["JOLT"].index >= "2022-11-30"],
    )[0, 1]
    print(f"[After chatgpt] Correlation between SP500 and JOLT: {corr}")

    plt.close()
    fig, ax1 = plt.subplots(figsize=(16, 12))
    ax1.plot(df["SPY"].index, list(df["SPY"]), color="blue", label="SPY")
    ax1.set_ylabel("SPY", color="blue")
    ax2 = ax1.twinx()
    ax2.plot(df["JOLT"].index, list(df["JOLT"]), color="red", label="JOLT")
    ax2.set_ylabel("JOLT", color="red")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    plt.savefig(path_savefile)


def plot_return_with_ma(
    group: str,
    start_date: str,
    window: int = 100,
    path_savefile: str = "return_leverage_with_ma.png",
) -> None:
    df = download_by_group(group, start_date)
    df_return = yf_return(df)
    df_ma = df_return.rolling(window=window).mean().dropna()
    df_return = df_return.loc[df_ma.index]
    df_volume = df["Volume"].loc[df_ma.index]

    tickers = df_return.columns

    plt.close()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [5, 1]}, figsize=(20, 18)
    )

    colors = [
        "orange",
        "purple",
        "brown",
        "blue",
        "red",
        "green",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]
    t2c = {tickers[i]: colors[i] for i in range(len(tickers))}

    for ticker in tickers:
        ax1.plot(
            df_return[ticker].index,
            list(df_return[ticker]),
            label=ticker,
            color=t2c[ticker],
        )

    for ticker in tickers:
        ax1.plot(df_ma[ticker].index, list(df_ma[ticker]), color=t2c[ticker])

        peaks = scipy.signal.find_peaks(df_ma[ticker], width=1, rel_height=0.01)[0]
        bottomes = scipy.signal.find_peaks(-df_ma[ticker], width=1, rel_height=0.01)[0]
        ax1.scatter(
            [df_ma[ticker].index[p] for p in peaks],
            [df_ma[ticker].iloc[p] for p in peaks],
            color="r",
            marker="v",
            s=30,
        )
        ax1.scatter(
            [df_ma[ticker].index[p] for p in bottomes],
            [df_ma[ticker].iloc[p] for p in bottomes],
            color="b",
            marker="v",
            s=30,
        )
    ax1.set_ylabel("Return", color="red")
    ax1.legend()

    for ticker in tickers:
        ax2.plot(
            df_volume[ticker].index,
            list(df_volume[ticker]),
            label=ticker,
            color=t2c[ticker],
        )
    ax2.set_xlabel("Date", color="black")
    ax2.set_ylabel("Volume", color="red")

    plt.tight_layout()
    plt.savefig(path_savefile)


def plot_series(
    group: str,
    start_date: str,
    end_date: str = None,
    window: int = 100,
    savedir: str = "figures",
) -> None:
    df = download_by_group(group, start_date, end_date)
    tickers = df["Volume"].columns

    for ticker in tickers:
        p_ratio = divide_by_rolling_ma(df["Close"][ticker], window)
        df_p_ratio = p_ratio.to_frame(name=ticker)
        df_p_ratio["mean"] = p_ratio.rolling(window=window).mean()

        v_ratio = divide_by_rolling_ma(df["Volume"][ticker], window)
        df_v_ratio = v_ratio.to_frame(name=ticker)
        df_v_ratio["mean"] = v_ratio.rolling(window=window).mean()

        plot_mean_std(
            [df_p_ratio, df_v_ratio],
            [ticker, ticker],
            width_plot=1.1,
            path_savefile=os.path.join(savedir, f"div_by_ma_{ticker}.png"),
        )


def plot_pdf(
    group: str,
    start_date: str,
    end_date: str = None,
    window: int = 100,
    savedir: str = "figures",
):
    def nearest_index(axis, value):
        return min(range(len(axis)), key=lambda i: abs(axis[i] - value))

    df = download_by_group(group, start_date, end_date)
    tickers = df["Close"].columns

    for ticker in tickers:
        plt.close()
        fig, axes = plt.subplots(
            1, 2, gridspec_kw={"width_ratios": [1, 1]}, figsize=(20, 9)
        )
        for i_axis, dtype in enumerate(["price", "log-volume"]):
            if dtype == "price":
                series = divide_by_rolling_ma(df["Close"][ticker], window)
            elif dtype == "log-volume":
                series = np.log(divide_by_rolling_ma(df["Volume"][ticker], window))
            p = density_function(list(series))
            if dtype == "price":
                x = np.linspace(0, 2, 1000)
            elif dtype == "log-volume":
                x = np.linspace(-2, 2, 1000)
            y = [p(i)[0] for i in x]
            index_curr = nearest_index(x, series.iloc[-1])
            x_curr = x[index_curr]
            y_curr = y[index_curr]

            axes[i_axis].hist(series, bins=100, color="orange", density=True)
            axes[i_axis].plot(x, y, color="red")
            axes[i_axis].scatter(
                [x_curr],
                [y_curr],
                color="blue",
                marker="v",
                s=100,
            )
            axes[i_axis].plot(
                [x_curr, x_curr],
                [y_curr, 0],
                linestyle="--",
                color="red",
            )

            if dtype == "price":
                bet_ratios = bet_ratios_martingale_from_pdf(p)
                p_r_sorted = sorted(bet_ratios.keys())
                indices_nearest_x = [nearest_index(x, p_r) for p_r in p_r_sorted]
                for i, p_r in enumerate(p_r_sorted):
                    bet = bet_ratios[p_r]
                    frac = Fraction(bet).limit_denominator()
                    i_nearest_x = indices_nearest_x[i]
                    axes[i_axis].plot(
                        [x[i_nearest_x], x[i_nearest_x]],
                        [y[i_nearest_x], 0],
                        linestyle="--",
                        color="blue",
                    )
                    axes[i_axis].text(
                        (x[i_nearest_x] + x[indices_nearest_x[i - 1]]) / 2
                        if i > 0
                        else x[i_nearest_x] - 0.2,
                        y[i_nearest_x] / 2,
                        f"{frac}",
                        fontsize=12,
                        color="blue"
                        if (i_nearest_x == 0 and x_curr < x[0])
                        or (x[indices_nearest_x[i - 1]] <= x_curr < x[i_nearest_x])
                        else "red",
                    )

                axes[i_axis].set_title(f"Price divdeded by {window}MA for {ticker}")
                axes[i_axis].set_xlabel(f"Price / {window}MA")
            else:
                axes[i_axis].set_title(
                    f"Log of volume divdeded by {window}MA for {ticker}"
                )
                axes[i_axis].set_xlabel(f"Log of Volume / {window}MA")

        plt.savefig(os.path.join(savedir, f"pdf_{ticker}.png"))
