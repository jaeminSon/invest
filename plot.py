import os

from gauger import (
    plot_kelly,
    plot_correlation,
    plot_return_by_group,
    plot_sort_by_return,
    plot_sandp_divided_by_m2,
    plot_sandp_divided_by_gdp,
    plot_nasdaq_divided_by_m2,
    plot_nasdaq_divided_by_gdp,
    plot_SP500_JOLT,
    plot_return_with_ma,
    plot_series,
    plot_pdf,
)

if __name__ == "__main__":
    # ########################
    # # for research purpose #
    # ########################
    # plot_kelly("p_win", os.path.join("figures", "kelly_pwin.png"))
    # plot_kelly("profit", os.path.join("figures", "kelly_profit.png"))
    # plot_kelly("loss", os.path.join("figures", "kelly_loss.png"))

    # ##################
    # # plot quarterly #
    # ##################
    plot_nasdaq_divided_by_gdp(
        start_date="1900-01-01",
        path_savefile=os.path.join("figures", "nasdaq_over_gdp.png"),
    )
    plot_nasdaq_divided_by_m2(
        start_date="1900-01-01",
        path_savefile=os.path.join("figures", "nasdaq_over_m2.png"),
    )
    plot_sandp_divided_by_gdp(
        start_date="1900-01-01",
        path_savefile=os.path.join("figures", "sp500_over_gdp.png"),
    )
    plot_sandp_divided_by_m2(
        start_date="1900-01-01",
        path_savefile=os.path.join("figures", "sp500_over_m2.png"),
    )

    # ###################
    # # plot every month #
    # ###################
    plot_SP500_JOLT(
        start_date="1900-01-01", path_savefile=os.path.join("figures", "SP500_JOLT.png")
    )
    plot_correlation(
        "sectors",
        start_date="2024-01-01",
        path_savefile=os.path.join("figures", "correlation_sectors.png"),
    )
    plot_sort_by_return(
        "selected_etfs",
        "2025-01-01",
        path_savefile=os.path.join("figures", "sorted_etf.png"),
    )
    plot_sort_by_return(
        "sectors",
        "2025-01-01",
        path_savefile=os.path.join("figures", "sorted_sector.png"),
    )

    plot_sort_by_return(
        "index",
        "2025-01-01",
        path_savefile=os.path.join("figures", "sorted_index.png"),
    )
    plot_return_by_group(
        "x1_x3",
        start_date="1900-01-01",
        path_savefile=os.path.join("figures", "return_x1_x3.png"),
    )

    ##############
    # plot daily #
    ##############
    # plot_return_with_ma(
    #     "leverage",
    #     start_date="1900-01-01",
    #     path_savefile=os.path.join("figures", "return_leverage.png"),
    # )

    # plot_series("leverage", start_date="1900-04-01", savedir="figures")
    # plot_series("basic_index", start_date="1900-04-01", savedir="figures")

    plot_pdf("watchlist", start_date="1900-04-01", savedir="figures")
    exit(1)
    # plot_pdf("leverage", start_date="1900-04-01", savedir="figures")
    plot_pdf("basic_index", start_date="1900-04-01", savedir="figures")
    plot_pdf("index", start_date="1900-04-01", savedir="verify_normal")
    plot_pdf("sectors", start_date="1900-04-01", savedir="verify_normal")
    plot_pdf("selected_etfs", start_date="1900-04-01", savedir="verify_normal")
    plot_pdf("s&p500", start_date="1900-04-01", savedir="verify_normal")
