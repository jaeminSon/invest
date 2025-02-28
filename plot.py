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
    plot_price_ratio,
    plot_density_function,
)

if __name__ == "__main__":
    ########################
    # for research purpose #
    ########################
    # plot_kelly("p_win", os.path.join("figures", "kelly_pwin.png"))
    # plot_kelly("profit", os.path.join("figures", "kelly_profit.png"))
    # plot_kelly("loss", os.path.join("figures", "kelly_loss.png"))

    ##################
    # plot quarterly #
    ##################
    # plot_nasdaq_divided_by_gdp(
    #     start_date="1900-01-01",
    #     path_savefile=os.path.join("figures", "nasdaq_over_gdp.png"),
    # )
    # plot_nasdaq_divided_by_m2(
    #     start_date="1900-01-01",
    #     path_savefile=os.path.join("figures", "nasdaq_over_m2.png"),
    # )
    # plot_sandp_divided_by_gdp(
    #     start_date="1900-01-01",
    #     path_savefile=os.path.join("figures", "sp500_over_gdp.png"),
    # )
    # plot_sandp_divided_by_m2(
    #     start_date="1900-01-01",
    #     path_savefile=os.path.join("figures", "sp500_over_m2.png"),
    # )

    ###################
    # plot every month #
    ###################
    # plot_SP500_JOLT(
    #     start_date="1900-01-01", path_savefile=os.path.join("figures", "SP500_JOLT.png")
    # )
    # plot_correlation(
    #     "sectors",
    #     start_date="2024-01-01",
    #     path_savefile=os.path.join("figures", "correlation_sectors.png"),
    # )
    # plot_sort_by_return(
    #     "selected_etfs",
    #     "2025-01-01",
    #     path_savefile=os.path.join("figures", "sorted_etf.png"),
    # )

    ###################
    # plot every week #
    ###################
    # plot_return_by_group(
    #     "sectors",
    #     start_date="2025-01-01",
    #     path_savefile=os.path.join("figures", "return_sectors.png"),
    # )
    # plot_return_by_group(
    #     "index",
    #     start_date="2025-01-01",
    #     path_savefile=os.path.join("figures", "return_index.png"),
    # )
    # plot_return_by_group(
    #     "x1_x3",
    #     start_date="1900-01-01",
    #     path_savefile=os.path.join("figures", "return_x1_x3.png"),
    # )

    ##############
    # plot daily #
    ##############
    # plot_return_with_ma(
    #     "leverage",
    #     start_date="1900-01-01",
    #     path_savefile=os.path.join("figures", "return_leverage.png"),
    # )

    plot_density_function("leverage", start_date="1900-04-01", savedir="figures")
    plot_density_function(
        "sp500", start_date="1900-04-01", savedir="figures", domain=[0.6, 1.25]
    )

    # plot_price_ratio("leverage", start_date="1900-04-01", savedir="figures")
    # plot_price_ratio("sp500", start_date="1900-04-01", savedir="figures")
