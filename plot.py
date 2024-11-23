from utils import (
    plot_kelly_2d,
    plot_return_sector,
    plot_return_etf,
    plot_return_index,
    plot_return_leverage_with_ma,
    plot_return_portfolio_stocks,
    plot_return_measure,
    plot_return_leverage,
    plot_correlation,
    plot_soaring_stocks,
    plot_rebalancing_backtest,
    plot_portfolio_via_sampling,
    plot_sandp_divided_by_m2,
    plot_nasdaq_divided_by_m2,
    plot_nasdaq_divided_by_gdp,
    plot_sandp_correlation_by_date,
    plot_SP500_MOVE,
)

if __name__ == "__main__":
    ########################
    # for research purpose #
    ########################
    # plot_kelly_2d()
    # plot_SP500_MOVE(start_date="1900-01-01")
    # plot_portfolio_via_sampling("2012-01-01")
    # plot_return_measure(start_date="2024-01-01")
    # plot_return_index(start_date="2024-01-01")
    # plot_return_leverage(start_date="2020-01-01")
    # plot_return_portfolio_stocks(start_date="2024-01-01")
    # plot_soaring_stocks(top_k=10, date_back=10)
    # plot_sandp_correlation_by_date("1900-01-01")

    ##################
    # plot quarterly #
    ##################
    plot_correlation(start_date="2024-01-01", target="sectors")
    plot_correlation(start_date="2012-01-01", target="assets")
    plot_return_sector(start_date="2000-01-01")
    plot_return_etf(start_date="2000-01-01")

    ###################
    # plot every week #
    ###################
    plot_return_leverage_with_ma(start_date="2012-01-01")
    plot_nasdaq_divided_by_gdp(start_date="1900-01-01")
    plot_nasdaq_divided_by_m2(start_date="1900-01-01")
    plot_sandp_divided_by_m2(start_date="1900-01-01")

    ############
    # backtest #
    ############
    plot_rebalancing_backtest(
        start_date="2024-01-01",
        end_date="2024-11-19",
        path_savefile="backtest_from_20240101.png",
    )
    plot_rebalancing_backtest(
        start_date="2022-01-01",
        end_date="2024-11-19",
        path_savefile="backtest_from_20220101.png",
    )
    plot_rebalancing_backtest(
        start_date="2020-03-01",
        end_date="2024-11-19",
        path_savefile="backtest_from_20200301.png",
    )
    plot_rebalancing_backtest(
        start_date="2018-06-01",
        end_date="2024-11-19",
        path_savefile="backtest_from_20180601.png",
    )
    plot_rebalancing_backtest(
        start_date="2016-01-01",
        end_date="2024-01-01",
        path_savefile="backtest_from_20160101.png",
    )
    plot_rebalancing_backtest(
        start_date="2014-01-01",
        end_date="2024-11-19",
        path_savefile="backtest_from_20140101.png",
    )
