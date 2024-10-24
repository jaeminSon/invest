from utils import (
    plot_kelly_2d,
    plot_return_by_sector,
    plot_return_by_asset,
    plot_return_index,
    plot_return_leverage_with_ma,
    plot_return_portfolio_stocks,
    plot_return_measure,
    plot_return_leverage,
    plot_correlation,
    plot_soaring_stocks,
    plot_rebalancing_backtest,
    plot_portfolio_via_sampling,
)

if __name__ == "__main__":
    plot_kelly_2d()
    plot_portfolio_via_sampling("2012-01-01")
    plot_return_leverage(start_date="2020-01-01")
    plot_return_leverage_with_ma(start_date="2020-01-01")
    plot_return_by_sector(start_date="2000-01-01")
    plot_return_by_asset(start_date="2000-01-01")
    plot_return_index(start_date="2024-01-01")
    plot_return_portfolio_stocks(start_date="2024-01-01")
    plot_correlation(start_date="2024-01-01", target="sectors")
    plot_correlation(start_date="2012-01-01", target="assets")
    plot_return_measure(start_date="2024-01-01")
    plot_soaring_stocks()

    plot_rebalancing_backtest(
        start_date="2024-01-01",
        end_date="2024-10-20",
        path_savefile="backtest_from_20240101.png",
    )
    plot_rebalancing_backtest(
        start_date="2020-01-01",
        end_date="2024-01-01",
        path_savefile="backtest_from_20200101.png",
    )
    plot_rebalancing_backtest(
        start_date="2016-01-01",
        end_date="2020-01-01",
        path_savefile="backtest_from_20160101.png",
    )
    plot_rebalancing_backtest(
        start_date="2014-01-01",
        end_date="2016-01-01",
        path_savefile="backtest_from_20140101.png",
    )
    plot_rebalancing_backtest(
        start_date="2014-01-01",
        end_date="2024-10-20",
        path_savefile="backtest_from_2014_to_2024.png",
    )