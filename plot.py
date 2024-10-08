from utils import (
    plot_kelly_2d,
    plot_return_by_sector,
    plot_return_by_asset,
    plot_return_index,
    plot_return_leverage_with_ma,
    plot_return_portfolio_stocks,
    plot_return_measure,
    plot_return_leverage,
    plot_portfolio_via_sampling,
    plot_portfolio_via_efficient_frontier,
    plot_correlation,
    plot_soaring_stocks,
    plot_backtest,
    plot_periodic_update_backtest,
    plot_rebalancing_backtest,
)

if __name__ == "__main__":
    # plot_kelly_2d()
    # plot_return_leverage(start_date="2020-01-01")
    # plot_return_leverage_with_ma(start_date="2020-01-01")
    # plot_return_by_sector(start_date="2000-01-01")
    # plot_return_by_asset(start_date="2000-01-01")
    # plot_return_index(start_date="2024-01-01")
    # plot_return_portfolio_stocks(start_date="2024-01-01")
    # plot_portfolio_via_sampling(start_date="2024-01-01")
    # plot_portfolio_via_efficient_frontier(start_date="2024-01-01")
    # plot_correlation(start_date="2024-01-01", target="sectors")
    # plot_correlation(start_date="2024-01-01", target="assets")
    # plot_return_measure(start_date="2024-01-01")
    # plot_soaring_stocks()
    # plot_backtest(start_date="2024-09-01", end_date="2024-10-01")

    # plot_periodic_update_backtest(start_date="2020-01-01", end_date="2024-10-04", method="efficient_frontier")

    # plot_periodic_update_backtest(start_date="2020-01-01", end_date="2024-10-04", method="sampling")

    # plot_periodic_update_backtest(start_date="2020-01-01", end_date="2024-10-04", method="equal_weight")

    plot_rebalancing_backtest(start_date="2020-01-01", end_date="2024-10-04")
