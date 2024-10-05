from utils import (
    download_assets,
    optimize_asset_portfolio,
    write_portfolio,
    read_portfolio,
    plot_portfolio,
    plot_return_portfolio,
    compute_budge,
)

if __name__ == "__main__":
    # df = download_assets(start_date="2024-09-01")
    # tickers, optimal_w, optimal_r, optimal_v = optimize_asset_portfolio(df)
    # write_portfolio(tickers, optimal_w, optimal_r, optimal_v)
    # plot_portfolio(start_date="2024-09-01")
    # plot_return_portfolio(start_date="2024-09-01")
    print(compute_budge(100_000))
