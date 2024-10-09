from utils import (
    write_new_portfolio,
    compute_budge,
)

if __name__ == "__main__":
    write_new_portfolio(rebalacing_period=10, path_savefile="portfolio_10.json")
    write_new_portfolio(rebalacing_period=20, path_savefile="portfolio_20.json")
    write_new_portfolio(rebalacing_period=50, path_savefile="portfolio_50.json")
    write_new_portfolio(rebalacing_period=100, path_savefile="portfolio_100.json")
    # print(compute_budge(100_000))
