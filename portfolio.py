from utils import (
    write_new_portfolio,
    compute_budge,
)

if __name__ == "__main__":
    write_new_portfolio(rebalacing_period=10, path_savefile="portfolio.json")
    print(
        compute_budge(total_budget=70_000_000 / 1400, path_portfolio="portfolio.json")
    )
    # print(compute_budge(100_000))
