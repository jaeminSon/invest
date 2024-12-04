from utils import (
    write_new_portfolio,
    compute_budge,
    compute_win_rates_assets,
)

if __name__ == "__main__":
    print("win rate: ", compute_win_rates_assets("1900-01-01"))
    write_new_portfolio(rebalacing_period=10, path_savefile="portfolio.json")
    # print(
    #     compute_budge(total_budget=70_000_000 / 1400, path_portfolio="portfolio.json")
    # )
