from utils import (
    write_new_portfolio,
    compute_budge,
    compute_bet_ratio,
)

if __name__ == "__main__":
    print("(prob_win, bet_ratio): ", compute_bet_ratio("1900-01-01"))
    exit(1)
    write_new_portfolio(rebalacing_period=10, path_savefile="portfolio.json")
    print(compute_budge(total_budget=5_000_000 / 1450, path_portfolio="portfolio.json"))
