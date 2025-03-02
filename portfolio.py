from gauger import (
    warnings,
    find_undervalued,
    compute_budge,
    bet_ratios_by_group,
)

if __name__ == "__main__":
    print(
        find_undervalued(
            "s&p500", winrate_threshold=0.95, start_date="1900-01-01", window=100
        )
    )
    # print(warnings())
    print("(profit=0.1, loss=0.5)", bet_ratios_by_group("leverage", "1900-01-01"))
    exit(1)
    print(compute_budge(total_budget=5_000_000 / 1450, path_portfolio="portfolio.json"))
