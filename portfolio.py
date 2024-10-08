from utils import (
    generate_portfolio_via_sampling,
    generate_portfolio_via_efficient_frontier,
    compute_budge,
)

if __name__ == "__main__":
    # generate_portfolio_via_sampling(start_date="2024-01-01", end_date="2024-08-30")
    generate_portfolio_via_efficient_frontier(
        start_date="2024-01-01", end_date="2024-08-30"
    )
    # print(compute_budge(100_000))
