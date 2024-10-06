from utils import (
    plot_kelly_2d,
    plot_return_by_sector,
    plot_return_index,
    plot_portfolio,
    plot_correlation,
    plot_return_measure,
    plot_covariance,
    plot_soaring_stocks,
)

if __name__ == "__main__":
    plot_kelly_2d()
    plot_return_by_sector(start_date="2024-01-01")
    plot_return_index(start_date="2024-01-01")
    plot_portfolio(start_date="2024-01-01")
    plot_correlation(start_date="2024-01-01", target="sectors")
    plot_covariance(start_date="2024-01-01", target="sectors")
    plot_correlation(start_date="2024-01-01", target="assets")
    plot_covariance(start_date="2024-01-01", target="assets")
    plot_return_measure(start_date="2024-01-01")
    plot_soaring_stocks()
