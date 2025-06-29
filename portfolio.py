from gauger import (
    warnings,
    compute_valuation,
    compute_budge,
    bet_ratios_by_group,
    sort_valuation,
    plot_closing_prices_for_tickers
)

if __name__ == "__main__":
    compute_valuation("tickers/sp500.json", "sp500_valuation.json")
    compute_valuation("tickers/selected_etfs.json", "etf_valuation.json")
    sort_valuation(["sp500_valuation.json", "etf_valuation.json"], path_save="rank_20250629.json")
    import json
    plot_closing_prices_for_tickers([e["ticker"] for e in json.load(open("rank_20250629.json"))[-50:]],
                                    savedir="20250629")
    exit(1)
    # print(warnings())
    print("(profit=0.1, loss=0.5)")
    results = bet_ratios_by_group("watchlist", "1900-01-01")
    for ticker, bet_ratios in results.items():
        print(ticker, bet_ratios)

    exit(1)
    print(compute_budge(total_budget=5_000_000 /
          1450, path_portfolio="portfolio.json"))
