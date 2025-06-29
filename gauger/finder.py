import json
import numpy as np
from typing import List, Dict

from .yahoo_finance import download
from .portfolio import divide_by_rolling_ma


def percentile(values: List[float], index_target: int) -> float:
    target_value = values[index_target]
    return np.mean(np.array(values) < target_value) * 100


def compute_p_ratio(path_json: str, windows: List[int] = [20, 50, 100, 200]) -> Dict:
    """
    Compute price/MA ratio for each ticker in the given JSON file for multiple windows.
    Args:
        path_json: path to JSON file with [[ticker, name], ...]
        windows: list of window sizes for moving average
    Returns:
        Dict: {ticker: {"name": name, "p_ratio": {window: value, ...}}}
    """
    with open(path_json, 'r') as f:
        ticker_name_list = json.load(f)

    results = {}
    for ticker, name in ticker_name_list:
        p_ratio_dict = {}
        try:
            df = download(ticker, "1900-01-01", None)
            for window in windows:
                if len(df['Close'].dropna()) > window:
                    p_ratio_df = divide_by_rolling_ma(df['Close'], window)
                    p_ratios = p_ratio_df.iloc[:, 0].tolist()
                    p = percentile(p_ratios, len(p_ratios) - 1)
                    if not p_ratio_df.empty:
                        p_ratio_dict[window] = p
                else:
                    p_ratio_dict[window] = 101 # infinity
        except Exception as e:
            continue
        if p_ratio_dict:
            results[ticker] = {"name": name, "p_ratio": p_ratio_dict}
            print(name)

    return results


def compute_valuation(path_json: str, save_path: str):
    results = compute_p_ratio(path_json)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)


def sort_valuation(json_paths: list, windows: list = [20, 50, 100, 200], path_save: str = None):
    """
    Reads the given JSON files, computes a custom score for each ticker using:
    (ma=20's percentile + ma=50's percentile) - (ma=100's percentile + ma=200's percentile),
    sorts by this value, and prints/saves the results.
    Args:
        json_paths: list of JSON file paths to read (e.g., ["etf_valuation.json", "sp500_valuation.json"])
        windows: list of moving average windows to use (should include 20, 50, 100, 200)
        path_save: path to save the sorted results as JSON (optional)
    """
    import json
    all_results = {}
    for path in json_paths:
        with open(path, 'r') as f:
            data = json.load(f)
            all_results.update(data)

    # Compute the custom score for each ticker
    ticker2score = {}
    for ticker, info in all_results.items():
        p = info["p_ratio"]
        # Try both int and str keys for compatibility
        p20 = p.get(20) if 20 in p else p.get("20")
        p50 = p.get(50) if 50 in p else p.get("50")
        p100 = p.get(100) if 100 in p else p.get("100")
        p200 = p.get(200) if 200 in p else p.get("200")
        if None not in (p20, p50, p100, p200):
            score = (p20 + p50) - (p100 + p200)
            ticker2score[ticker] = score
        else:
            ticker2score[ticker] = float('inf')  # Put incomplete data at the end

    # Sort tickers by the custom score ascending
    sorted_tickers = sorted(ticker2score.items(), key=lambda x: x[1])

    sorted_results = []
    for ticker, score in sorted_tickers:
        sorted_results.append({
            "ticker": ticker,
            "score": score,
            "info": all_results[ticker]
        })

    # Print results
    print("====================\nSorted by Custom Score (ascending):\n====================")
    for entry in sorted_results:
        print(f"Ticker: {entry['ticker']}, Score: {entry['score']}, Info: {json.dumps(entry['info'], indent=2)}\n")

    if path_save:
        with open(path_save, 'w') as f:
            json.dump(sorted_results, f, indent=2)
