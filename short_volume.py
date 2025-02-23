import os
import requests
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import yfinance as yf

DIR_SHORT_VOLUME = "short_volume"


def overlap_price_shortvolume(ticker: str, start_date: str, end_date: str):
    start_date = datetime.strptime(start_date, "%Y%m%d")
    end_date = datetime.strptime(end_date, "%Y%m%d")

    stock_data = yf.download(ticker, start=start_date, end=end_date)
    list_date_price = [e.strftime("%Y%m%d") for e in stock_data["Close"].keys()]
    list_close_price = list(stock_data["Close"].values)

    list_ratio = []
    list_date = []
    for date in list_date_price:
        path_short_volume = os.path.join(DIR_SHORT_VOLUME, f"{date}.txt")
        if not os.path.exists(path_short_volume):
            list_ratio.append(-1)
            list_date.append(date)
            continue

        with open(path_short_volume) as f:
            lines = f.readlines()

        for line in lines:
            chunks = line.split("|")
            if chunks[1].lower() == ticker.lower():
                short_volume = chunks[2]
                total_volume = chunks[4]
                ratio = 1.0 * float(short_volume) / float(total_volume)
                break

        list_ratio.append(ratio)
        list_date.append(date)

    fig, ax1 = plt.subplots()

    ax1.plot(list_date_price, list_close_price, "b-")
    ax1.set_xlabel("dates")
    ax1.set_ylabel("price", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    xtick_positions = [5 * i for i in range(len(list_date_price) // 5)]
    ax1.set_xticks(xtick_positions)
    ax1.set_xticklabels(xtick_positions, rotation=90)
    ax1.set_xticklabels([list_date_price[i] for i in xtick_positions], rotation=90)

    ax2 = ax1.twinx()
    ax2.plot(list_date, list_ratio, "r-")
    ax2.set_ylabel("short ratio (short volume / entire volume)", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    plt.savefig("DJT.png", bbox_inches="tight")


def get_date_list(start_date: datetime, end_date: datetime):
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)
    return date_list


def save_short_volume(start_date: str, end_date: str, savedir: str):
    start_date = datetime.strptime(start_date, "%Y%m%d")
    end_date = datetime.strptime(end_date, "%Y%m%d")
    list_date_str = get_date_list(start_date, end_date)

    os.makedirs(savedir, exist_ok=True)

    for date_str in list_date_str:
        path_save = os.path.join(savedir, f"{date_str}.txt")
        if os.path.exists(path_save):
            continue

        response = requests.get(
            f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date_str}.txt"
        )

        if response.status_code == 200:
            with open(path_save, "w") as f:
                f.write(response.text)


def read_json(path):
    with open(path, "r") as f:
        content = json.load(f)
    return content


if __name__ == "__main__":
    start_date = "20240327"
    today = datetime.today().strftime("%Y%m%d")
    save_short_volume(start_date=start_date, end_date=today, savedir=DIR_SHORT_VOLUME)

    overlap_price_shortvolume("DJT", start_date, today)
