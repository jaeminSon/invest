from typing import Tuple, List, Dict, Iterable, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import os
import ssl
import shutil
import json
from io import BytesIO

import numpy as np
import scipy
import scipy.stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

import xml.etree.ElementTree as ET
import urllib.request as url_request
import urllib.parse as url_parse
import urllib.error as url_error
import imageio
from PIL import Image

from hmmlearn import hmm
from lppls import lppls

############
### FRED ###
############
urlopen = url_request.urlopen
quote_plus = url_parse.quote_plus
urlencode = url_parse.urlencode
HTTPError = url_error.HTTPError


class Fred:
    earliest_realtime_start = "1776-07-04"
    latest_realtime_end = "9999-12-31"
    nan_char = "."
    max_results_per_request = 1000
    root_url = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key=None, api_key_file=None, proxies=None):
        """
        Initialize the Fred class that provides useful functions to query the Fred dataset. You need to specify a valid
        API key in one of 3 ways: pass the string via api_key, or set api_key_file to a file with the api key in the
        first line, or set the environment variable 'FRED_API_KEY' to the value of your api key.

        Parameters
        ----------
        api_key : str
            API key. A free api key can be obtained on the Fred website at http://research.stlouisfed.org/fred2/.
        api_key_file : str
            Path to a file containing the api key.
        proxies : dict
            Proxies specifications: a dictionary mapping protocol names (e.g. 'http', 'https') to proxy URLs. If not provided, environment variables 'HTTP_PROXY', 'HTTPS_PROXY' are used.

        """
        self.api_key = None
        if api_key is not None:
            self.api_key = api_key
        elif api_key_file is not None:
            f = open(api_key_file, "r")
            self.api_key = f.readline().strip()
            f.close()
        else:
            self.api_key = os.environ.get("FRED_API_KEY")

        if self.api_key is None:
            import textwrap

            raise ValueError(
                textwrap.dedent(
                    """\
                    You need to set a valid API key. You can set it in 3 ways:
                    pass the string with api_key, or set api_key_file to a
                    file with the api key in the first line, or set the
                    environment variable 'FRED_API_KEY' to the value of your
                    api key. You can sign up for a free api key on the Fred
                    website at http://research.stlouisfed.org/fred2/"""
                )
            )

        if not proxies:
            http_proxy, https_proxy = os.getenv("HTTP_PROXY"), os.getenv("HTTPS_PROXY")
            if http_proxy or https_proxy:
                proxies = {"http": http_proxy, "https": https_proxy}

        self.proxies = proxies

        if self.proxies:
            opener = url_request.build_opener(url_request.ProxyHandler(self.proxies))
            url_request.install_opener(opener)

    def __fetch_data(self, url):
        """
        helper function for fetching data given a request URL
        """
        url += "&api_key=" + self.api_key
        try:
            response = urlopen(url)
            root = ET.fromstring(response.read())
        except HTTPError as exc:
            root = ET.fromstring(exc.read())
            raise ValueError(root.get("message"))
        return root

    def _parse(self, date_str, format="%Y-%m-%d"):
        """
        helper function for parsing FRED date string into datetime
        """
        rv = pd.to_datetime(date_str, format=format)
        if hasattr(rv, "to_pydatetime"):
            rv = rv.to_pydatetime()
        return rv

    def get_series_info(self, series_id):
        """
        Get information about a series such as its title, frequency, observation start/end dates, units, notes, etc.

        Parameters
        ----------
        series_id : str
            Fred series id such as 'CPIAUCSL'

        Returns
        -------
        info : Series
            a pandas Series containing information about the Fred series
        """
        url = "%s/series?series_id=%s" % (self.root_url, series_id)
        root = self.__fetch_data(url)
        if root is None or not len(root):
            raise ValueError("No info exists for series id: " + series_id)
        info = pd.Series(list(root)[0].attrib)
        return info

    def get_series(
        self, series_id, observation_start=None, observation_end=None, **kwargs
    ):
        """
        Get data for a Fred series id. This fetches the latest known data, and is equivalent to get_series_latest_release()

        Parameters
        ----------
        series_id : str
            Fred series id such as 'CPIAUCSL'
        observation_start : datetime or datetime-like str such as '7/1/2014', optional
            earliest observation date
        observation_end : datetime or datetime-like str such as '7/1/2014', optional
            latest observation date
        kwargs : additional parameters
            Any additional parameters supported by FRED. You can see https://api.stlouisfed.org/docs/fred/series_observations.html for the full list

        Returns
        -------
        data : Series
            a Series where each index is the observation date and the value is the data for the Fred series
        """
        url = "%s/series/observations?series_id=%s" % (self.root_url, series_id)
        if observation_start is not None:
            observation_start = pd.to_datetime(observation_start, errors="raise")
            url += "&observation_start=" + observation_start.strftime("%Y-%m-%d")
        if observation_end is not None:
            observation_end = pd.to_datetime(observation_end, errors="raise")
            url += "&observation_end=" + observation_end.strftime("%Y-%m-%d")
        if kwargs.keys():
            url += "&" + urlencode(kwargs)
        root = self.__fetch_data(url)
        if root is None:
            raise ValueError("No data exists for series id: " + series_id)
        data = {}
        for child in root:
            val = child.get("value")
            if val == self.nan_char:
                val = float("NaN")
            else:
                val = float(val)
            data[self._parse(child.get("date"))] = val
        return pd.Series(data)

    def get_series_latest_release(self, series_id):
        """
        Get data for a Fred series id. This fetches the latest known data, and is equivalent to get_series()

        Parameters
        ----------
        series_id : str
            Fred series id such as 'CPIAUCSL'

        Returns
        -------
        info : Series
            a Series where each index is the observation date and the value is the data for the Fred series
        """
        return self.get_series(series_id)

    def get_series_first_release(self, series_id):
        """
        Get first-release data for a Fred series id. This ignores any revision to the data series. For instance,
        The US GDP for Q1 2014 was first released to be 17149.6, and then later revised to 17101.3, and 17016.0.
        This will ignore revisions after the first release.

        Parameters
        ----------
        series_id : str
            Fred series id such as 'GDP'

        Returns
        -------
        data : Series
            a Series where each index is the observation date and the value is the data for the Fred series
        """
        df = self.get_series_all_releases(series_id)
        first_release = df.groupby("date").head(1)
        data = first_release.set_index("date")["value"]
        return data

    def get_series_as_of_date(self, series_id, as_of_date):
        """
        Get latest data for a Fred series id as known on a particular date. This includes any revision to the data series
        before or on as_of_date, but ignores any revision on dates after as_of_date.

        Parameters
        ----------
        series_id : str
            Fred series id such as 'GDP'
        as_of_date : datetime, or datetime-like str such as '10/25/2014'
            Include data revisions on or before this date, and ignore revisions afterwards

        Returns
        -------
        data : Series
            a Series where each index is the observation date and the value is the data for the Fred series
        """
        as_of_date = pd.to_datetime(as_of_date)
        df = self.get_series_all_releases(series_id)
        data = df[df["realtime_start"] <= as_of_date]
        return data

    def get_series_all_releases(
        self, series_id, realtime_start=None, realtime_end=None
    ):
        """
        Get all data for a Fred series id including first releases and all revisions. This returns a DataFrame
        with three columns: 'date', 'realtime_start', and 'value'. For instance, the US GDP for Q4 2013 was first released
        to be 17102.5 on 2014-01-30, and then revised to 17080.7 on 2014-02-28, and then revised to 17089.6 on
        2014-03-27. You will therefore get three rows with the same 'date' (observation date) of 2013-10-01 but three
        different 'realtime_start' of 2014-01-30, 2014-02-28, and 2014-03-27 with corresponding 'value' of 17102.5, 17080.7
        and 17089.6

        Parameters
        ----------
        series_id : str
            Fred series id such as 'GDP'
        realtime_start : str, optional
            specifies the realtime_start value used in the query, defaults to the earliest possible start date allowed by Fred
        realtime_end : str, optional
            specifies the realtime_end value used in the query, defaults to the latest possible end date allowed by Fred

        Returns
        -------
        data : DataFrame
            a DataFrame with columns 'date', 'realtime_start' and 'value' where 'date' is the observation period and 'realtime_start'
            is when the corresponding value (either first release or revision) is reported.
        """
        if realtime_start is None:
            realtime_start = self.earliest_realtime_start
        if realtime_end is None:
            realtime_end = self.latest_realtime_end
        url = (
            "%s/series/observations?series_id=%s&realtime_start=%s&realtime_end=%s"
            % (self.root_url, series_id, realtime_start, realtime_end)
        )
        root = self.__fetch_data(url)
        if root is None:
            raise ValueError("No data exists for series id: " + series_id)
        data = {}
        i = 0
        for child in root:
            val = child.get("value")
            if val == self.nan_char:
                val = float("NaN")
            else:
                val = float(val)
            realtime_start = self._parse(child.get("realtime_start"))
            # realtime_end = self._parse(child.get('realtime_end'))
            date = self._parse(child.get("date"))

            data[i] = {
                "realtime_start": realtime_start,
                # 'realtime_end': realtime_end,
                "date": date,
                "value": val,
            }
            i += 1
        data = pd.DataFrame(data).T
        return data

    def get_series_vintage_dates(self, series_id):
        """
        Get a list of vintage dates for a series. Vintage dates are the dates in history when a
        series' data values were revised or new data values were released.

        Parameters
        ----------
        series_id : str
            Fred series id such as 'CPIAUCSL'

        Returns
        -------
        dates : list
            list of vintage dates
        """
        url = "%s/series/vintagedates?series_id=%s" % (self.root_url, series_id)
        root = self.__fetch_data(url)
        if root is None:
            raise ValueError("No vintage date exists for series id: " + series_id)
        dates = []
        for child in root:
            dates.append(self._parse(child.text))
        return dates

    def __do_series_search(self, url):
        """
        helper function for making one HTTP request for data, and parsing the returned results into a DataFrame
        """
        root = self.__fetch_data(url)

        series_ids = []
        data = {}

        num_results_returned = 0  # number of results returned in this HTTP request
        num_results_total = int(
            root.get("count")
        )  # total number of results, this can be larger than number of results returned
        for child in root:
            num_results_returned += 1
            series_id = child.get("id")
            series_ids.append(series_id)
            data[series_id] = {"id": series_id}
            fields = [
                "realtime_start",
                "realtime_end",
                "title",
                "observation_start",
                "observation_end",
                "frequency",
                "frequency_short",
                "units",
                "units_short",
                "seasonal_adjustment",
                "seasonal_adjustment_short",
                "last_updated",
                "popularity",
                "notes",
            ]
            for field in fields:
                data[series_id][field] = child.get(field)

        if num_results_returned > 0:
            data = pd.DataFrame(data, columns=series_ids).T
            # parse datetime columns
            for field in [
                "realtime_start",
                "realtime_end",
                "observation_start",
                "observation_end",
                "last_updated",
            ]:
                data[field] = data[field].apply(self._parse, format=None)
            # set index name
            data.index.name = "series id"
        else:
            data = None
        return data, num_results_total

    def __get_search_results(self, url, limit, order_by, sort_order, filter):
        """
        helper function for getting search results up to specified limit on the number of results. The Fred HTTP API
        truncates to 1000 results per request, so this may issue multiple HTTP requests to obtain more available data.
        """

        order_by_options = [
            "search_rank",
            "series_id",
            "title",
            "units",
            "frequency",
            "seasonal_adjustment",
            "realtime_start",
            "realtime_end",
            "last_updated",
            "observation_start",
            "observation_end",
            "popularity",
        ]
        if order_by is not None:
            if order_by in order_by_options:
                url = url + "&order_by=" + order_by
            else:
                raise ValueError(
                    "%s is not in the valid list of order_by options: %s"
                    % (order_by, str(order_by_options))
                )

        if filter is not None:
            if len(filter) == 2:
                url = url + "&filter_variable=%s&filter_value=%s" % (
                    filter[0],
                    filter[1],
                )
            else:
                raise ValueError(
                    "Filter should be a 2 item tuple like (filter_variable, filter_value)"
                )

        sort_order_options = ["asc", "desc"]
        if sort_order is not None:
            if sort_order in sort_order_options:
                url = url + "&sort_order=" + sort_order
            else:
                raise ValueError(
                    "%s is not in the valid list of sort_order options: %s"
                    % (sort_order, str(sort_order_options))
                )

        data, num_results_total = self.__do_series_search(url)
        if data is None:
            return data

        if limit == 0:
            max_results_needed = num_results_total
        else:
            max_results_needed = limit

        if max_results_needed > self.max_results_per_request:
            for i in range(1, max_results_needed // self.max_results_per_request + 1):
                offset = i * self.max_results_per_request
                next_data, _ = self.__do_series_search(url + "&offset=" + str(offset))
                data = pd.concat([data, next_data])
        return data.head(max_results_needed)

    def search(self, text, limit=1000, order_by=None, sort_order=None, filter=None):
        """
        Do a fulltext search for series in the Fred dataset. Returns information about matching series in a DataFrame.

        Parameters
        ----------
        text : str
            text to do fulltext search on, e.g., 'Real GDP'
        limit : int, optional
            limit the number of results to this value. If limit is 0, it means fetching all results without limit.
        order_by : str, optional
            order the results by a criterion. Valid options are 'search_rank', 'series_id', 'title', 'units', 'frequency',
            'seasonal_adjustment', 'realtime_start', 'realtime_end', 'last_updated', 'observation_start', 'observation_end',
            'popularity'
        sort_order : str, optional
            sort the results by ascending or descending order. Valid options are 'asc' or 'desc'
        filter : tuple, optional
            filters the results. Expects a tuple like (filter_variable, filter_value).
            Valid filter_variable values are 'frequency', 'units', and 'seasonal_adjustment'

        Returns
        -------
        info : DataFrame
            a DataFrame containing information about the matching Fred series
        """
        url = "%s/series/search?search_text=%s&" % (self.root_url, quote_plus(text))
        info = self.__get_search_results(url, limit, order_by, sort_order, filter)
        return info

    def search_by_release(
        self, release_id, limit=0, order_by=None, sort_order=None, filter=None
    ):
        """
        Search for series that belongs to a release id. Returns information about matching series in a DataFrame.

        Parameters
        ----------
        release_id : int
            release id, e.g., 151
        limit : int, optional
            limit the number of results to this value. If limit is 0, it means fetching all results without limit.
        order_by : str, optional
            order the results by a criterion. Valid options are 'search_rank', 'series_id', 'title', 'units', 'frequency',
            'seasonal_adjustment', 'realtime_start', 'realtime_end', 'last_updated', 'observation_start', 'observation_end',
            'popularity'
        sort_order : str, optional
            sort the results by ascending or descending order. Valid options are 'asc' or 'desc'
        filter : tuple, optional
            filters the results. Expects a tuple like (filter_variable, filter_value).
            Valid filter_variable values are 'frequency', 'units', and 'seasonal_adjustment'

        Returns
        -------
        info : DataFrame
            a DataFrame containing information about the matching Fred series
        """
        url = "%s/release/series?release_id=%d" % (self.root_url, release_id)
        info = self.__get_search_results(url, limit, order_by, sort_order, filter)
        if info is None:
            raise ValueError("No series exists for release id: " + str(release_id))
        return info

    def search_by_category(
        self, category_id, limit=0, order_by=None, sort_order=None, filter=None
    ):
        """
        Search for series that belongs to a category id. Returns information about matching series in a DataFrame.

        Parameters
        ----------
        category_id : int
            category id, e.g., 32145
        limit : int, optional
            limit the number of results to this value. If limit is 0, it means fetching all results without limit.
        order_by : str, optional
            order the results by a criterion. Valid options are 'search_rank', 'series_id', 'title', 'units', 'frequency',
            'seasonal_adjustment', 'realtime_start', 'realtime_end', 'last_updated', 'observation_start', 'observation_end',
            'popularity'
        sort_order : str, optional
            sort the results by ascending or descending order. Valid options are 'asc' or 'desc'
        filter : tuple, optional
            filters the results. Expects a tuple like (filter_variable, filter_value).
            Valid filter_variable values are 'frequency', 'units', and 'seasonal_adjustment'

        Returns
        -------
        info : DataFrame
            a DataFrame containing information about the matching Fred series
        """
        url = "%s/category/series?category_id=%d&" % (self.root_url, category_id)
        info = self.__get_search_results(url, limit, order_by, sort_order, filter)
        if info is None:
            raise ValueError("No series exists for category id: " + str(category_id))
        return info


def get_fred_ticker(ticker: str):
    # ticker of https://fred.stlouisfed.org/series/M2SL == M2SL
    if ticker.lower() == "s&p500":
        return "SP500"
    elif ticker.lower() == "nasdaq":
        return "NASDAQCOM"
    elif ticker.lower() == "gdp":
        return "GDP"
    elif ticker.lower() == "2-10-spread":
        return "T10Y2Y"
    elif ticker.lower() == "10y-yield":
        return "DGS10"
    elif ticker.lower() == "m2":
        return "M2SL"
    elif ticker.lower() == "jolt":
        return "JTSJOL"


def get_fred_series(ticker: str, start_date: str, end_date: str):
    fred = Fred(api_key=json.load(open("fredkey.json"))["key"])
    ticker_fred = get_fred_ticker(ticker)
    return fred.get_series(
        ticker_fred, observation_start=start_date, observation_end=end_date
    )


#####################
### yahoo finance ###
#####################
def download_SandP(start_date, end_date: Optional[str] = None) -> pd.DataFrame:
    if end_date is None:
        start_date, end_date = period(start_date)
    tickers = get_ticker("s&p500")
    return download(tickers, start_date, end_date)


def download_assets(start_date, end_date: Optional[str] = None) -> pd.DataFrame:
    if end_date is None:
        start_date, end_date = period(start_date)
    tickers = get_ticker("assets")
    return download(tickers, start_date, end_date)


def download_sectors(start_date, end_date: Optional[str] = None) -> pd.DataFrame:
    if end_date is None:
        start_date, end_date = period(start_date)
    tickers = get_ticker("sectors")
    return download(tickers, start_date, end_date)


def download_selected_etfs(start_date, end_date: Optional[str] = None) -> pd.DataFrame:
    if end_date is None:
        start_date, end_date = period(start_date)
    tickers = get_ticker("selected_etfs")
    return download(tickers, start_date, end_date)


def download_portfolio(start_date, end_date: Optional[str] = None) -> pd.DataFrame:
    pf = read_portfolio()
    tickers = [e[0] for e in pf["weights"] if e[0] != "cash"]

    if end_date is None:
        start_date, end_date = period(start_date)

    df = download(tickers, start_date, end_date)

    return df


def download(symbol, start_date, end_date) -> pd.DataFrame:
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    if len(symbol) == 1:
        stock_data.columns = pd.MultiIndex.from_tuples(
            [(c, symbol[0]) for c in stock_data.columns]
        )

    return stock_data


def period(start_date: Optional[str] = None, date_back: int = None) -> Tuple[str, str]:
    assert not (
        start_date and date_back
    ), "Both start_date and date_back cannot be set at the same time."
    if date_back:
        now = datetime.now()
        start_date = (now - timedelta(days=date_back)).strftime("%Y-%m-%d")
        end_date = now.strftime("%Y-%m-%d")
        return start_date, end_date
    elif start_date:
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
        except:
            raise ValueError(f"Failed to process {start_date}")
        end_date = datetime.now().strftime("%Y-%m-%d")
        return start_date, end_date


def benchmark_return(benchmark: str, start_date, end_date) -> pd.Series:
    try:
        benchmark_ticker = find_ticker(benchmark)
    except:
        benchmark_ticker = benchmark

    df_benchmark = download([benchmark_ticker], start_date, end_date)
    return (df_benchmark["Close"].pct_change().fillna(0) + 1).cumprod().dropna()


def portfolio_return(test_date) -> pd.Series:
    portfolio = read_portfolio()
    portfolio_return = portfolio["cash"] + sum(e[1] for e in portfolio["weights"])
    return pd.Series([portfolio_return], index=[test_date], name="Close")


###########
### I/O ###
###########
def get_ticker(category: str) -> List:
    if category.lower() == "s&p500":
        ssl._create_default_https_context = ssl._create_unverified_context
        tickers = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )[0]
        return tickers.Symbol.to_list()
    elif category.lower() == "sectors":
        return [i[0] for i in json.load(open("sectors.json"))]
    elif category.lower() == "assets":
        return [i[0] for i in json.load(open("assets.json"))]
    elif category.lower() == "selected_etfs":
        return [i[0] for i in json.load(open("selected_etfs.json"))]
    else:
        raise ValueError("Unknown category.")


def name(category: str) -> List:
    if category.lower() == "sectors":
        return [i[1] for i in json.load(open("sectors.json"))]
    elif category.lower() == "assets":
        return [i[1] for i in json.load(open("assets.json"))]
    else:
        raise ValueError("Unknown category.")


def ticker2name(category: str) -> Dict:
    if category.lower() == "sectors":
        return {i[0]: i[1] for i in json.load(open("sectors.json"))}
    elif category.lower() == "assets":
        return {i[0]: i[1] for i in json.load(open("assets.json"))}
    elif category.lower() == "selected_etfs":
        return {i[0]: i[1] for i in json.load(open("selected_etfs.json"))}
    else:
        raise ValueError("Unknown category.")


def portfolio_start_date() -> Dict:
    return json.load(open("portfolio.json"))["date_generated"]


def find_ticker(keyword):
    if keyword == "s&p":
        return "SPY"
    elif keyword == "nasdaq":
        return "QQQ"
    elif keyword == "dow":
        return "DIA"
    elif keyword == "gold":
        return "GLD"
    else:
        raise ValueError(f"Unknown keyword {keyword}")


def write_portfolio(
    date_generated,
    tickers: List[str],
    weight: np.ndarray,
    cash_amount: float,
    path_savefile: str = "portfolio.json",
) -> None:
    assert len(tickers) == len(
        weight
    ), "tickers and weight should have the same length."

    t2n = ticker2name("assets")

    data = {
        "date_generated": date_generated
        if isinstance(date_generated, str)
        else date_generated.strftime("%Y-%m-%d"),
        "cash": cash_amount,
        "weights": [(t, w, t2n[t]) for t, w in zip(tickers, weight)],
    }

    json.dump(data, open(path_savefile, "w"), indent=4)


def read_portfolio(path_portfolio: str = "portfolio.json"):
    return json.load(open(path_portfolio))


def write_new_portfolio(
    rebalacing_period,
    path_savefile: str = "portfolio.json",
    dir_prev_portfolio="prev_portfolio",
):
    start_date, end_date = period(date_back=rebalacing_period)

    if os.path.exists(path_savefile):
        os.makedirs(dir_prev_portfolio, exist_ok=True)
        shutil.copy(
            path_savefile,
            os.path.join(
                dir_prev_portfolio, f"portfolio_{start_date}_to_{end_date}.json"
            ),
        )
    else:
        create_portfolio_file(end_date, path_savefile)

    generate_portfolio_via_rebalancing(
        end_date=datetime.strptime(end_date, "%Y-%m-%d"),
        path_savefile=path_savefile,
    )


#######################
### stock selection ###
#######################
def select_stock(
    yf_df: pd.DataFrame, window: int, top_k: int, key="Close"
) -> List[str]:
    pct_ch = (
        yf_df[key][yf_df[key].columns]
        .rolling(window=window)
        .mean()
        .pct_change(periods=window)
        .dropna(axis=1, how="all")
    )

    # sharpe is a series that maps ticker to Sharpe value
    sharpe = ((pct_ch.iloc[-1] - pct_ch.mean()) / np.sqrt(pct_ch.var())).dropna()
    sharpe.sort_values(ascending=False, inplace=True)

    return list(sharpe.index[:top_k])


def select_stock_varying_windows(yf_df: pd.DataFrame, top_k: int) -> List[str]:
    windows = [3, 5, 10, 20]
    return {w: select_stock(yf_df, w, top_k) for w in windows}


#################
### portfolio ###
#################
def kelly(win_rate: float, net_profit: float, net_loss: float) -> float:
    assert 0 <= win_rate <= 1
    assert net_loss > 0

    if net_profit < 0:
        return 0

    bet = (1.0 * win_rate / (net_loss + 1e-6)) - (
        1.0 * (1 - win_rate) / (net_profit + 1e-6)
    )

    return np.clip(bet, 0, 1)


def all_cash(portfolio: Dict):
    return all(e[1] < 1e-6 for e in portfolio["weights"])


def sell_signal(end_date, yf_df: pd.DataFrame, key="Close"):
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    if yf_df is None:
        df = download_assets(
            start_date=(end_date - timedelta(days=200)).strftime("%Y-%m-%d"),
            end_date=end_date,
        )
    else:
        df = yf_df[
            (yf_df.index >= end_date - timedelta(days=200)) & (yf_df.index <= end_date)
        ]

    df_ma100 = df[key].rolling(window=100).mean().dropna()
    for ticker in df[key].columns:
        if (
            sum(df_ma100[ticker].diff(periods=1)[-5:] <= 0) >= 4
            and sum(df_ma100[ticker].diff(periods=1).diff(periods=1)[-5:] <= 0) >= 4
        ):
            return True

    return False


def buy_signal(end_date, yf_df: pd.DataFrame, key="Close"):
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    if yf_df is None:
        df = download_assets(start_date=asset_open_date, end_date=end_date)
    else:
        df = yf_df[
            (yf_df.index >= end_date - timedelta(days=365)) & (yf_df.index <= end_date)
        ]

    df_ma100 = df[key].rolling(window=100).mean().dropna()
    for ticker in df[key].columns:
        if (
            sum(df_ma100[ticker].diff(periods=1)[-5:] >= 0) >= 3
            and sum(df_ma100[ticker].diff(periods=1).diff(periods=1)[-5:] >= 0) >= 3
        ):
            return True

    return False


def sell(portfolio: Dict, transaction_fee_rate) -> Dict:
    tickers = get_ticker("assets")
    optimal_w = [0] * len(tickers)

    cash_amount = portfolio["cash"]
    t2w = {e[0]: e[1] for e in portfolio["weights"]}
    total = sum(t2w.values()) + cash_amount

    cash = cash_amount + (total - cash_amount) / (1 + transaction_fee_rate)

    return tickers, optimal_w, cash


def compute_percentile_by_percent_change(yf_df: pd.DataFrame, ticker: str) -> float:
    # empirical wining rate of the previous period
    list_prob_win = []
    for window in [10, 20, 50, 100]:
        pct_ch = percent_change(yf_df, periods=window, fillna_zero=False)
        pct_ch_sorted = sorted(np.array(pct_ch[ticker]))
        prob_win = np.sum(pct_ch_sorted <= pct_ch[ticker].iloc[-1]) / len(pct_ch_sorted)
        list_prob_win.append(prob_win)
    return np.mean(list_prob_win)


def compute_trend(yf_df: pd.DataFrame, ticker: str, key="Close"):
    list_confidence = []
    for window in [10, 20, 50, 100]:
        df_ma = yf_df[key].rolling(window=window).mean().dropna()
        confidence = sum(
            np.array(df_ma[ticker].diff(periods=1)[-window:] >= 0)
            * linear_weight(window)
        )
        list_confidence.append(confidence)
    return np.mean(list_confidence)


def linear_weight(length):
    return np.arange(length) / sum(np.arange(length))


def set_target_weights(
    end_date,
    asset_open_date="2010-04-01",
    key="Close",
    yf_df: pd.DataFrame = None,
) -> Dict:
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    if yf_df is None:
        df = download_assets(start_date=asset_open_date, end_date=end_date)
    else:
        df = yf_df[(yf_df.index >= asset_open_date) & (yf_df.index <= end_date)]

    weights = {}
    for ticker in df[key].columns:
        win_rate = compute_win_rate_by_ratio_density_function(df, ticker)
        weight = np.clip(2 * win_rate, 0, 1)  # maintain at half

        trend = compute_trend(df, ticker)

        weights[ticker] = weight * trend

    sum_weights_assets = sum(weights.values())
    if sum_weights_assets > 1:
        weights = {t: 1.0 * w / sum_weights_assets for t, w in weights.items()}
        weights["cash"] = 0
    else:
        weights["cash"] = 1 - sum_weights_assets

    return weights


def generate_initial_portfolio_backtest(end_date, yf_df: pd.DataFrame = None) -> None:
    target_weights = set_target_weights(end_date, yf_df=yf_df)
    tickers = [t for t in target_weights.keys() if t != "cash"]
    write_portfolio(
        end_date,
        tickers,
        [target_weights[t] for t in tickers],
        target_weights["cash"],
        path_savefile="portfolio.json",
    )


def create_portfolio_file(date, path_savefile="portfolio.json"):
    tickers = get_ticker("assets")
    write_portfolio(
        date,
        tickers,
        weight=[0 for t in tickers],
        cash_amount=1,
        path_savefile=path_savefile,
    )


def generate_portfolio_via_rebalancing(
    end_date,
    yf_df: pd.DataFrame = None,
    transaction_fee_rate: float = 0.005,
    path_savefile: str = "portfolio.json",
) -> None:
    portfolio = read_portfolio(path_savefile)

    target_weights = set_target_weights(end_date, yf_df=yf_df)

    (
        tickers,
        optimal_w,
        new_cash_amount,
    ) = optimize_asset_portfolio_via_rebalancing(
        portfolio, target_weights, transaction_fee_rate
    )

    write_portfolio(end_date, tickers, optimal_w, new_cash_amount, path_savefile)


def optimize_asset_portfolio_via_rebalancing(
    portfolio: Dict,
    target_weights: Iterable[float],
    transaction_fee_rate: float,
) -> Tuple:
    assert abs(sum(target_weights.values()) - 1) < 1e-6

    cash_amount = portfolio["cash"]
    t2w = {e[0]: e[1] for e in portfolio["weights"]}
    total = sum(t2w.values()) + cash_amount

    optimal_w = []
    for ticker in t2w.keys():
        new_weight = target_weights[ticker] * total
        if new_weight > t2w[ticker]:
            new_weight = t2w[ticker] + (new_weight - t2w[ticker]) / (
                1 + transaction_fee_rate
            )  # compensate for transaction fee by reducing weight
        optimal_w.append(new_weight)

    new_cash_amount = total * target_weights["cash"]
    if new_cash_amount > cash_amount:
        new_cash_amount = cash_amount + (new_cash_amount - cash_amount) / (
            1 + transaction_fee_rate
        )

    return t2w.keys(), optimal_w, new_cash_amount


def cov_mat_from_pct_ch(pct_ch: pd.DataFrame) -> np.ndarray:
    return np.array(pct_ch.cov())


def compute_return_volatility(
    pct_ch: pd.DataFrame,
) -> Tuple[List, np.ndarray, np.ndarray]:
    tickers = list(pct_ch.columns)
    r = expected_return_from_pct_ch(pct_ch)
    cov = cov_mat_from_pct_ch(pct_ch)

    # check the order of columns
    assert all((pct_ch[t].mean() - r[i]) < 1e-6 for i, t in enumerate(tickers))
    assert all((pct_ch[t].var() - cov[i, i]) < 1e-6 for i, t in enumerate(tickers))

    return tickers, r, cov


def random_portfolio(
    ret: np.ndarray, cov_mat: np.ndarray, N=10_000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(0)

    weights = np.random.random((N, len(ret)))
    weights /= np.sum(weights, axis=1, keepdims=True)

    returns = np.array([w.T @ ret for w in weights])

    volatilities = np.sqrt([w.T @ cov_mat @ w for w in weights])

    return weights, returns, volatilities


def optimize_asset_portfolio_via_sampling(
    df: pd.DataFrame, key="Close", return_samples: bool = False
) -> Tuple:
    pct_ch = percent_change(df, key)
    pct_ch_filted = pct_ch
    # pct_ch_filted = filter_assets(pct_ch)

    tickers, r, cov = compute_return_volatility(pct_ch_filted)

    random_w, random_r, random_v = random_portfolio(r, cov)

    random_sharpe = random_r / random_v
    index_opt = np.argmax(random_sharpe)
    optimal_w = random_w[index_opt]
    optimal_v = random_v[index_opt]
    optimal_r = random_r[index_opt]
    assert abs(optimal_w.T @ r - optimal_r) < 1e-6
    assert abs(np.sqrt(optimal_w.T @ cov @ optimal_w) - optimal_v) < 1e-6

    if return_samples:
        return (
            tickers,
            optimal_w,
            optimal_r,
            optimal_v,
            random_r,
            random_v,
            random_sharpe,
        )
    else:
        return tickers, optimal_w, optimal_r, optimal_v


def compute_budge(total_budget: int, path_portfolio: str = "portfolio.json"):
    """
    Args:
        total_budget: dollars
    Return:
        list of money for each weight
    """
    pf = read_portfolio(path_portfolio)
    return {e[0]: e[1] * total_budget for e in pf["weights"]}


def need_to_update_portfolio(date: datetime.date, rebalacing_period: int):
    portfolio = read_portfolio()
    stale = (
        date - datetime.strptime(portfolio["date_generated"], "%Y-%m-%d")
    ).days >= rebalacing_period
    return stale or all_cash(portfolio)


def compute_win_rate_by_ratio_density_function(
    yf_df: pd.DataFrame, ticker: str
) -> float:
    list_win_rate = []
    for window in [20, 50, 100, 200]:
        df_mean = yf_df["Close"][ticker].to_frame("price_ratio")
        df_mean["price_ratio"] /= df_mean["price_ratio"].rolling(window=window).mean()
        df_mean.dropna(inplace=True)

        p = density_function(list(df_mean["price_ratio"]))

        list_win_rate.append(
            win_rate_given_density_function(p, df_mean["price_ratio"].iloc[-1])
        )

    return np.mean(list_win_rate)


def compute_win_rates_assets(
    start_date: str,
    end_date: str = None,
    key: str = "Close",
):
    if end_date is None:
        start_date, end_date = period(start_date)

    tickers = get_ticker("assets")
    df = download(tickers, start_date, end_date)

    win_rates = {}
    for ticker in tickers:
        list_win_rate = []
        for window in [20, 50, 100, 200]:
            df_mean = df[key][ticker].to_frame("price_ratio")
            df_mean["price_ratio"] /= (
                df_mean["price_ratio"].rolling(window=window).mean()
            )
            df_mean.dropna(inplace=True)

            p = density_function(list(df_mean["price_ratio"]))

            list_win_rate.append(
                win_rate_given_density_function(p, df_mean["price_ratio"].iloc[-1])
            )

        win_rates[ticker] = np.mean(list_win_rate)

    return win_rates


###############
### Utility ###
###############
def percent_change(
    yf_df: pd.DataFrame, key="Close", periods=1, fillna_zero=True
) -> pd.DataFrame:
    if fillna_zero:
        return (
            yf_df[key]
            .rolling(window=periods)
            .mean()
            .dropna()
            .pct_change(periods=periods)
            .fillna(0)
            .dropna(axis=1, how="all")
            .dropna(axis=0)
        )
    else:
        return (
            yf_df[key]
            .rolling(window=periods)
            .mean()
            .dropna()
            .pct_change(periods=periods)
            .dropna(axis=1, how="all")
            .dropna(axis=0)
        )


def yf_return(yf_df: pd.DataFrame, key="Close") -> pd.DataFrame:
    pct_ch = percent_change(yf_df, key)
    return (pct_ch + 1).cumprod()


def expected_return(yf_df: pd.DataFrame, key="Close") -> np.ndarray:
    pct_ch = percent_change(yf_df, key)
    return expected_return_from_pct_ch(pct_ch)


def expected_return_from_pct_ch(pct_ch: pd.DataFrame) -> np.ndarray:
    return np.array(pct_ch).mean(axis=0)


def simulate_market(yf_df: pd.DataFrame, eval_date, key="Close"):
    """Update portfolio based on market results on eval_date"""
    dday = yf_df.iloc[np.where(yf_df.index == eval_date)[0][0]]
    prev_day = yf_df.iloc[np.where(yf_df.index == eval_date)[0][0] - 1]

    portfolio = read_portfolio()
    t2w = {e[0]: e[1] for e in portfolio["weights"]}
    for ticker in t2w.keys():
        t2w[ticker] *= dday[key][ticker] / prev_day[key][ticker]

    write_portfolio(
        portfolio["date_generated"],
        list(t2w.keys()),
        list(t2w.values()),
        portfolio["cash"],
    )


def predict_fourier(x: List, n_predict: int, n_harmonics=5) -> np.ndarray:
    n = len(x)
    x_freq_dom = np.fft.fft(x)
    frequency = np.fft.fftfreq(n)
    key_indices = list(range(n))
    key_indices.sort(key=lambda i: np.absolute(frequency[i]))

    extended_time = np.arange(0, n + n_predict)
    restored_sig = np.zeros(extended_time.size)
    for i in key_indices[: 1 + n_harmonics * 2]:
        amplitude = np.absolute(x_freq_dom[i]) / n
        phase = np.angle(x_freq_dom[i])
        restored_sig += amplitude * np.cos(
            2 * np.pi * frequency[i] * extended_time + phase
        )

    return restored_sig


def density_function(x: List[float]):
    return scipy.stats.gaussian_kde(x)


def win_rate_given_density_function(
    p: callable, x0: float, upper_limit=2, n_samples=1000
):
    x = np.linspace(x0, upper_limit, n_samples)
    return np.sum([(x[i] - x[i - 1]) * p(x[i - 1]) for i in range(1, len(x))])


def compute_net_profit_by_density_function(
    p: callable, x0: float, upper_limit=2, n_samples=1000
):
    x = np.linspace(x0, upper_limit, n_samples)
    return np.sum(
        [
            (x[i - 1] / x0 - 1) * (x[i] - x[i - 1]) * p(x[i - 1])
            for i in range(1, len(x))
        ]
    )


def compute_net_loss_by_density_function(
    p: callable, x0: float, lower_limit=0, n_samples=1000
):
    x = np.linspace(lower_limit, x0, n_samples)
    return sum(
        [
            (1 - x[i - 1] / x0) * (x[i] - x[i - 1]) * p(x[i - 1])
            for i in range(1, len(x))
        ]
    )


############
### plot ###
############
def plot_kelly_2d(path_savefile="kelly_criterion.png") -> None:
    win_rates = np.linspace(0, 1.0, num=101)
    profits = np.linspace(0, 1.0, num=101)

    losses = np.linspace(0.1, 1.0, num=10)
    all_data = []
    for loss in losses:
        data = [[0] * len(profits) for _ in range(len(win_rates))]
        for i in range(len(win_rates)):
            for j in range(len(profits)):
                ratio = kelly(win_rates[i], profits[j], loss)
                data[i][j] = ratio if ratio > 0 else 0
        all_data.append(data)

    plt.close()
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(2):
        for j in range(5):
            draw = axes[i, j].imshow(
                np.array(all_data[5 * i + j]),
                cmap="viridis",
                interpolation="nearest",
                vmin=0,
                vmax=1,
            )
            axes[i, j].set_title(f"Loss {losses[5 * i + j]:.1}")
            axes[i, j].set_xlabel("Net profit")
            if j == 0:
                axes[i, j].set_ylabel("Win rate")
            axes[i, j].set_xticks(
                range(0, len(profits), len(profits) // 5),
                [profits[i] for i in range(0, len(profits), len(profits) // 5)],
            )
            axes[i, j].set_yticks(
                range(0, len(win_rates), len(win_rates) // 5),
                [win_rates[i] for i in range(0, len(win_rates), len(win_rates) // 5)],
            )

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(draw, cax=cbar_ax)

    plt.savefig(path_savefile)


def plot_return(df: pd.DataFrame, path_savefile: str) -> None:
    df_return = yf_return(df)

    plt.close()
    df_return.plot(figsize=(16, 12))
    plt.savefig(path_savefile)


def plot_SP500_Nasdaq(
    start_date: str,
    end_date: str = None,
    path_savefile: str = "return_SP500_Nasdaq.png",
) -> None:
    if end_date is None:
        start_date, end_date = period(start_date)
    end_date = "2000-01-01"
    tickers = ["SPY", "^IXIC"]
    df = download(tickers, start_date, end_date)

    plot_return(df, path_savefile)


def plot_return_leverage_with_ma(
    start_date: str, path_savefile: str = "figures/return_leverage_with_ma.png"
) -> None:
    tickers = ["SPXL", "TQQQ", "SOXL"]
    start_date, end_date = period(start_date)
    df = download(tickers, start_date, end_date)

    plt.close()
    plt.figure(figsize=(20, 15))
    df_return = yf_return(df)
    # df_ma5 = df["Close"].rolling(window=5).mean().dropna()
    # df_ma5 /= df_ma5.iloc[0]
    # df_ma10 = df["Close"].rolling(window=10).mean().dropna()
    # df_ma10 /= df_ma10.iloc[0]
    df_ma20 = df["Close"].rolling(window=20).mean().dropna()
    df_ma20 /= df_ma20.iloc[0]
    df_ma50 = df["Close"].rolling(window=50).mean().dropna()
    df_ma50 /= df_ma50.iloc[0]
    df_ma100 = df["Close"].rolling(window=100).mean().dropna()
    df_ma100 /= df_ma100.iloc[0]
    df_ma200 = df["Close"].rolling(window=200).mean().dropna()
    df_ma200 /= df_ma200.iloc[0]

    for ticker in tickers:
        plt.plot(df_return[ticker].index, list(df_return[ticker]), label=ticker)
        # plt.plot(df_ma5[ticker].index, list(df_ma5[ticker]), label=ticker + "_ma5")
        # plt.plot(df_ma10[ticker].index, list(df_ma10[ticker]), label=ticker + "_ma10")
        # plt.plot(df_ma20[ticker].index, list(df_ma20[ticker]), label=ticker + "_ma20")
        # plt.plot(df_ma50[ticker].index, list(df_ma50[ticker]), label=ticker + "_ma50")
        plt.plot(
            df_ma100[ticker].index, list(df_ma100[ticker]), label=ticker + "_ma100"
        )
        # plt.plot(df_ma200[ticker].index, list(df_ma200[ticker]), label=ticker + "_ma200")

        peaks = scipy.signal.find_peaks(df_ma100[ticker], width=1, rel_height=0.01)[0]
        bottomes = scipy.signal.find_peaks(-df_ma100[ticker], width=1, rel_height=0.01)[
            0
        ]
        plt.scatter(
            [df_ma100[ticker].index[p] for p in peaks],
            [df_ma100[ticker].iloc[p] for p in peaks],
            color="r",
            marker="v",
            s=30,
        )
        plt.scatter(
            [df_ma100[ticker].index[p] for p in bottomes],
            [df_ma100[ticker].iloc[p] for p in bottomes],
            color="b",
            marker="v",
            s=30,
        )

    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()

    plt.savefig(path_savefile)


def plot_return_leverage(
    start_date, path_savefile: str = "return_leverage.png"
) -> None:
    tickers = ["SPY", "SPXL", "TQQQ", "QQQ", "SOXX", "SOXL"]
    start_date, end_date = period(start_date)
    df = download(tickers, start_date, end_date)

    plot_return(df, path_savefile)


def plot_return_sector(start_date, path_savefile: str = "return_sector.png") -> None:
    df = download_sectors(start_date)
    t2n = {t: " ".join(n.split()[2:]) for t, n in ticker2name("sectors").items()}
    df.columns = pd.MultiIndex.from_tuples([(c1, t2n[c2]) for c1, c2 in df.columns])

    plot_return(df, path_savefile)


def plot_return_etf(start_date, path_savefile: str = "return_etf.png") -> None:
    df = download_selected_etfs(start_date)

    pct_ch = df["Close"].pct_change(periods=1).dropna(axis=1, how="all")
    df_return = (pct_ch + 1).cumprod()
    t2n = {t: n for t, n in ticker2name("selected_etfs").items()}
    df_return.columns = [t2n[c] for c in df_return.columns]

    plt.close()
    plt.figure(figsize=(20, 15))
    list_ticker_return = sorted(
        list(zip(df_return.iloc[-1].index, df_return.iloc[-1].values)),
        key=lambda x: x[1],
    )
    x, y = zip(*list_ticker_return)
    plt.bar(x, y, color="skyblue")
    plt.axhline(y=1, color="red", linestyle="--", linewidth=1)
    plt.ylabel("Return")
    plt.xticks(rotation=90)
    plt.savefig(path_savefile, bbox_inches="tight")


def plot_return_index(start_date, path_savefile: str = "return_index.png") -> None:
    start_date, end_date = period(start_date)
    tickers = [
        "SPY",
        "DIA",
        "QQQ",
        "IWM",
        "GLD",
        "IBIT",
        "IEF",
        "TLT",
        "USO",
        "DBA",
        "DBB",
    ]
    names = [
        "S&P",
        "Dow",
        "Nasdaq",
        "Russell",
        "Gold",
        "Bitcoin",
        "Bond (7-10Y)",
        "Bond (20+Y)",
        "Oil",
        "Agriculutre",
        "Base Metals",
    ]
    t2n = {t: n for t, n in zip(tickers, names)}

    df = download(tickers, start_date, end_date)
    df.columns = pd.MultiIndex.from_tuples([(c1, t2n[c2]) for c1, c2 in df.columns])

    plot_return(df, path_savefile)


def plot_return_portfolio_stocks(
    start_date, path_savefile: str = "return_portfolio_stocks.png"
) -> None:
    pf = read_portfolio()
    t2n = {e[0]: e[2] for e in pf["weights"]}

    df = download_portfolio(start_date)

    df.columns = pd.MultiIndex.from_tuples([(c1, t2n[c2]) for c1, c2 in df.columns])

    plot_return(df, path_savefile)


def plot_matrix(start_date, target: str, matrix_data: str):
    start_date, end_date = period(start_date)

    if target == "assets":
        tickers = get_ticker("assets")
        names = name("assets")
        t2n = ticker2name("assets")
    elif target == "sectors":
        tickers = get_ticker("sectors")
        names = name("sectors")
        t2n = {t: " ".join(n.split()[2:]) for t, n in ticker2name("sectors").items()}
    else:
        raise ValueError(f"Unknown target {target}.")

    df = download(tickers, start_date, end_date)
    df.columns = pd.MultiIndex.from_tuples([(c1, t2n[c2]) for c1, c2 in df.columns])

    if matrix_data == "correlation":
        data = df["Close"].corr()
    else:
        raise ValueError(f"Unknown matrix_data {matrix_data}")

    plt.close()
    # assets have many elements that figsize should be big
    sns.heatmap(data, cmap="seismic", annot=True)
    plt.savefig(f"{matrix_data}_{target}.png", bbox_inches="tight")


def plot_correlation(start_date, target: str = "assets"):
    plot_matrix(start_date, target, "correlation")


def plot_covariance(start_date, target: str = "assets"):
    plot_matrix(start_date, target, "cov")


def plot_return_measure(start_date, path_savefile: str = "return_measure.png"):
    df = download_sectors(start_date)

    x = np.array(percent_change(df))
    log_x_1 = np.log(x + 1)

    plt.close()
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    for i in range(2):
        for j in range(5):
            axes[2 * i, j].hist(
                x[:, 5 * i + j], bins=50, color="skyblue", edgecolor="black"
            )
            axes[2 * i + 1, j].hist(
                log_x_1[:, 5 * i + j], bins=50, color="lightblue", edgecolor="black"
            )
            axes[2 * i, j].set_title(f"x (sector {5 * i + j})")
            axes[2 * i + 1, j].set_title(f"log(x+1) (sector {5 * i + j})")

    plt.savefig(path_savefile)


def plot_soaring_stocks(top_k=7, date_back=365):
    start_date, end_date = period(date_back=date_back)
    # df = download_SandP(start_date)
    # df = download_sectors(start_date)
    df = download_selected_etfs(start_date)
    window2stocks = select_stock_varying_windows(df, top_k)

    for window in window2stocks:
        plt.close()

        pct_ch = df["Close"][window2stocks[window]].pct_change().fillna(0).dropna()
        df_return = (pct_ch + 1).cumprod()

        df_return.plot(figsize=(16, 12))

        plt.savefig(f"{window}_days_best_{top_k}.png")


def optimality(returns: np.ndarray, volatilities: np.ndarray) -> np.ndarray:
    covered = np.array([False] * len(returns))
    for i in range(len(returns)):
        if not covered[i]:
            covered[(returns <= returns[i]) & (volatilities >= volatilities[i])] = True
            covered[i] = False

    return covered


def plot_portfolio_via_sampling(
    start_date: str, key="Close", path_savefile: str = "portfolio_via_sampling.png"
) -> None:
    df = download_assets(start_date)

    (
        tickers,
        weight,
        optimal_r,
        optimal_v,
        random_r,
        random_v,
        random_sharpe,
    ) = optimize_asset_portfolio_via_sampling(df, key, return_samples=True)

    df_random = pd.DataFrame(
        {
            "return": random_r,
            "volatility": random_v,
            "Sharpe": random_sharpe,
        }
    )

    plt.close()
    sns.scatterplot(
        df_random, x="volatility", y="return", hue="Sharpe", s=3, legend=False
    )
    # draw frontiers
    covered = optimality(random_r, random_v)
    plt.scatter(
        random_v[~covered], random_r[~covered], color="orange", marker=".", s=10
    )
    # draw tangent point
    plt.scatter(
        [optimal_v],
        [optimal_r],
        color="r",
        marker="*",
        s=30,
        label="Tangent",
    )
    # plot Nasdaq, Dow, Gold
    plt.scatter(
        [np.sqrt(df[key]["TQQQ"].pct_change().var())],
        [df[key]["TQQQ"].pct_change().mean()],
        color="blue",
        marker="o",
        s=10,
        label="TQQQ",
    )
    plt.scatter(
        [np.sqrt(df[key]["SPXL"].pct_change().var())],
        [df[key]["SPXL"].pct_change().mean()],
        color="green",
        marker="o",
        s=10,
        label="SPXL",
    )
    plt.scatter(
        [np.sqrt(df[key]["SOXL"].pct_change().var())],
        [df[key]["SOXL"].pct_change().mean()],
        color="yellow",
        marker="o",
        s=10,
        label="SOXL",
    )

    plt.legend()
    plt.savefig(path_savefile, bbox_inches="tight")


def append_returns(prev_r: Optional[pd.Series], next_r: pd.Series) -> pd.Series:
    if prev_r is None:
        return next_r
    else:
        return pd.concat([prev_r, next_r])


def set_dates_backtest(current_date: datetime, rebalacing_period: int) -> Tuple:
    # optimize porfolio with data between [opt_s, opt_e] and test on [test_s, test_e]
    opt_s = current_date - timedelta(days=rebalacing_period)
    opt_e = current_date
    test_s = opt_e
    test_e = test_s + timedelta(days=rebalacing_period)
    return opt_s, opt_e, test_s, test_e


def get_benchmark_start_date(dict_benchmark):
    df_benchmark = pd.DataFrame(dict_benchmark)
    df_benchmark.dropna(inplace=True)
    return df_benchmark.index[0].to_pydatetime()


def get_benchmark_data_backtest(start_date, end_date):
    df_benchmark = download(["SPY", "SPXL", "TQQQ", "SOXL"], start_date, end_date)
    equal_rate = (
        (df_benchmark["Close"].pct_change().fillna(0) + 1)
        .cumprod()
        .mean(axis=1)
        .dropna()
    )
    spy = (df_benchmark["Close"]["SPY"].pct_change().fillna(0) + 1).cumprod().dropna()
    spxl = (df_benchmark["Close"]["SPXL"].pct_change().fillna(0) + 1).cumprod().dropna()
    tqqq = (df_benchmark["Close"]["TQQQ"].pct_change().fillna(0) + 1).cumprod().dropna()
    soxl = (df_benchmark["Close"]["SOXL"].pct_change().fillna(0) + 1).cumprod().dropna()

    return {
        "s&p": spy,
        "SPXL": spxl,
        "TQQQ": tqqq,
        "SOXL": soxl,
        "equal_rate": equal_rate,
    }


def plot_rebalancing_backtest(
    start_date,
    end_date,
    rebalacing_periods: List[int] = [10, 20, 50, 100, 200],
    asset_open_date="2010-04-01",
    path_savefile: str = "backtest.png",
) -> None:
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    dict_benchmark = get_benchmark_data_backtest(start_date, end_date)
    start_date = get_benchmark_start_date(dict_benchmark)

    yf_df = download_assets(start_date=asset_open_date, end_date=end_date)

    period2series = {}
    for rebalacing_period in rebalacing_periods:
        generate_initial_portfolio_backtest(start_date, yf_df=yf_df)

        portfolio_returns = pd.Series([1], index=[start_date], name="Close")

        test_date = start_date + timedelta(days=1)
        while test_date <= end_date:
            if sum(test_date == yf_df["Close"].index) > 0:
                if need_to_update_portfolio(test_date, rebalacing_period):
                    generate_portfolio_via_rebalancing(
                        end_date=test_date - timedelta(days=1),
                        yf_df=yf_df[yf_df.index <= test_date - timedelta(days=1)],
                    )

                simulate_market(yf_df, test_date)

                pf_r = portfolio_return(test_date)
                portfolio_returns = append_returns(portfolio_returns, pf_r)

            test_date += timedelta(days=1)

        period2series[rebalacing_period] = portfolio_returns

        os.system("rm portfolio.json")

    plt.close()
    data = {f"portfolio_update_{p}": s for p, s in period2series.items()}
    data.update(dict_benchmark)
    df_return = pd.DataFrame(data)
    df_return.plot(figsize=(16, 8))
    plt.savefig(path_savefile)


def plot_A_divided_by_B(
    A: pd.Series, B: pd.Series, column_name: str, ma_window: int, path_savefile: str
):
    series = pd.concat([A, B], axis=1).dropna()
    df = (series[0] / series[1]).to_frame(name=column_name)
    df = (df.pct_change().fillna(0) + 1).cumprod().dropna()
    df["mean"] = df.rolling(window=ma_window).mean()
    std = np.std(df["mean"])
    df["+1s"] = df["mean"] + std
    df["+2s"] = df["mean"] + 2 * std
    df["-1s"] = df["mean"] - std
    df["-2s"] = df["mean"] - 2 * std
    df.dropna(inplace=True)

    plt.close()
    fig, ax = plt.subplots(figsize=(16, 8))
    colors = {
        column_name: "blue",
        "mean": "yellow",
        "+1s": "orange",
        "+2s": "red",
        "-1s": "green",
        "-2s": "lime",
    }
    styles = {
        column_name: "-",
        "mean": "--",
        "+1s": "--",
        "+2s": "--",
        "-1s": "--",
        "-2s": "--",
    }
    linewidths = {column_name: 2, "mean": 1, "+1s": 1, "+2s": 1, "-1s": 1, "-2s": 1}
    for column in df.columns:
        df[column].plot(
            ax=ax,
            color=colors[column],
            linestyle=styles[column],
            linewidth=linewidths[column],
            label=column,
        )
    plt.legend(loc="best")
    plt.savefig(path_savefile)


def plot_sandp_divided_by_m2(
    start_date: str,
    end_date: str = None,
    path_savefile: str = "figures/SP500_divided_by_M2.png",
):
    if end_date is None:
        start_date, end_date = period(start_date)

    sandp = get_fred_series("s&p500", start_date, end_date)
    m2 = get_fred_series("m2", start_date, end_date)

    plot_A_divided_by_B(sandp, m2, "S&P / M2", 12, path_savefile)


def plot_nasdaq_divided_by_m2(
    start_date: str,
    end_date: str = None,
    path_savefile: str = "figures/Nasdaq_divided_by_M2.png",
):
    if end_date is None:
        start_date, end_date = period(start_date)

    sandp = get_fred_series("nasdaq", start_date, end_date)
    m2 = get_fred_series("m2", start_date, end_date)

    plot_A_divided_by_B(sandp, m2, "Nasdaq / M2", 12, path_savefile)


def plot_nasdaq_divided_by_gdp(
    start_date: str,
    end_date: str = None,
    path_savefile: str = "figures/Nasdaq_divided_by_gdp.png",
):
    if end_date is None:
        start_date, end_date = period(start_date)
    nasdaq = get_fred_series("nasdaq", start_date, end_date)
    gdp = get_fred_series("gdp", start_date, end_date)

    plot_A_divided_by_B(nasdaq, gdp, "Nasdaq / GDP", 4, path_savefile)


def plot_sandp_correlation_by_date(
    start_date: str, key="Close", path_savefile: str = "figures/correlation_by_day.png"
) -> None:
    year, month, date = start_date.split("-")
    assert month == "01" and date == "01"

    start_date, end_date = period(start_date)
    df = download("SPY", start_date, end_date)
    df.dropna(inplace=True)

    year2start_price = {}
    year2end_price = {}
    for year in range(df[key].index.year.min(), df[key].index.year.max()):
        series_year = df[key][df[key].index.year == year].sort_index()
        start_price = series_year.iloc[:20].mean()
        end_price = series_year.iloc[-20:].mean()
        year2start_price[year] = start_price
        year2end_price[year] = end_price

    day2correlation = {}
    for day in pd.date_range(start="2000-02-01", end="2000-12-01", freq="D").strftime(
        "%m-%d"
    ):
        df_day = df[key][df[key].index.strftime("%m-%d") == day]

        list_return_so_far = []
        list_return_left = []
        for year in range(df[key].index.year.min(), df[key].index.year.max()):
            row = df_day[df_day.index.year == year]
            if len(row) == 1:
                list_return_so_far.append(row.iloc[0] / year2start_price[year] - 1)
                list_return_left.append(year2end_price[year] / row.iloc[0] - 1)

        if len(list_return_so_far) > 0:
            day2correlation[day] = np.corrcoef(list_return_so_far, list_return_left)[
                0, 1
            ]

    df = pd.DataFrame(
        {
            "day": sorted(day2correlation.keys()),
            "correlation": [day2correlation[d] for d in sorted(day2correlation.keys())],
        }
    )
    df["mean"] = df["correlation"].rolling(window=20).mean()

    plt.close()
    df.plot(figsize=(16, 8), x="day", y=["correlation", "mean"])
    plt.savefig(path_savefile)


def plot_SP500_MOVE(
    start_date: str,
    end_date: str = None,
    path_savefile: str = "figures/SP500_MOVE.png",
):
    if end_date is None:
        start_date, end_date = period(start_date)

    tickers = ["SPY", "^MOVE"]
    df = download(tickers, start_date, end_date)
    df.dropna(inplace=True)

    corr = np.corrcoef(df["Close"]["SPY"], df["Close"]["^MOVE"])[0, 1]
    print(f"Correlation between SP500 and MOVE: {corr}")

    plt.close()
    fig, ax1 = plt.subplots(figsize=(20, 15))
    ax1.plot(df["Close"]["SPY"].index, list(df["Close"]["SPY"]), color="blue")
    ax2 = ax1.twinx()
    ax2.plot(
        df["Close"]["^MOVE"].index,
        list(-df["Close"]["^MOVE"]),
        color="red",
    )
    plt.xlabel("Date")
    plt.savefig(path_savefile)


def plot_SP500_JOLT(
    start_date: str,
    end_date: str = None,
    path_savefile: str = "figures/SP500_JOLT.png",
):
    if end_date is None:
        start_date, end_date = period(start_date)

    jolt = get_fred_series("jolt", start_date, end_date).to_frame("JOLT")
    jolt.index.name = "Date"
    sp500 = download(["SPY"], start_date, end_date)
    df = jolt.merge(sp500["Close"]["SPY"].to_frame(), on="Date")
    df.dropna(inplace=True)

    # compute drop ratio for several periods
    list_drop = []
    for start_date, end_date in [
        ("2000-01-01", "2004-01-01"),
        ("2007-01-01", "2009-01-01"),
        ("2019-01-01", "2020-05-01"),
    ]:
        spy_top = df["SPY"][
            (df["SPY"].index >= start_date) & (df["SPY"].index <= end_date)
        ].max()
        spy_bottom = df["SPY"][
            (df["SPY"].index >= start_date) & (df["SPY"].index <= end_date)
        ].min()

        jolt_top = df["JOLT"][
            (df["JOLT"].index >= start_date) & (df["JOLT"].index <= end_date)
        ].max()
        jolt_bottom = df["JOLT"][
            (df["JOLT"].index >= start_date) & (df["JOLT"].index <= end_date)
        ].min()

        drop_ratio = (1 - spy_bottom / spy_top) / (1 - jolt_bottom / jolt_top)
        list_drop.append(drop_ratio)
    print(f"spy/jolt drop ratio: {list_drop}")

    # apply spy/jolt drop ratio of 0.535
    spy_top = df["SPY"][(df["SPY"].index >= "2022-01-01")].max()
    jolt_bottom = df["JOLT"][df["JOLT"].index >= "2022-01-01"].min()
    jolt_top = df["JOLT"][df["JOLT"].index >= "2022-01-01"].max()
    est_drop_spy_min = (1 - jolt_bottom / jolt_top) * min(list_drop)
    est_drop_spy_max = (1 - jolt_bottom / jolt_top) * max(list_drop)
    print(f"estimated drop in spy: {est_drop_spy_min}~{est_drop_spy_max}")

    corr = np.corrcoef(df["SPY"], df["JOLT"])[0, 1]
    print(f"[All] Correlation between SP500 and JOLT: {corr}")

    # chatgpt debut
    corr = np.corrcoef(
        df["SPY"][df["SPY"].index <= "2022-11-30"],
        df["JOLT"][df["JOLT"].index <= "2022-11-30"],
    )[0, 1]
    print(f"[Before chatgpt] Correlation between SP500 and JOLT: {corr}")

    # chatgpt debut
    corr = np.corrcoef(
        df["SPY"][df["SPY"].index >= "2022-11-30"],
        df["JOLT"][df["JOLT"].index >= "2022-11-30"],
    )[0, 1]
    print(f"[After chatgpt] Correlation between SP500 and JOLT: {corr}")

    plt.close()
    fig, ax1 = plt.subplots(figsize=(20, 15))
    ax1.plot(df["SPY"].index, list(df["SPY"]), color="blue")
    ax2 = ax1.twinx()
    ax2.plot(
        df["JOLT"].index,
        list(df["JOLT"]),
        color="red",
    )
    plt.savefig(path_savefile)


def plot_predict_fourier(
    regression_start_date: str,
    regression_end_date: str = None,
    test_end_date: str = None,
    key: str = "Close",
    n_predict: int = 100,
):
    if regression_end_date is None:
        assert test_end_date is None
        regression_end_date = datetime.now().strftime("%Y-%m-%d")
        test_end_date = (datetime.now() + timedelta(days=n_predict)).strftime(
            "%Y-%m-%d"
        )

    tickers = ["SPY", "SPXL", "TQQQ", "SOXL"]
    for ticker in tickers:
        df = download([ticker], regression_start_date, test_end_date)
        df.dropna(inplace=True)

        plt.close()
        fig, ax = plt.subplots(figsize=(16, 8))

        df_mean = df[key][ticker].to_frame()
        df_mean[ticker] /= df_mean[ticker].rolling(window=100).mean()
        df_mean.dropna(inplace=True)

        std = np.std(df_mean[ticker])
        mean = np.mean(df_mean[ticker])

        pred = predict_fourier(
            list(df_mean[ticker][df_mean.index <= regression_end_date]),
            n_predict=n_predict,
        )
        new_dates = pd.date_range(
            start=(df_mean.index[-1] + timedelta(days=1)).strftime("%Y-%m-%d"),
            periods=len(pred) - len(df_mean.index),
            freq="D",
        )

        extended_index = df_mean.index.append(new_dates)
        df_mean = df_mean.reindex(extended_index)

        df_mean.loc[:, f"{ticker}_fourier"] = pred[: len(df_mean)]

        df_mean.plot(ax=ax)

        plt.axhline(
            mean + 2 * std,
            color="red",
            linestyle="--",
        )
        plt.axhline(
            mean + std,
            color="orange",
            linestyle="--",
        )
        plt.axhline(
            mean,
            color="black",
            linestyle="--",
        )
        plt.axhline(
            mean - std,
            color="green",
            linestyle="--",
        )
        plt.axhline(
            mean - 2 * std,
            color="lime",
            linestyle="--",
        )
        plt.xlabel("Date")
        plt.legend(loc="best")
        plt.title("Fourier Transformation of Price/MA100")
        plt.savefig(f"figures/predict_fourier_{ticker}.png")


def plot_predict_hmm(
    regression_start_date: str,
    regression_end_date: str = None,
    test_end_date: str = None,
    n_predict: int = 100,
):
    if regression_end_date is None:
        assert test_end_date is None
        regression_end_date = datetime.now().strftime("%Y-%m-%d")
        test_end_date = (datetime.now() + timedelta(days=n_predict)).strftime(
            "%Y-%m-%d"
        )

    tickers = ["SPY", "SPXL", "TQQQ", "SOXL"]
    for ticker in tickers:
        df = download([ticker], regression_start_date, test_end_date)

        df_mean = df["Close"][ticker].to_frame()
        df_mean[ticker] /= df_mean[ticker].rolling(window=100).mean()
        df_mean_volume = df["Volume"][ticker].to_frame("volume_ratio")
        df_mean_volume["volume_ratio"] /= (
            df_mean_volume["volume_ratio"].rolling(window=100).mean()
        )
        df_ratio = df_mean.merge(df_mean_volume, on="Date")
        df_ratio.dropna(inplace=True)

        n_components = 12
        model = hmm.GaussianHMM(
            n_components=n_components, covariance_type="full", n_iter=10_000
        )
        model.fit(df_ratio)
        currstate = model.predict(df_ratio)[-1]

        df_mean.dropna(inplace=True)
        ori_seq = df_mean[ticker]
        new_dates = pd.date_range(
            start=(df_mean.index[-1] + timedelta(days=1)).strftime("%Y-%m-%d"),
            periods=n_predict,
            freq="D",
        )
        extended_index = df_mean.index.append(new_dates)
        df_mean = df_mean.reindex(extended_index)

        for i in range(5):
            df_mean.loc[:, f"{ticker}_hmm_{i}"] = list(ori_seq) + list(
                model.sample(n_predict, currstate=currstate)[0][:, 0]
            )

        plt.close()
        df_mean.plot(figsize=(16, 8))
        plt.xlabel("Date")
        plt.legend(loc="best")
        plt.title("Hidden Markov Model of Price/MA100")
        plt.savefig(f"figures/predict_hmm_{ticker}.png")


def plot_lppls(
    start_date: str,
    end_date: str = None,
):
    if end_date is None:
        start_date, end_date = period(start_date)

    sp500 = download(["SPY"], start_date, end_date)

    time = [pd.Timestamp.toordinal(e) for e in sp500.index]
    price = np.log(sp500["Adj Close"].values)
    observations = np.array([time, price[:, 0]])

    lppls_model = lppls.LPPLS(observations=observations)

    lppls_model.fit(25)
    lppls_model.plot_fit()
    plt.savefig("figures/SP500_lppls_fit.png")

    plt.close()
    res = lppls_model.mp_compute_nested_fits(
        workers=8,
        window_size=360,
        smallest_window_size=30,
        outer_increment=1,
        inner_increment=5,
        max_searches=25,
    )

    lppls_model.plot_confidence_indicators(res)

    plt.savefig("figures/SP500_lppls_risk.png")


def plot_return_volume_leverage_with_ma(start_date: str, window=200) -> None:
    tickers = ["SPY", "SPXL", "TQQQ", "SOXL"]
    start_date, end_date = period(start_date)
    df = download(tickers, start_date, end_date)

    df_return = yf_return(df)

    df_volume = df["Volume"] / df["Volume"].iloc[0]

    for ticker in tickers:
        plt.close()

        fig, ax1 = plt.subplots(figsize=(20, 15))

        ax1.plot(df_volume[ticker].index, list(df_volume[ticker]), "r", label="volume")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("volume ratio", color="r")
        ax1.tick_params(axis="y", labelcolor="r")

        ax2 = ax1.twinx()
        ax2.plot(df_return[ticker].index, list(df_return[ticker]), "b", label="return")
        ax2.set_ylabel("return", color="b")
        ax2.tick_params(axis="y", labelcolor="b")

        fig.legend()

        plt.savefig(f"figures/return_volume_{ticker}.png")


def plot_price_divided_by_ma(
    start_date: str,
    end_date: str = None,
    key: str = "Close",
):
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    tickers = ["SPY", "SPXL", "TQQQ", "SOXL"]
    for ticker in tickers:
        df = download([ticker], start_date, end_date)
        df.dropna(inplace=True)

        frames = []
        for window in range(1, 300):
            df_mean = df[key][ticker].to_frame()
            df_mean[ticker] /= df_mean[ticker].rolling(window=window).mean()
            df_mean.dropna(inplace=True)

            plt.close()
            fig, ax = plt.subplots(figsize=(16, 8))
            df_mean.plot(ax=ax)
            ax.set_ylim(0, 2)

            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            image = Image.open(buf)
            frames.append(image)

        imageio.mimsave(
            f"figures/price_divided_by_ma_{ticker}.gif", frames, duration=0.2
        )
