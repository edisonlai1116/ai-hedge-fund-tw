import datetime
import logging
import os
import pandas as pd
import requests
import time

logger = logging.getLogger(__name__)

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    CompanyFactsResponse,
)

# Global cache instance
_cache = get_cache()

# Point-in-time：財報 report_period 是「期末日」，不是公布日。10-Q 通常期末後 ~40 天、10-K ~60-90 天才公布。
# 回測（end_date 在過去）時，只允許使用「期末日 + 公布延遲」已經過去的財報，避免偷看尚未公布的數字。
# 即時分析（end_date = 今天）不套延遲：資料商只會有已公布的財報。
PUBLICATION_LAG_DAYS = {"quarterly": 45, "ttm": 45, "annual": 90}


# 區間快取：回測逐日查詢的價格/內部人交易，若落在已抓過的日期區間內，直接在本機切片，
# 避免同一份資料每天重打一次付費 API。只在「完全涵蓋」時使用，不會回傳範圍外資料。
_RANGE_CACHE: dict[tuple, tuple[str, str, list[dict]]] = {}


def _range_get(kind: str, ticker: str, start: str, end: str, date_field: str) -> list[dict] | None:
    hit = _RANGE_CACHE.get((kind, ticker))
    if not hit:
        return None
    s0, e0, rows = hit
    if start < s0 or end > e0:
        return None
    return [r for r in rows if start <= str(r.get(date_field, ""))[:10] <= end]


def _range_put(kind: str, ticker: str, start: str, end: str, rows: list[dict], date_field: str) -> None:
    hit = _RANGE_CACHE.get((kind, ticker))
    if hit and not (end < hit[0] or start > hit[1]):     # 重疊 → 合併
        s0, e0, old = hit
        merged = {str(r.get(date_field)): r for r in old}
        merged.update({str(r.get(date_field)): r for r in rows})
        _RANGE_CACHE[(kind, ticker)] = (min(start, s0), max(end, e0), sorted(merged.values(), key=lambda r: str(r.get(date_field))))
    else:
        _RANGE_CACHE[(kind, ticker)] = (start, end, rows)


def pit_report_cutoff(end_date: str, period: str = "ttm", today: datetime.date | None = None) -> str:
    """回傳在 end_date 當下「已經公布」的財報所允許的最晚 report_period。"""
    today = today or datetime.date.today()
    end = datetime.date.fromisoformat(end_date[:10])
    if end >= today:
        return end_date[:10]
    lag = PUBLICATION_LAG_DAYS.get(period, 90)
    return (end - datetime.timedelta(days=lag)).isoformat()


def _filter_point_in_time(items: list, cutoff: str) -> list:
    """防禦性過濾：剔除 report_period 晚於 cutoff 的財報（資料商若沒照參數過濾也擋得住）。"""
    out = []
    for it in items:
        rp = str(getattr(it, "report_period", "") or "")[:10]
        if not rp or rp <= cutoff:
            out.append(it)
    return out


def _make_api_request(url: str, headers: dict, method: str = "GET", json_data: dict = None, max_retries: int = 3) -> requests.Response:
    """
    Make an API request with rate limiting handling and moderate backoff.
    
    Args:
        url: The URL to request
        headers: Headers to include in the request
        method: HTTP method (GET or POST)
        json_data: JSON data for POST requests
        max_retries: Maximum number of retries (default: 3)
    
    Returns:
        requests.Response: The response object
    
    Raises:
        Exception: If the request fails with a non-429 error
    """
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        if method.upper() == "POST":
            response = requests.post(url, headers=headers, json=json_data)
        else:
            response = requests.get(url, headers=headers)
        
        if response.status_code == 429 and attempt < max_retries:
            # Linear backoff: 60s, 90s, 120s, 150s...
            delay = 60 + (30 * attempt)
            print(f"Rate limited (429). Attempt {attempt + 1}/{max_retries + 1}. Waiting {delay}s before retrying...")
            time.sleep(delay)
            continue
        
        # Return the response (whether success, other errors, or final 429)
        return response


_FALLBACK_WARNED = set()


def _yfinance_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """financialdatasets 不可用（金鑰無效/額度用完/無資料）時的免費備援。歷史日線為 point-in-time 安全資料。"""
    try:
        import yfinance as yf
        end_excl = (datetime.date.fromisoformat(end_date[:10]) + datetime.timedelta(days=1)).isoformat()
        df = yf.download(ticker, start=start_date[:10], end=end_excl, interval="1d", auto_adjust=False, progress=False)
        if df is None or df.empty:
            return []
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        out = []
        for d, r in df.dropna(subset=["Close"]).iterrows():
            out.append(Price(open=float(r["Open"]), close=float(r["Close"]), high=float(r["High"]), low=float(r["Low"]),
                             volume=int(r["Volume"]) if pd.notna(r["Volume"]) else 0, time=pd.Timestamp(d).strftime("%Y-%m-%d")))
        return out
    except Exception as e:
        logger.warning("yfinance fallback failed for %s: %s", ticker, e)
        return []


def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> list[Price]:
    """Fetch price data from cache or API."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{start_date}_{end_date}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_prices(cache_key):
        return [Price(**price) for price in cached_data]
    if (ranged := _range_get("prices", ticker, start_date[:10], end_date[:10], "time")) is not None:
        return [Price(**price) for price in ranged]

    # If not in cache, fetch from API
    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    url = f"https://api.financialdatasets.ai/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}"
    response = _make_api_request(url, headers)
    if response.status_code != 200:
        if response.status_code not in _FALLBACK_WARNED:
            _FALLBACK_WARNED.add(response.status_code)
            logger.warning("financialdatasets prices HTTP %s → 改用 yfinance 備援", response.status_code)
        prices = _yfinance_prices(ticker, start_date, end_date)
        if prices:
            _cache.set_prices(cache_key, [p.model_dump() for p in prices])
            _range_put("prices", ticker, start_date[:10], end_date[:10], [p.model_dump() for p in prices], "time")
        return prices

    # Parse response with Pydantic model
    try:
        price_response = PriceResponse(**response.json())
        prices = price_response.prices
    except Exception as e:
        logger.warning("Failed to parse price response for %s: %s", ticker, e)
        return []

    if not prices:
        return []

    # Cache the results using the comprehensive cache key
    _cache.set_prices(cache_key, [p.model_dump() for p in prices])
    _range_put("prices", ticker, start_date[:10], end_date[:10], [p.model_dump() for p in prices], "time")
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from cache or API."""
    # Create a cache key that includes all parameters to ensure exact matches
    cutoff = pit_report_cutoff(end_date, period)
    cache_key = f"{ticker}_{period}_{cutoff}_{limit}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**metric) for metric in cached_data]

    # If not in cache, fetch from API
    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    url = f"https://api.financialdatasets.ai/financial-metrics/?ticker={ticker}&report_period_lte={cutoff}&limit={limit}&period={period}"
    response = _make_api_request(url, headers)
    if response.status_code != 200:
        return []

    # Parse response with Pydantic model
    try:
        metrics_response = FinancialMetricsResponse(**response.json())
        financial_metrics = _filter_point_in_time(metrics_response.financial_metrics, cutoff)
    except Exception as e:
        logger.warning("Failed to parse financial metrics response for %s: %s", ticker, e)
        return []

    if not financial_metrics:
        return []

    # Cache the results as dicts using the comprehensive cache key
    _cache.set_financial_metrics(cache_key, [m.model_dump() for m in financial_metrics])
    return financial_metrics


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    """Fetch line items from API."""
    # If not in cache or insufficient data, fetch from API
    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    url = "https://api.financialdatasets.ai/financials/search/line-items"

    cutoff = pit_report_cutoff(end_date, period)
    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "end_date": cutoff,
        "period": period,
        "limit": limit,
    }
    response = _make_api_request(url, headers, method="POST", json_data=body)
    if response.status_code != 200:
        return []
    
    try:
        data = response.json()
        response_model = LineItemResponse(**data)
        search_results = _filter_point_in_time(response_model.search_results, cutoff)
    except Exception as e:
        logger.warning("Failed to parse line items response for %s: %s", ticker, e)
        return []
    if not search_results:
        return []

    # Cache the results
    return search_results[:limit]


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or API."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_insider_trades(cache_key):
        return [InsiderTrade(**trade) for trade in cached_data]
    if start_date and (ranged := _range_get("insider", ticker, start_date[:10], end_date[:10], "filing_date")) is not None:
        return [InsiderTrade(**trade) for trade in ranged[:limit]]

    # If not in cache, fetch from API
    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    all_trades = []
    current_end_date = end_date

    while True:
        url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
        if start_date:
            url += f"&filing_date_gte={start_date}"
        url += f"&limit={limit}"

        response = _make_api_request(url, headers)
        if response.status_code != 200:
            break

        try:
            data = response.json()
            response_model = InsiderTradeResponse(**data)
            insider_trades = response_model.insider_trades
        except Exception as e:
            logger.warning("Failed to parse insider trades response for %s: %s", ticker, e)
            break

        if not insider_trades:
            break

        all_trades.extend(insider_trades)

        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(insider_trades) < limit:
            break

        # Update end_date to the oldest filing date from current batch for next iteration
        current_end_date = min(trade.filing_date for trade in insider_trades).split("T")[0]

        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_trades:
        return []

    # Cache the results using the comprehensive cache key
    _cache.set_insider_trades(cache_key, [trade.model_dump() for trade in all_trades])
    if start_date:
        _range_put("insider", ticker, start_date[:10], end_date[:10], [t.model_dump() for t in all_trades], "filing_date")
    return all_trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[CompanyNews]:
    """Fetch company news from cache or API."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_company_news(cache_key):
        return [CompanyNews(**news) for news in cached_data]

    # If not in cache, fetch from API
    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    all_news = []
    current_end_date = end_date

    while True:
        url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}"
        if start_date:
            url += f"&start_date={start_date}"
        url += f"&limit={limit}"

        response = _make_api_request(url, headers)
        if response.status_code != 200:
            break

        try:
            data = response.json()
            response_model = CompanyNewsResponse(**data)
            company_news = response_model.news
        except Exception as e:
            logger.warning("Failed to parse company news response for %s: %s", ticker, e)
            break

        if not company_news:
            break

        all_news.extend(company_news)

        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(company_news) < limit:
            break

        # Update end_date to the oldest date from current batch for next iteration
        current_end_date = min(news.date for news in company_news).split("T")[0]

        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_news:
        return []

    # Cache the results using the comprehensive cache key
    _cache.set_company_news(cache_key, [news.model_dump() for news in all_news])
    return all_news


def get_market_cap(
    ticker: str,
    end_date: str,
    api_key: str = None,
) -> float | None:
    """Fetch market cap from the API."""
    # Check if end_date is today
    if end_date == datetime.datetime.now().strftime("%Y-%m-%d"):
        # Get the market cap from company facts API
        headers = {}
        financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
        if financial_api_key:
            headers["X-API-KEY"] = financial_api_key

        url = f"https://api.financialdatasets.ai/company/facts/?ticker={ticker}"
        response = _make_api_request(url, headers)
        if response.status_code != 200:
            print(f"Error fetching company facts: {ticker} - {response.status_code}")
            return None

        data = response.json()
        response_model = CompanyFactsResponse(**data)
        return response_model.company_facts.market_cap

    financial_metrics = get_financial_metrics(ticker, end_date, api_key=api_key)
    if not financial_metrics:
        return None

    market_cap = financial_metrics[0].market_cap

    if not market_cap:
        return None

    return market_cap


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date, api_key=api_key)
    return prices_to_df(prices)
