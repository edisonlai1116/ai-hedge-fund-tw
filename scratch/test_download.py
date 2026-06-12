import sys
import os
import yfinance as yf

# Ensure sys.path includes the current project directory so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sp500_daily import fetch_sp500_constituents, download_sp500_price_map, build_report

constituents = fetch_sp500_constituents()
print(f"Fetched {len(constituents)} constituents.")

symbols = [item.yf_symbol for item in constituents[:5]]
print(f"Downloading price map for {symbols}...")
try:
    price_map = download_sp500_price_map(symbols, "3y")
    print(f"Downloaded price map with keys: {list(price_map.keys())}")
    for k, v in price_map.items():
        print(f"  {k}: shape={v.shape}")
        
        try:
            report = build_report(k, v, fetch_fundamentals=False)
            print(f"  {k} build_report: success! composite_score={report.composite_score}")
        except Exception as e:
            print(f"  {k} build_report failed: {e}")
except Exception as e:
    print(f"Download failed: {e}")
