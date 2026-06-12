import requests
import json
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')

url = "http://localhost:8000/simple-signals/sp500-daily-top"
payload = {
    "market": "us",
    "scan_type": "lagging_value",
    "limit": 10,
    "use_ai_committee": False
}

print("Querying /simple-signals/sp500-daily-top with scan_type='lagging_value'...")
try:
    response = requests.post(url, json=payload, timeout=30)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        picks = data.get("picks", [])
        print(f"\nSuccessfully retrieved {len(picks)} undervalued stock picks!")
        print("\n--- TOP 5 UNDERVALUED STOCK PICKS IN DAILY SCAN ---")
        for i, p in enumerate(picks[:5]):
            print(f"Rank {i+1}: {p['symbol']} - {p['company_name']}")
            print(f"  AI Chain Layer: {p.get('ai_chain_layer')}")
            print(f"  Critical Bottleneck: {p.get('critical_bottleneck')}")
            print(f"  Novice Rating: {p.get('novice_rating')}")
            print(f"  Daily Score: {p.get('daily_score')} | F-Score: {p.get('fundamental_score')}/9 | Graham Defensive Price: {p.get('graham_number')}")
            print(f"  Reason Summary: {p.get('reason')[:120]}...")
            print("-" * 50)
    else:
        print(f"Request failed: {response.text}")
except Exception as e:
    print(f"HTTP Request failed: {e}")
