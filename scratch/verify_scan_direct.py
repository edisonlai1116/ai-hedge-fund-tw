import sys
import os
import json

# Ensure sys.path includes the current project directory so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sp500_daily import get_sp500_daily_top_picks

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')

print("Starting direct lagging_value scan verification...")
try:
    # Run daily scan for US market with lagging_value type
    result = get_sp500_daily_top_picks(market="us", scan_type="lagging_value", limit=10, use_ai_committee=False)
    picks = result.get("picks", [])
    print(f"\nSuccessfully completed daily scan! Found {len(picks)} candidates.")
    print("\n--- STOCK PICKS DETAIL ---")
    for i, p in enumerate(picks[:8]):
        print(f"Rank {i+1}: {p['symbol']} - {p['company_name']}")
        print(f"  Technical Bias: {p.get('bias')} | F-Score: {p.get('fundamental_score')}/9")
        print(f"  InvestingPro Fair Value: {p.get('investingpro_fair_value')} | Gap: {p.get('valuation_gap_pct')}%")
        print(f"  Composite Score: {p.get('composite_score')} | Daily Score: {p.get('daily_score')}")
        print(f"  Action Label (sp500_daily): {p.get('action_label')} | Today Action (simple_signal): {p.get('today_action')}")
        print(f"  Today Note: {p.get('today_note')}")
        print(f"  Expected Return: {p.get('expected_return_pct') or 0.0:.2f}% | Expected Buy Zone: {p.get('today_entry_zone')}")
        print(f"  Reason: {p.get('reason')[:150]}...")
        print("-" * 80)
except Exception as e:
    import traceback
    print(f"Error occurred during scan: {e}")
    traceback.print_exc()
