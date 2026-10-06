"""自動提醒：持股解析、盤中時段、去重的離線單元測試。"""
from datetime import datetime, timezone

from src.pipeline import alerts


def test_parse_holdings_merges_and_normalizes():
    text = "# 註解\nceg 306 40\nMSFT 405 22\n2330 600 1000\nceg 300 10\nbad line\n"
    rows = {h["symbol"]: h for h in alerts.parse_holdings(text)}
    assert set(rows) == {"CEG", "MSFT", "2330.TW"}
    assert rows["CEG"]["shares"] == 50
    assert abs(rows["CEG"]["cost"] - 304.8) < 1e-6


def test_session_fraction_us_open_and_closed():
    # 2026-10-06 是週二；美東 12:45（EDT = UTC-4）→ 開盤 195/390 分鐘
    assert abs(alerts.session_fraction("us", datetime(2026, 10, 6, 16, 45, tzinfo=timezone.utc)) - 0.5) < 1e-6
    assert alerts.session_fraction("us", datetime(2026, 10, 6, 21, 0, tzinfo=timezone.utc)) is None
    # 週六不開盤
    assert alerts.session_fraction("tw", datetime(2026, 10, 10, 3, 0, tzinfo=timezone.utc)) is None


def test_filter_new_dedups_recent_keys():
    a = [{"key": "buy:CEG"}, {"key": "buy:VST"}]
    state = {alerts._hash("buy:CEG"): datetime.now(alerts.TPE).isoformat()}
    assert [x["key"] for x in alerts.filter_new(a, state)] == ["buy:VST"]
