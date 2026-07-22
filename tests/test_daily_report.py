"""每日報告 pipeline 純函式測試（stdlib，任何環境可跑）。"""
from src.pipeline.daily_report import (
    compute_buy_score, recommendation, rank_rows, parse_holding_tickers, WEIGHTS,
    diversify_head,
)


def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_compute_buy_score_full():
    out = compute_buy_score(80, 85, 60, 55, relative_strength=90)
    # 80*.30 + 90*.25 + 50*.20(chip缺→中性) + 85*.15 + 60*.05 + 55*.05 = 24+22.5+10+12.75+3+2.75 = 75
    assert out["buy_score"] == 75
    assert out["components"] == {
        "technical": 80, "relative_strength": 90, "chip_flow": 50,
        "gooaye": 85, "ticker_news": 60, "macro": 55,
    }


def test_compute_buy_score_missing_defaults_to_neutral():
    out = compute_buy_score(None, None, None, None)
    assert out["buy_score"] == 50
    # 缺相對強度時以中性 50 代入，不影響全中性結果
    assert out["components"]["relative_strength"] == 50


def test_recommendation_bands():
    assert recommendation(75) == "強力買進"
    assert recommendation(60) == "偏多分批"
    assert recommendation(50) == "中性觀望"
    assert recommendation(40) == "偏空減碼"
    assert recommendation(20) == "避險／賣出"


def test_rank_rows_desc():
    rows = [{"ticker": "A", "buy_score": 40}, {"ticker": "B", "buy_score": 70}, {"ticker": "C", "buy_score": 55}]
    ranked = rank_rows(rows)
    assert [r["ticker"] for r in ranked] == ["B", "C", "A"]


def test_diversify_head_caps_theme():
    # MU/WDC/STX/2330(記憶體+製造混合驗證)：同主題（記憶體）最多 3 檔進前段，第 4 檔遞延。
    rows = [
        {"ticker": "MU", "buy_score": 90},
        {"ticker": "WDC", "buy_score": 88},
        {"ticker": "STX", "buy_score": 86},
        {"ticker": "SNDK", "buy_score": 84},   # 2026-07-22 起 SNDK 也對映記憶體主題
        {"ticker": "TSM", "buy_score": 82},
    ]
    out = diversify_head(rows, head=4, max_per_theme=2)
    tickers = [r["ticker"] for r in out]
    # 記憶體主題 MU/WDC 佔滿 2 檔上限後，STX/SNDK 都被遞延到 head 之後；名單成員不變。
    assert tickers[:4] == ["MU", "WDC", "TSM", "STX"]
    assert set(tickers) == {"MU", "WDC", "STX", "SNDK", "TSM"}


def test_diversify_head_no_theme_passthrough():
    rows = [{"ticker": f"X{i}", "buy_score": 90 - i} for i in range(6)]
    assert [r["ticker"] for r in diversify_head(rows, head=3)] == [f"X{i}" for i in range(6)]


def test_parse_holding_tickers():
    text = "alab 120 46\n2330 980 5\n# comment\nbad line here\n00403A 10 5000\n\n"
    assert parse_holding_tickers(text) == ["alab", "2330", "00403A"]
