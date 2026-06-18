"""每日報告 pipeline 純函式測試（stdlib，任何環境可跑）。"""
from src.pipeline.daily_report import (
    compute_buy_score, recommendation, rank_rows, parse_holding_tickers, WEIGHTS,
)


def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_compute_buy_score_full():
    out = compute_buy_score(80, 85, 60, 55, relative_strength=90)
    # 80*.35 + 90*.30 + 85*.20 + 60*.05 + 55*.10 = 28+27+17+3+5.5 = 80.5 → 80
    assert out["buy_score"] == 80
    assert out["components"] == {
        "technical": 80, "relative_strength": 90, "gooaye": 85, "ticker_news": 60, "macro": 55,
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


def test_parse_holding_tickers():
    text = "alab 120 46\n2330 980 5\n# comment\nbad line here\n00403A 10 5000\n\n"
    assert parse_holding_tickers(text) == ["alab", "2330", "00403A"]
