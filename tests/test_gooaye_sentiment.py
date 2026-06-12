"""股癌 (Gooaye) 輿情共識 agent 與共識引擎測試。

- 共識引擎測試為純 stdlib（sqlite3），任何環境都可跑。
- agent 測試需要 langchain_core；未安裝時自動 skip（importorskip）。
"""
import pytest


def test_consensus_engine_missing_db_returns_neutral(tmp_path):
    """DB 不存在 → 預設中性觀察、score 50、無 opinions（不崩潰、不臆造）。"""
    from src.sentiment.consensus_engine import WeightedConsensusEngine

    engine = WeightedConsensusEngine(db_path=str(tmp_path / "nope.db"))
    result = engine.get_stock_consensus("2330.TW")
    assert result["consensus_score"] == 50
    assert result["consensus_label"] == "中性觀察"
    assert result["opinions"] == []


def test_score_to_signal_thresholds():
    """共識分數 → 訊號門檻：>=65 bullish、<45 bearish、其餘 neutral。"""
    pytest.importorskip("langchain_core")
    from src.agents.gooaye_sentiment import _score_to_signal

    assert _score_to_signal(90) == "bullish"
    assert _score_to_signal(65) == "bullish"
    assert _score_to_signal(64) == "neutral"
    assert _score_to_signal(45) == "neutral"
    assert _score_to_signal(44) == "bearish"
    assert _score_to_signal(10) == "bearish"


def _fake_state(tickers):
    return {
        "data": {"tickers": tickers, "analyst_signals": {}},
        "metadata": {"show_reasoning": False},
    }


def test_agent_bullish_with_consensus(monkeypatch):
    """有強多頭共識 (score 85) → bullish、confidence 70、寫進 analyst_signals。"""
    pytest.importorskip("langchain_core")
    from src.agents import gooaye_sentiment as mod

    monkeypatch.setattr(mod, "_get_consensus", lambda ticker: {
        "consensus_score": 85,
        "consensus_label": "強力多頭共識",
        "opinions": [{
            "source_name": "Gooaye", "analyst_name": "謝孟恭 (股癌)",
            "sentiment_label": "Bullish", "sentiment_score": 85,
            "core_logic": "先進製程供不應求", "original_quote": "定海神針",
        }],
    })

    state = _fake_state(["2330.TW"])
    out = mod.gooaye_sentiment_agent(state)

    sig = out["data"]["analyst_signals"]["gooaye_sentiment_agent"]["2330.TW"]
    assert sig["signal"] == "bullish"
    assert sig["confidence"] == 70
    assert sig["reasoning"]["gooaye_consensus"]["consensus_score"] == 85
    assert len(sig["reasoning"]["gooaye_consensus"]["opinions"]) == 1


def test_agent_neutral_without_data(monkeypatch):
    """無輿情資料 (None) → neutral、confidence 0，不臆造。"""
    pytest.importorskip("langchain_core")
    from src.agents import gooaye_sentiment as mod

    monkeypatch.setattr(mod, "_get_consensus", lambda ticker: None)

    state = _fake_state(["AAPL"])
    out = mod.gooaye_sentiment_agent(state)

    sig = out["data"]["analyst_signals"]["gooaye_sentiment_agent"]["AAPL"]
    assert sig["signal"] == "neutral"
    assert sig["confidence"] == 0


def test_registered_in_analyst_config():
    """gooaye_sentiment 已註冊進 ANALYST_CONFIG（portfolio_manager 才會納入）。"""
    pytest.importorskip("langchain_core")
    from src.utils.analysts import ANALYST_CONFIG

    assert "gooaye_sentiment" in ANALYST_CONFIG
    assert ANALYST_CONFIG["gooaye_sentiment"]["agent_func"].__name__ == "gooaye_sentiment_agent"
