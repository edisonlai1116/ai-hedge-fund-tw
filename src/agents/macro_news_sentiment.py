"""總經 / 地緣政治新聞輿情 agent（市場級，免 API Key）。

把會影響大盤走勢的總經與地緣新聞（川普關稅、戰爭衝突、Fed 利率、通膨…）以
免 Key 的新聞 RSS + 關鍵字情緒（src/sentiment/news_feed.py）轉成「市場級」訊號，
對所有標的套用同一個市場輿情偏向，作為買賣判斷因素之一。

設計：惰性匯入 + 優雅降級。抓不到新聞或模組不可用時回中性、confidence 0，不臆造、不崩潰。
這是市場 overlay：個股層級的新聞情緒見 news_feed.ticker_news_sentiment（由每日 pipeline 使用）。
"""

import json

from langchain_core.messages import HumanMessage

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress


def _macro():
    """惰性取得市場級新聞情緒；失敗回 None。"""
    try:
        from src.sentiment.news_feed import macro_sentiment
    except Exception:
        return None
    try:
        return macro_sentiment()
    except Exception:
        return None


def _score_to_signal(score: int):
    if score >= 65:
        return "bullish"
    if score < 45:
        return "bearish"
    return "neutral"


def macro_news_sentiment_agent(state: AgentState, agent_id: str = "macro_news_sentiment_agent"):
    """市場級新聞輿情 → 對每檔標的套用相同的市場偏向訊號。"""
    data = state.get("data", {})
    tickers = data.get("tickers", []) or []

    macro = _macro()
    if not macro or macro.get("n", 0) == 0:
        signal, confidence, score = "neutral", 0, (macro or {}).get("score", 50)
        headlines = []
        label = "無新聞資料"
    else:
        score = int(macro.get("score", 50))
        signal = _score_to_signal(score)
        confidence = min(100, round(abs(score - 50) * 2))
        headlines = macro.get("sample_headlines", [])
        label = macro.get("label", "")

    reasoning = {
        "macro_news": {
            "signal": signal,
            "confidence": confidence,
            "market_sentiment_score": score,
            "label": label,
            "headline_count": (macro or {}).get("n", 0),
            "sample_headlines": headlines,
            "note": "市場級總經/地緣新聞情緒（免 Key 關鍵字啟發式），對所有標的套用同一偏向。",
        }
    }

    # 市場 overlay：對每檔標的給相同訊號。
    sentiment_analysis = {
        ticker: {"signal": signal, "confidence": confidence, "reasoning": reasoning}
        for ticker in tickers
    }

    message = HumanMessage(
        content=json.dumps(sentiment_analysis, ensure_ascii=False),
        name=agent_id,
    )

    if state.get("metadata", {}).get("show_reasoning"):
        show_agent_reasoning(sentiment_analysis, "Macro News Sentiment Agent (總經/地緣輿情)")

    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}
    state["data"]["analyst_signals"][agent_id] = sentiment_analysis

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": state["data"],
    }
