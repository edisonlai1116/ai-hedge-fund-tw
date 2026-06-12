"""Gooaye (股癌) 輿情共識分析師 agent。

把「股癌」Podcast 與其他社群輿情（由 src/sentiment 的 AudioFeedAdapter / CustomIngestService 抓取、
存進 SQLite，再由 WeightedConsensusEngine 加權）轉成標準的 bullish / bearish / neutral 交易訊號，
寫進 state["data"]["analyst_signals"]，讓 portfolio_manager 把股癌共識列為買賣參考因素之一。

設計重點：
- **惰性匯入 + 優雅降級**：跑到本 agent 時才載入共識引擎；若引擎/資料庫不可用或該檔無輿情資料，
  回中性訊號、confidence=0（不臆造），不會讓整個 CLI 分析崩潰。
- 共識分數（0–100）→ 訊號門檻：>=65 bullish、<45 bearish、其餘 neutral；
  confidence = min(100, |score-50|*2)，反映偏離中性的信念強度（無輿情則為 0）。
"""

import json

from langchain_core.messages import HumanMessage

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress


def _get_consensus(ticker: str):
    """惰性取得單檔共識；任何失敗都回 None（交由呼叫端降級為中性）。"""
    try:
        from src.sentiment.consensus_engine import WeightedConsensusEngine
    except Exception:
        return None
    try:
        engine = WeightedConsensusEngine()
        return engine.get_stock_consensus(ticker)
    except Exception:
        return None


def _score_to_signal(score: int):
    if score >= 65:
        return "bullish"
    if score < 45:
        return "bearish"
    return "neutral"


def gooaye_sentiment_agent(state: AgentState, agent_id: str = "gooaye_sentiment_agent"):
    """讀股癌/輿情加權共識，為每檔股票產生交易訊號。"""
    data = state.get("data", {})
    tickers = data.get("tickers", []) or []
    sentiment_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching Gooaye / 輿情共識")
        consensus = _get_consensus(ticker)

        if not consensus or not consensus.get("opinions"):
            # 無輿情覆蓋（如多數美股）或引擎不可用 → 中性、零信心，不臆造。
            sentiment_analysis[ticker] = {
                "signal": "neutral",
                "confidence": 0,
                "reasoning": {
                    "gooaye_consensus": {
                        "signal": "neutral",
                        "confidence": 0,
                        "consensus_score": (consensus or {}).get("consensus_score", 50),
                        "consensus_label": (consensus or {}).get("consensus_label", "無輿情資料"),
                        "note": "此標的目前無股癌/輿情共識資料（可先用 /sentiment/podcast-scan 或 /custom-ingest 匯入）。",
                        "opinions": [],
                    }
                },
            }
            progress.update_status(agent_id, ticker, "Done (no data)")
            continue

        score = int(consensus.get("consensus_score", 50))
        overall_signal = _score_to_signal(score)
        confidence = min(100, round(abs(score - 50) * 2))

        opinions = consensus.get("opinions", [])
        reasoning = {
            "gooaye_consensus": {
                "signal": overall_signal,
                "confidence": confidence,
                "consensus_score": score,
                "consensus_label": consensus.get("consensus_label", ""),
                "metrics": {
                    "opinion_count": len(opinions),
                    "sources": sorted({o.get("source_name", "") for o in opinions}),
                },
                "opinions": [
                    {
                        "source": o.get("source_name"),
                        "analyst": o.get("analyst_name"),
                        "sentiment": o.get("sentiment_label"),
                        "score": o.get("sentiment_score"),
                        "logic": o.get("core_logic"),
                        "quote": o.get("original_quote"),
                    }
                    for o in opinions
                ],
            }
        }

        sentiment_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=json.dumps(reasoning, ensure_ascii=False, indent=2))

    message = HumanMessage(
        content=json.dumps(sentiment_analysis, ensure_ascii=False),
        name=agent_id,
    )

    if state.get("metadata", {}).get("show_reasoning"):
        show_agent_reasoning(sentiment_analysis, "Gooaye (股癌) Consensus Agent")

    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}
    state["data"]["analyst_signals"][agent_id] = sentiment_analysis

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": state["data"],
    }
