"""尼可拉斯楊 (Nicholas Yang) Live 檢核分析師 agent。

把「尼可拉斯楊Live」Podcast 的個股觀點（AI 算力蛋糕 + 卡脖子瓶頸框架）轉成標準的
bullish / bearish / neutral 交易訊號，作為**獨立的檢核 agent**：在其他 agent 之外，
額外用尼可拉斯楊的觀點對個股做一次多空交叉檢核，寫進 state["data"]["analyst_signals"]。

資料來源：`docs/data/nicolas_opinions.json`（由 RSS 自動追蹤最新集數 + 種子/人工觀點維護；
真實逐集轉錄需本機 Whisper，雲端僅偵測新集數）。

設計同 gooaye_sentiment：惰性讀取 + 優雅降級（無資料回中性、confidence=0，不臆造）。
共識分數 0–100 → >=65 bullish、<45 bearish、其餘 neutral；confidence=min(100,|score-50|*2)。
"""

import json
import os

from langchain_core.messages import HumanMessage

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress

_NICOLAS_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "docs", "data", "nicolas_opinions.json",
)


def _nicolas_consensus(ticker: str):
    """從 nicolas_opinions.json 取得單檔尼可拉斯楊觀點共識；失敗回 None。"""
    try:
        from src.sentiment.consensus_engine import score_opinions
        with open(_NICOLAS_JSON, encoding="utf-8") as f:
            store = json.load(f)
        ops = store.get("opinions", []) if isinstance(store, dict) else (store or [])
        simple = str(ticker).split(".")[0].upper()
        matched = [o for o in ops if str(o.get("target_ticker", "")).split(".")[0].upper() == simple]
        if not matched:
            return {"consensus_score": 50, "consensus_label": "無尼可拉斯楊觀點", "opinions": []}
        return score_opinions(matched)
    except Exception:
        return None


def _score_to_signal(score: int):
    if score >= 65:
        return "bullish"
    if score < 45:
        return "bearish"
    return "neutral"


def nicolas_sentiment_agent(state: AgentState, agent_id: str = "nicolas_sentiment_agent"):
    """讀尼可拉斯楊Live 觀點，為每檔股票產生檢核用的多空訊號。"""
    data = state.get("data", {})
    tickers = data.get("tickers", []) or []
    sentiment_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching 尼可拉斯楊 觀點")
        consensus = _nicolas_consensus(ticker)

        if not consensus or not consensus.get("opinions"):
            sentiment_analysis[ticker] = {
                "signal": "neutral",
                "confidence": 0,
                "reasoning": {
                    "nicolas_consensus": {
                        "signal": "neutral",
                        "confidence": 0,
                        "consensus_score": (consensus or {}).get("consensus_score", 50),
                        "consensus_label": (consensus or {}).get("consensus_label", "無觀點資料"),
                        "note": "此標的目前無尼可拉斯楊Live 觀點覆蓋（多為其 AI 算力蛋糕鏈上個股才有）。",
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
            "nicolas_consensus": {
                "signal": overall_signal,
                "confidence": confidence,
                "consensus_score": score,
                "consensus_label": consensus.get("consensus_label", ""),
                "metrics": {"opinion_count": len(opinions)},
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

    message = HumanMessage(content=json.dumps(sentiment_analysis, ensure_ascii=False), name=agent_id)

    if state.get("metadata", {}).get("show_reasoning"):
        show_agent_reasoning(sentiment_analysis, "尼可拉斯楊Live 檢核 Agent")

    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}
    state["data"]["analyst_signals"][agent_id] = sentiment_analysis

    progress.update_status(agent_id, None, "Done")
    return {"messages": [message], "data": state["data"]}
