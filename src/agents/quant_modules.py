"""六個量化分析模組（取代人設型 LLM 分析師作為預設）：

  1. valuation_quality      估值與品質
  2. growth_earnings        成長與財報（含財報加速）
  3. technical_momentum     技術與動能（核心：6 個月風險調整動能排名；相對強度/過熱為診斷）
  4. sentiment_positioning  情緒與籌碼（放空比例、分析師共識、內部人交易）
  5. news_catalyst          新聞與催化（近 30 天事件、大跌診斷）
  6. ai_industry_macro      AI 產業與總經（AI 曝險、產業動能、市場狀態）

規則：數字分數一律由 src.ranking 確定性計算；LLM 不參與打分。資料不足 → insufficient_data（不當中性）。
回測（end_date 在過去）時自動切換為 point-in-time 模式：只用價格模組，基本面/新聞標記為不可用。
"""
from __future__ import annotations

import json
import threading
from datetime import date
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress

_LOCK = threading.Lock()
_CACHE: Dict[Tuple, Dict] = {}

MODULES = {
    "valuation_quality": {"name": "估值與品質", "scores": ("valuation", "quality")},
    "growth_earnings": {"name": "成長與財報", "scores": ("earnings_acceleration", "growth")},
    "technical_momentum": {"name": "技術與動能", "scores": ("momentum_rank",)},
    "sentiment_positioning": {"name": "情緒與籌碼", "scores": ("positioning",)},
    "news_catalyst": {"name": "新聞與催化", "scores": ("catalyst",)},
    "ai_industry_macro": {"name": "AI 產業與總經", "scores": ("industry_momentum", "ai_exposure")},
}


def _holdings_from_portfolio(portfolio: Dict) -> Dict[str, Dict]:
    out = {}
    for t, p in (portfolio or {}).get("positions", {}).items():
        if (p.get("long") or 0) > 0:
            out[t] = {"shares": p["long"], "cost": p.get("long_cost_basis")}
    return out


def get_ranking(tickers: List[str], end_date: str, portfolio: Optional[Dict] = None) -> Dict:
    """同一天、同一批股票、同一持股只算一次（六個模組與 PM 共用）。"""
    holdings = _holdings_from_portfolio(portfolio or {})
    key = (tuple(sorted(tickers)), end_date, tuple(sorted(holdings)))
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        from src.ranking.engine import rank_stocks
        asof = None if end_date >= date.today().isoformat() else end_date
        out = rank_stocks(tickers, holdings=holdings, asof=asof)
        _CACHE[key] = out
        return out


def _row(ranking: Dict, ticker: str) -> Optional[Dict]:
    t = ticker.upper()
    for r in ranking.get("ranking", []):
        if r["ticker"].split(".")[0] == t.split(".")[0]:
            return r
    return None


def _signal(score: Optional[float]) -> Tuple[str, int]:
    if score is None:
        return "insufficient_data", 0
    if score >= 60:
        return "bullish", int(min(100, 50 + (score - 50)))
    if score <= 40:
        return "bearish", int(min(100, 50 + (50 - score)))
    return "neutral", 50


def _insider_score(ticker: str, end_date: str, api_key: Optional[str]) -> Optional[Dict]:
    """近 90 天內部人淨買賣（financialdatasets，filing_date ≤ end_date → point-in-time）。"""
    try:
        from datetime import datetime, timedelta
        from src.tools.api import get_insider_trades
        start = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")
        trades = get_insider_trades(ticker, end_date, start_date=start, limit=500, api_key=api_key)
        if not trades:
            return None
        buy = sum(abs(t.transaction_value or 0) for t in trades if (t.transaction_shares or 0) > 0)
        sell = sum(abs(t.transaction_value or 0) for t in trades if (t.transaction_shares or 0) < 0)
        if buy + sell == 0:
            return None
        net = (buy - sell) / (buy + sell)
        return {"score": round(50 + 50 * net, 1), "buy_value": buy, "sell_value": sell, "n": len(trades)}
    except Exception:
        return None


def _make_agent(module_key: str):
    spec = MODULES[module_key]
    agent_id = f"{module_key}_module"

    def agent(state: AgentState, agent_id: str = agent_id):
        data = state["data"]
        tickers, end_date = data["tickers"], data["end_date"]
        progress.update_status(agent_id, None, "Cross-sectional ranking")
        ranking = get_ranking(tickers, end_date, data.get("portfolio"))
        out = {}
        for t in tickers:
            r = _row(ranking, t)
            if r is None:
                out[t] = {"signal": "insufficient_data", "confidence": 0, "reasoning": {"error": "no data"}}
                continue
            s = r["scores"]
            vals = [s.get(k) for k in spec["scores"] if s.get(k) is not None]
            metrics = {k: s.get(k) for k in spec["scores"]}
            extra = {}
            if module_key == "sentiment_positioning":
                from src.utils.api_key import get_api_key_from_state
                ins = _insider_score(t, end_date, get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY"))
                if ins:
                    vals.append(ins["score"])
                    extra["insider_90d"] = ins
            if module_key == "technical_momentum":
                extra = {"relative_strength": s.get("relative_strength"), "technical": s.get("technical"),
                         "overextension": s.get("overextension"), "returns": r.get("returns")}
            if module_key == "news_catalyst":
                extra = {"key_catalyst": (r.get("catalyst") or {}).get("key_catalyst"), "shock": r.get("shock")}
            if module_key == "ai_industry_macro":
                extra = {"ai": r.get("ai"), "regime": ranking["regime"]["regime"]}
            score = round(sum(vals) / len(vals), 1) if vals else None
            sig, conf = _signal(score)
            out[t] = {"signal": sig, "confidence": conf, "score": score,
                      "reasoning": {"module": spec["name"], "metrics": metrics, **extra,
                                    "mode": ranking.get("mode")}}
        state["data"]["analyst_signals"][agent_id] = out
        if state["metadata"].get("show_reasoning"):
            show_agent_reasoning(out, spec["name"])
        progress.update_status(agent_id, None, "Done")
        return {"messages": state["messages"] + [HumanMessage(content=json.dumps(out, default=str), name=agent_id)],
                "data": data}

    agent.__name__ = f"{module_key}_agent"
    return agent


valuation_quality_agent = _make_agent("valuation_quality")
growth_earnings_agent = _make_agent("growth_earnings")
technical_momentum_agent = _make_agent("technical_momentum")
sentiment_positioning_agent = _make_agent("sentiment_positioning")
news_catalyst_agent = _make_agent("news_catalyst")
ai_industry_macro_agent = _make_agent("ai_industry_macro")
