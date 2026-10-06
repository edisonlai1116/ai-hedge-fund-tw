"""Portfolio Manager：先對全部股票做 cross-sectional ranking，再產生 buy / sell / hold。

舊版每檔股票獨立丟給 LLM（只看到 signal + confidence）決定買賣；新版：
  1. 由 src.ranking 取得同一批股票的完整評分表（現價、各模組分數、估值、成長、財報加速、AI 主題、
     催化、技術、相對強度、過熱、風險、持股、成本、組合曝險）。
  2. 決策是確定性的：先處理賣出/減碼/輪動（SELL A → BUY B），再依排名與目標權重買進。
     BUY_ON_PULLBACK 只在價格進入進場區時成交（不追高）。
  3. LLM（可選）只負責把表格寫成解讀文字，不改任何數字或動作。
"""
import json
import os

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress


class PortfolioDecision(BaseModel):
    action: Literal["buy", "sell", "short", "cover", "hold"]
    quantity: int = Field(description="Number of shares to trade")
    confidence: int = Field(description="Confidence 0-100")
    reasoning: str = Field(description="Reasoning for the decision")


class PortfolioManagerOutput(BaseModel):
    decisions: dict[str, PortfolioDecision] = Field(description="Dictionary of ticker to trading decisions")


##### Portfolio Management Agent #####
def portfolio_management_agent(state: AgentState, agent_id: str = "portfolio_manager"):
    from src.agents.quant_modules import get_ranking

    portfolio = state["data"]["portfolio"]
    analyst_signals = state["data"]["analyst_signals"]
    tickers = state["data"]["tickers"]
    end_date = state["data"]["end_date"]

    risk_manager_id = f"risk_management_agent_{agent_id.split('_')[-1]}" if agent_id.startswith("portfolio_manager_") else "risk_management_agent"
    risk = analyst_signals.get(risk_manager_id, {})
    current_prices, max_shares = {}, {}
    for t in tickers:
        rd = risk.get(t, {})
        current_prices[t] = float(rd.get("current_price", 0.0))
        lim = float(rd.get("remaining_position_limit", 0.0))
        max_shares[t] = int(lim // current_prices[t]) if current_prices[t] > 0 else 0
    state["data"]["current_prices"] = current_prices

    progress.update_status(agent_id, None, "Cross-sectional ranking")
    ranking = get_ranking(tickers, end_date, portfolio)
    table = build_ranking_table(ranking, tickers, analyst_signals, portfolio, current_prices)
    state["data"]["ranking_table"] = table

    progress.update_status(agent_id, None, "Deciding (ranking → actions)")
    decisions = decide_from_ranking(table, ranking, portfolio, current_prices, max_shares)
    decisions = _add_llm_interpretation(decisions, table, ranking, state, agent_id)
    result = PortfolioManagerOutput(decisions=decisions)

    message = HumanMessage(content=json.dumps({t: d.model_dump() for t, d in result.decisions.items()}), name=agent_id)
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning({"ranking": [{k: r[k] for k in ("rank", "ticker", "opportunity", "status")} for r in table],
                              "rotations": ranking.get("rotations"),
                              "decisions": {t: d.model_dump() for t, d in result.decisions.items()}}, "Portfolio Manager")
    progress.update_status(agent_id, None, "Done")
    return {"messages": state["messages"] + [message], "data": state["data"]}


def build_ranking_table(ranking: dict, tickers: list[str], analyst_signals: dict, portfolio: dict,
                        current_prices: dict) -> list[dict]:
    """PM 一次看到全部股票的完整資訊（不是只有 signal + confidence）。"""
    positions = portfolio.get("positions", {}) or {}
    equity = _equity(portfolio, current_prices)
    by_ticker = {r["ticker"].split(".")[0]: r for r in ranking.get("ranking", [])}
    rows = []
    for t in tickers:
        r = by_ticker.get(t.split(".")[0], {})
        pos = positions.get(t, {})
        long_sh = int(pos.get("long", 0) or 0)
        px = current_prices.get(t) or r.get("price") or 0.0
        modules = {a: {"signal": s.get(t, {}).get("signal"), "score": s.get(t, {}).get("score"),
                       "confidence": s.get(t, {}).get("confidence")}
                   for a, s in analyst_signals.items() if not a.startswith("risk_management_agent") and t in s}
        rows.append({
            "ticker": t, "rank": r.get("batch_rank"), "price": px, "opportunity": r.get("opportunity"),
            "status": r.get("status", "WAIT"), "flags": r.get("flags", []), "scores": r.get("scores", {}),
            "theme": r.get("theme"), "ai_cycle": (r.get("ai") or {}).get("cycle_position"),
            "valuation_class": r.get("valuation_class"), "entry": r.get("entry", {}), "target_weight": r.get("target_weight", 0.0),
            "narrative": r.get("narrative", {}), "modules": modules,
            "position_shares": long_sh, "cost_basis": pos.get("long_cost_basis"),
            "position_weight": (long_sh * px / equity) if equity > 0 else 0.0,
        })
    rows.sort(key=lambda x: (x["rank"] is None, x["rank"] or 0))
    return rows


def _equity(portfolio: dict, prices: dict) -> float:
    eq = float(portfolio.get("cash", 0.0))
    for t, p in (portfolio.get("positions") or {}).items():
        px = prices.get(t, 0.0)
        eq += (p.get("long", 0) or 0) * px - (p.get("short", 0) or 0) * px
    return eq


def decide_from_ranking(table: list[dict], ranking: dict, portfolio: dict, prices: dict, max_shares: dict) -> dict:
    equity = _equity(portfolio, prices)
    cash = float(portfolio.get("cash", 0.0))
    decisions: dict[str, PortfolioDecision] = {}
    rotations = ranking.get("rotations", [])
    rot_sell: dict[str, float] = {}
    for rot in rotations:
        k = rot["sell"].split(".")[0]
        rot_sell[k] = max(rot_sell.get(k, 0.0), rot["fraction_of_A"])

    # 1) 賣出 / 減碼 / 輪動（先釋放資金）
    for row in table:
        t, sh, px = row["ticker"], row["position_shares"], row["price"]
        if sh <= 0 or px <= 0:
            continue
        frac, why = 0.0, ""
        if row["status"] == "SELL":
            frac, why = 1.0, "SELL：" + (row["narrative"].get("key_risk") or "機會分數與相對強度轉弱")
        elif row["status"] == "PARTIAL_PROFIT":
            frac, why = 0.33, "PARTIAL_PROFIT：短線過熱且估值不便宜，先了結三分之一"
        if rot_sell.get(t.split(".")[0], 0.0) > frac:
            pair = next(r for r in rotations if r["sell"].split(".")[0] == t.split(".")[0])
            frac = rot_sell[t.split(".")[0]]
            why = f"{pair['type']}：賣出 {t} → 買進 {pair['buy']}（rotation_score {pair['rotation_score']:+.3f}）"
        qty = int(round(sh * frac))
        if qty > 0:
            decisions[t] = PortfolioDecision(action="sell", quantity=min(qty, sh), confidence=_conf(row), reasoning=_reason(row, why))
            cash += qty * px

    # 2) 依排名買進（目標權重 × 權益，受風控上限與現金限制）
    for row in table:
        t, px = row["ticker"], row["price"]
        if t in decisions or px <= 0:
            continue
        status, entry = row["status"], row.get("entry") or {}
        want = (row.get("target_weight") or 0.0) * equity - row["position_shares"] * px
        buyable = status == "BUY_NOW" or (status == "BUY_ON_PULLBACK" and _in_zone(px, entry))
        if buyable and want > px and row["position_shares"] == 0:
            qty = min(int(want // px), int(cash // px), max_shares.get(t, 0))
            if qty > 0:
                why = f"{status}：排名 #{row['rank']}、機會分數 {row['opportunity']}、目標權重 {row['target_weight']:.1%}"
                decisions[t] = PortfolioDecision(action="buy", quantity=qty, confidence=_conf(row), reasoning=_reason(row, why))
                cash -= qty * px
                continue
        why = {"BUY_ON_PULLBACK": f"等拉回：Zone 1 {(entry.get('zone_1') or {}).get('high')} 以下才買",
               "WAIT": "機會分數不足，觀望", "HOLD": "續抱", "HOLD_CORE": "核心續抱"}.get(status, status)
        decisions[t] = PortfolioDecision(action="hold", quantity=0, confidence=_conf(row), reasoning=_reason(row, why))
    return decisions


def _in_zone(px: float, entry: dict) -> bool:
    z = entry.get("zone_1") or {}
    return bool(z) and px <= z.get("high", 0)


def _conf(row: dict) -> int:
    o = row.get("opportunity")
    return int(min(100, max(0, abs(o - 50) * 2))) if o is not None else 0


def _reason(row: dict, why: str) -> str:
    return (f"#{row['rank']} opp={row['opportunity']} {row['status']} | {why}")[:240]


def _add_llm_interpretation(decisions: dict, table: list[dict], ranking: dict, state: AgentState, agent_id: str) -> dict:
    """LLM 只寫解讀（衝突、情境），不改動作/數量。環境變數 PM_LLM_INTERPRET=0 可關閉。"""
    if os.environ.get("PM_LLM_INTERPRET", "1") == "0":
        return decisions
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from src.utils.llm import call_llm

        class Notes(BaseModel):
            notes: dict[str, str] = Field(description="ticker -> 1 sentence interpretation (<=80 chars)")

        compact = [{"t": r["ticker"], "rank": r["rank"], "opp": r["opportunity"], "status": r["status"],
                    "scores": {k: (round(v) if isinstance(v, (int, float)) else v) for k, v in (r["scores"] or {}).items()
                               if k in ("momentum_rank", "earnings_acceleration", "valuation", "catalyst", "overextension", "risk")},
                    "action": decisions[r["ticker"]].action} for r in table if r["ticker"] in decisions]
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You explain portfolio decisions. Numbers and actions are FINAL and computed by a quantitative model; "
                       "do not change them. For each ticker write one short sentence (<=80 chars, Traditional Chinese) "
                       "on the main conflict or scenario. Return JSON only."),
            ("human", "Regime: {regime}\nRanking: {table}\nFormat: {{\"notes\": {{\"TICKER\": \"...\"}}}}"),
        ]).invoke({"regime": ranking.get("regime", {}).get("regime"), "table": json.dumps(compact, ensure_ascii=False)})
        out = call_llm(prompt=prompt, pydantic_model=Notes, agent_name=agent_id, state=state,
                       default_factory=lambda: Notes(notes={}))
        for t, note in (out.notes or {}).items():
            if t in decisions and note:
                d = decisions[t]
                decisions[t] = PortfolioDecision(action=d.action, quantity=d.quantity, confidence=d.confidence,
                                                 reasoning=(d.reasoning + " | " + note)[:300])
    except Exception:
        pass
    return decisions
