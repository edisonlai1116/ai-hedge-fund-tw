"""單一策略（Sharpe 動能輪動）API：排名、持股評估。資料來自每日 GitHub Actions 產生的 docs/data/strategy.json。"""
from __future__ import annotations

import json
import os
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/strategy", tags=["strategy"])

_DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "docs", "data")
_FX_CACHE: dict = {"ts": 0.0, "v": 32.0}


def _report() -> dict:
    path = os.path.join(_DOCS, "strategy.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"策略排名尚未產生：{exc}") from exc


def _fx() -> float:
    if time.time() - _FX_CACHE["ts"] > 3600:
        from src.strategy.momentum import usd_twd
        _FX_CACHE.update(ts=time.time(), v=usd_twd())
    return _FX_CACHE["v"]


class HoldingIn(BaseModel):
    ticker: str
    cost: float = Field(default=0, ge=0)
    shares: float = Field(default=0, ge=0)


class EvaluateRequest(BaseModel):
    holdings: list[HoldingIn] = Field(default_factory=list)


@router.get("/report")
def strategy_report() -> dict:
    return _report()


@router.get("/advice")
def daily_advice() -> dict:
    """每日建議（市場情緒 + 今日該不該動）：每日 GitHub Actions 產生 docs/data/daily_advice.json。"""
    path = os.path.join(_DOCS, "daily_advice.json")
    try:
        with open(path, encoding="utf-8") as f:
            advice = json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"每日建議尚未產生：{exc}") from exc
    rep = _report()
    advice["sentiment"] = rep.get("sentiment")
    log = os.path.join(_DOCS, "advice_log.jsonl")
    try:
        with open(log, encoding="utf-8") as f:
            advice["history"] = [json.loads(x) for x in f if x.strip()][-30:][::-1]
    except Exception:
        advice["history"] = []
    try:
        with open(os.path.join(_DOCS, "strategy_backtest.json"), encoding="utf-8") as f:
            bt = json.load(f)
        advice["evidence"] = {m: {"sentiment_study": d.get("sentiment_study"), "sentiment_ablation": d.get("sentiment_ablation")}
                              for m, d in bt.get("markets", {}).items()}
    except Exception:
        advice["evidence"] = None
    return advice


@router.post("/evaluate")
def evaluate(request: EvaluateRequest) -> dict:
    from src.strategy.momentum import evaluate_holdings
    if len(request.holdings) > 80:
        raise HTTPException(status_code=400, detail="持股最多 80 檔。")
    try:
        return evaluate_holdings([h.model_dump() for h in request.holdings], _report(), _fx())
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"持股評估失敗：{exc}") from exc


@router.get("/lookup")
def lookup(symbols: str = "") -> list[dict]:
    from src.strategy.momentum import lookup_symbols
    tickers = [t for t in symbols.replace("，", ",").replace(" ", ",").split(",") if t.strip()][:10]
    if not tickers:
        raise HTTPException(status_code=400, detail="請輸入股票代號。")
    try:
        return lookup_symbols(tickers, _report())
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"查詢失敗：{exc}") from exc


# ---------------------------------------------------------------------------
# 機會評分（cross-sectional ranking）
# ---------------------------------------------------------------------------
class RankingRequest(BaseModel):
    symbols: list[str] = Field(default_factory=list)
    holdings: list[HoldingIn] = Field(default_factory=list)


@router.get("/opportunity")
def opportunity_report() -> dict:
    """每日自動產生的機會評分卡（驗收清單 + 策略前段班）。"""
    path = os.path.join(_DOCS, "opportunity.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"機會評分尚未產生：{exc}") from exc


@router.get("/signal-accuracy")
def signal_accuracy() -> dict:
    path = os.path.join(_DOCS, "signal_accuracy.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"n_signals": 0, "note": "尚無訊號紀錄"}


@router.post("/opportunity/live")
def opportunity_live(request: RankingRequest) -> dict:
    """即時對指定股票（含持股）做 cross-sectional ranking；持股會產生輪動建議。"""
    from src.ranking.engine import rank_stocks
    syms = [s.strip().upper() for s in request.symbols if s.strip()]
    holdings = {h.ticker.strip().upper(): {"shares": h.shares, "cost": h.cost} for h in request.holdings if h.ticker.strip()}
    syms = list(dict.fromkeys(syms + list(holdings)))
    if not syms:
        raise HTTPException(status_code=400, detail="請輸入股票代號。")
    if len(syms) > 25:
        raise HTTPException(status_code=400, detail="一次最多 25 檔。")
    try:
        out = rank_stocks(syms, holdings=holdings)
        for r in out["ranking"]:
            r.pop("fundamentals", None)
        return out
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"評分失敗：{exc}") from exc
