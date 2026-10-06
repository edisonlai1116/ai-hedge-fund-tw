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
