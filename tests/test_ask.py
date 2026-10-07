"""AI 問答：提示詞內容、換現金順序、無金鑰時不呼叫外部服務（不連網）。"""
from src.strategy import ask


def _ev():
    H = lambda sym, act, val, rank, pnl=10.0: {"symbol": sym, "action": act, "value_twd": val, "rank": rank, "pnl_pct": pnl,  # noqa: E731
                                               "shares": 10, "cost": 1, "price": 1, "weight_pct": 10, "today": "不用動",
                                               "reason": "x", "day_change_pct": 0.1}
    return {"total_twd": 1_000_000, "day_pnl_twd": -1000, "day_change_pct": -0.1, "fx_usd_twd": 32,
            "holdings": [H("TOP", "續抱", 100_000, 3), H("CORE", "核心 ETF", 100_000, None), H("BIG", "減碼", 500_000, 40),
                         H("WEAK", "續抱", 100_000, 99), H("MID", "續抱", 100_000, 70), H("OUT", "賣出換股", 100_000, 80)],
            "low_entry_buys": {}}


def _report():
    rows = [{"symbol": s, "rank": r, "quality_tier": t} for s, r, t in
            (("TOP", 3, "A"), ("BIG", 40, "B"), ("WEAK", 99, "D"), ("MID", 70, "C"), ("OUT", 80, "C"))]
    return {"markets": {"us": {"rows": rows, "universe_size": 102}}, "drawdown_types": {}}


def test_raise_cash_order_sells_flagged_and_weak_first_keeps_leaders_last():
    plan = ask.raise_cash_plan(_ev(), _report())
    order = [x["symbol"] for x in plan]
    assert order[:2] == ["OUT", "BIG"]                          # 系統建議賣出 → 集中度減碼
    assert order.index("WEAK") < order.index("MID") < order.index("CORE") < order.index("TOP")
    big = next(x for x in plan if x["symbol"] == "BIG")
    assert big["suggest_trim_twd"] == 300_000                   # 減到總資產 20%


def test_prompt_contains_question_holdings_plan_and_guardrails():
    ctx = {"evaluation": _ev(), "cash_plan": ask.raise_cash_plan(_ev(), _report()),
           "advice": {"date": "2026-10-08", "headline": "今天不用動", "sentiment_guidance": [], "markets": {}},
           "sentiment": {"fear_greed": {"score": 40, "label": "恐懼"}, "vix": {"value": 15.7}}, "strategy": {"rule": "R"}}
    p = ask.build_prompt("滿倉要賣什麼換現金？", ctx)
    for must in ("滿倉要賣什麼換現金？", "| BIG |", "換現金的先後順序", "不要自行捏造", "不是投資建議", "恐懼貪婪指數 40"):
        assert must in p


def test_no_api_key_returns_error_without_network(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("不應連網")))
    out = ask.call_gemini("hi")
    assert "error" in out and "GOOGLE_API_KEY" in out["error"]


def test_gemini_falls_back_to_next_model_when_busy(monkeypatch):
    calls = []

    class R:
        def __init__(self, ok, payload=None):
            self.ok, self.status_code, self._p = ok, 200 if ok else 503, payload or {}

        def json(self):
            return self._p

    def fake_post(url, **kw):
        calls.append(url)
        if len(calls) == 1:
            return R(False)
        return R(True, {"candidates": [{"content": {"parts": [{"text": "答案"}]}}]})

    import requests
    monkeypatch.setattr(requests, "post", fake_post)
    out = ask.call_gemini("hi", api_key="k")
    assert out["answer"] == "答案" and len(calls) == 2 and out["model"] == ask.GEMINI_MODELS[1]
