"""大跌買訊調整（2026-07-22 檢討）的回歸測試。

背景：7/17~7/19 CRWV/NBIS/SNDK/群聯/台達電 回檔時全被判「今天不要買」且掉出 Top 50，
7/21 全數反彈。根因：
1) 每日管線 fetch_fundamentals=False → fundamental_score 恆 0，AI 主線承接路徑永遠不觸發；
2) 恐慌殺盤跌破年線即視為「長線結構已壞」；
3) 動能因子在大跌時把優質股擠出 Top 50，錯殺買點無處呈現。
"""
from src.pipeline.daily_report import dip_radar_rows
from src.simple_signal import (
    BUY_SMALL,
    NO_BUY,
    derive_today_plan,
    map_ai_chain_and_bottleneck,
)


def _plan(**overrides):
    """跑 derive_today_plan 的共用預設參數（偏空、深度回檔的典型盤面）。"""
    kwargs = dict(
        latest_close=100.0,
        ma50=110.0,
        atr14=4.0,
        rsi14=40.0,
        bias="偏空",
        buy_strength="不建議進場",
        buy_zone="88.00-93.00",
        sell_zone="120.00-130.00",
        stop_loss="82.00-85.00",
        fundamental_score=0,
        valuation_gap_pct=None,
        ai_chain_layer=None,
        ma120=112.0,
        long_term_blocked=False,
        drawdown_from_high_pct=-15.0,
        ma20=105.0,
    )
    kwargs.update(overrides)
    return derive_today_plan(**kwargs)


def test_ai_mainline_dip_buys_without_fundamentals():
    # 每日管線抓不到基本面（fundamental_score=0）時，AI 主線對照表內的股票
    # 在恐慌超跌下應給「小量承接」而不是因 F-Score 門檻死鎖成「不要買」。
    out = _plan(ai_chain_layer="⚡ 資料中心基礎設施 DC Infra", fundamentals_available=False)
    assert out[0] == BUY_SMALL


def test_ai_mainline_still_gated_when_fundamentals_present_and_weak():
    # 基本面「真的有抓到」且很差（F-Score 1，觸發價值陷阱防護）時，維持原本的嚴格門檻：
    # 不因對照表身分就承接（年線之下、無其他品質證據 → 不要買）。
    out = _plan(
        ai_chain_layer="⚡ 資料中心基礎設施 DC Infra",
        fundamentals_available=True,
        fundamental_score=1,
        rsi14=47.0,               # 未達恐慌超跌
        drawdown_from_high_pct=-5.0,
    )
    assert out[0] == NO_BUY


def test_rising_ma120_pierced_still_quality_dip():
    # 年線仍上揚時，恐慌殺破年線 10% 以內應視為「長線結構未壞」→ 承接機會。
    out = _plan(
        latest_close=103.0,      # 年線 112 → 約 -8%（舊定義 0.97 門檻會判結構已壞）
        ma120=112.0,
        ma120_rising=True,
        rsi14=42.0,
    )
    assert out[0] == BUY_SMALL


def test_three_day_crash_triggers_panic_oversold():
    # RSI 未達 45、距 60 日高點未達 -8%，但 3 日急跌 -7% → 也應觸發恐慌超跌承接。
    out = _plan(
        latest_close=109.0,
        ma120=110.0,             # 站在年線附近（品質條件成立）
        rsi14=50.0,
        drawdown_from_high_pct=-6.0,
        drop_3d_pct=-7.0,
    )
    assert out[0] == BUY_SMALL


def test_deep_oversold_reversal_without_valuation_data():
    # 估值資料抓不到時，深跌 -35%＋RSI<38 仍可走「深度超跌反轉」最小倉位路徑。
    out = _plan(
        rsi14=34.0,
        drawdown_from_high_pct=-41.0,
        ma120=130.0,             # 遠低於年線、年線未上揚 → 非 quality dip，靠反轉路徑
        ma120_rising=False,
        valuation_gap_pct=None,
    )
    assert out[0] == BUY_SMALL
    assert "深度超跌" in out[2]


def test_long_term_blocked_still_refuses():
    # 長線虧損閘門否決者，再怎麼超跌都不承接（避免接真正向下的刀）。
    out = _plan(
        ai_chain_layer="💾 儲存與記憶體 Memory & Storage",
        fundamentals_available=False,
        long_term_blocked=True,
        rsi14=30.0,
        drawdown_from_high_pct=-40.0,
    )
    assert out[0] == NO_BUY


def test_chain_map_covers_review_tickers():
    # 7/17~19 檢討點名的標的都要在 AI 主線對照表內（含上櫃群聯 8299.TWO）。
    for sym in ["CRWV", "NBIS", "SNDK", "8299.TWO", "2308.TW", "2408.TW", "MRVL"]:
        layer, _ = map_ai_chain_and_bottleneck(sym, "")
        assert layer is not None, sym


def _row(ticker, dd, rsi, score=55):
    return {
        "ticker": ticker,
        "symbol": ticker,
        "buy_score": score,
        "technical": {"drawdown_from_high_pct": dd, "rsi14": rsi, "today_action": "x", "buy_zone": "1~2"},
    }


def test_dip_radar_picks_quality_dips_and_sorts_by_drawdown():
    rows = [
        _row("SNDK", -12.0, 38.0),          # 主線＋超跌 → 入選
        _row("CRWV", -25.0, 35.0),          # 主線＋更深 → 排最前
        _row("NVDA", -2.0, 55.0),           # 主線但未超跌 → 不入選
        _row("GLUE", -30.0, 25.0),          # 非主線（生技動能股）→ 不入選
    ]
    out = dip_radar_rows(rows)
    tickers = [r["ticker"] for r in out]
    assert tickers == ["CRWV", "SNDK"]
    assert out[0]["dip_reason"]  # 有回檔原因文字
    assert out[0]["ai_chain_layer"]


def test_dip_radar_skips_rows_without_technical():
    out = dip_radar_rows([{"ticker": "CRWV", "technical": None}])
    assert out == []
