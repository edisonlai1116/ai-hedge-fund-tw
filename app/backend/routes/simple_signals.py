import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.simple_signal import analyze_symbol_with_data, analyze_symbols_batch_with_data
from src.sp500_daily import MarketRegime, build_signal_backtest, get_sp500_daily_top_picks
from src.ai_mainline_backtest import AI_MAINLINE_UNIVERSE, run_ai_mainline_backtest


router = APIRouter(prefix="/simple-signals", tags=["simple-signals"])


class SimpleSignalRequest(BaseModel):
    ticker: str = Field(..., description="US ticker like AAPL or TW ticker like 2330")
    market: str | None = Field(default=None, description="Optional market hint: us or tw")
    period: str = Field(default="3y", description="Yahoo Finance lookback period")
    use_ai_committee: bool = Field(default=False, description="Enable AI committee scoring")
    committee_model: str = Field(default="gemma4:e4b", description="AI committee model name")


class SimpleSignalBatchRequest(BaseModel):
    tickers: list[str] = Field(..., description="List of tickers to analyze")
    market: str | None = Field(default=None, description="Optional market hint: us or tw")
    period: str = Field(default="3y", description="Yahoo Finance lookback period")
    use_ai_committee: bool = Field(default=False, description="Enable AI committee scoring")
    committee_model: str = Field(default="gemma4:e4b", description="AI committee model name")


class HoldingReviewItemRequest(BaseModel):
    ticker: str = Field(..., description="Ticker symbol")
    market: str | None = Field(default=None, description="Optional market hint: us or tw")
    cost_basis: float | None = Field(default=None, ge=0, description="Average cost basis")
    shares: float | None = Field(default=None, ge=0, description="Position size")


class HoldingReviewRequest(BaseModel):
    holdings: list[HoldingReviewItemRequest] = Field(..., description="Holdings to review")
    period: str = Field(default="3y", description="Yahoo Finance lookback period")
    use_ai_committee: bool = Field(default=False, description="Enable AI committee scoring")
    committee_model: str = Field(default="gemma4:e4b", description="AI committee model name")


class BacktestResponse(BaseModel):
    sample_size: int
    win_rate_5d: float
    avg_return_5d: float
    win_rate_20d: float
    avg_return_20d: float
    win_rate_60d: float
    avg_return_60d: float
    max_drawdown_20d: float
    downside_rate_20d: float
    confidence_score: int
    calibration_note: str


class TimelineTrade(BaseModel):
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_pct: float
    days_held: int
    outcome: str


class TimelineBacktestResponse(BaseModel):
    total_trades: int
    win_rate: float
    avg_return: float
    cumulative_return: float
    trades_log: list[TimelineTrade]


class SimpleSignalResponse(BaseModel):
    symbol: str
    latest_close: float
    ma20: float
    ma50: float
    ma120: float = 0.0
    rsi14: float
    atr14: float
    support: float
    resistance: float
    bias: str
    buy_zone: str
    sell_zone: str
    stop_loss: str
    reason: str
    rule_score: int
    ai_score: int | None
    composite_score: int
    buy_strength: str
    today_action: str
    today_entry_zone: str
    today_note: str
    today_exit_action: str
    today_exit_zone: str
    today_exit_note: str
    expected_return_pct: float
    risk_reward_ratio: float
    holding_days_estimate: int
    holding_window: str
    backtest: BacktestResponse | None
    committee_summary: str | None
    committee_model: str | None
    ai_enabled: bool
    ai_available: bool
    ai_error: str | None
    chart: list[dict]
    agents: list[dict]
    horizons: list[dict]
    fundamental_score: int = 0
    graham_number: float | None = None
    macd_value: float = 0.0
    macd_signal: float = 0.0
    macd_hist: float = 0.0
    bb_width: float = 0.0
    candlestick_pattern: str = "無"
    kelly_position_pct: float = 0.0
    decision_assistance: str = ""
    timeline_backtest: TimelineBacktestResponse | None = None
    ai_chain_layer: str | None = None
    critical_bottleneck: str | None = None
    novice_rating: str | None = None
    investingpro_fair_value: float | None = None
    valuation_gap_pct: float | None = None
    analyst_target_price: float | None = None
    warren_ai_momentum: str | None = None
    investingpro_models: list[dict] | None = None
    cognitive_temperature_gap: str | None = None
    geopolitical_timing_advice: str | None = None
    value_trap_risk: str | None = None
    price_forecast: dict | None = None
    long_term_risk: dict | None = None


class HoldingReviewResponse(BaseModel):
    symbol: str
    cost_basis: float | None
    shares: float | None
    latest_close: float
    pnl_pct: float | None
    market_value: float | None
    unrealized_pnl: float | None
    verdict: str
    urgency: str
    trim_ratio: str
    holding_reason: str
    protective_stop: str
    signal: SimpleSignalResponse


class SP500DailyScanRequest(BaseModel):
    period: str = Field(default="3y", description="Yahoo Finance lookback period")
    limit: int = Field(default=50, ge=10, le=100, description="Number of picks to return")
    use_ai_committee: bool = Field(default=False, description="Enable AI committee scoring for shortlisted picks")
    committee_model: str = Field(default="gemma4:e4b", description="AI committee model name")
    market: str = Field(default="us", description="Market hint: us or tw")
    scan_type: str = Field(default="optimal", description="Scan mode: optimal or lagging_value")


class AiMainlineBacktestRequest(BaseModel):
    market: str = Field(default="us", description="Market hint: us or tw")
    period: str = Field(default="5y", description="Yahoo Finance lookback period for long-term backtest")
    tickers: list[str] | None = Field(
        default=None, description="Optional custom AI universe; defaults to curated AI mainline stocks"
    )
    initial_capital: float = Field(default=100000.0, gt=0, description="Starting capital")
    max_positions: int = Field(default=8, ge=1, le=20, description="Max concurrent holdings")
    take_profit_pct: float = Field(default=35.0, gt=0, le=200, description="Target take-profit percentage")
    trailing_stop_pct: float = Field(default=18.0, gt=0, le=60, description="Trailing stop percentage")
    max_holding_days: int = Field(default=126, ge=60, le=900, description="Max holding days (~6 months default; backtest-optimal band)")


class AiMainlineTradeResponse(BaseModel):
    symbol: str
    layer: str | None
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_pct: float
    days_held: int
    outcome: str
    pnl: float


class AiMainlineLayerResponse(BaseModel):
    layer: str
    trades: int
    win_rate: float
    avg_return_pct: float
    net_pnl: float
    contribution_pct: float


class AiMainlineEquityPoint(BaseModel):
    date: str
    equity: float
    return_pct: float


class AiMainlineBacktestResponse(BaseModel):
    market: str
    start_date: str
    end_date: str
    years: float
    initial_capital: float
    final_equity: float
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    avg_holding_days: float
    avg_trade_return_pct: float
    benchmark_symbol: str
    benchmark_return_pct: float
    excess_return_pct: float
    universe: list[str]
    layer_breakdown: list[AiMainlineLayerResponse]
    equity_curve: list[AiMainlineEquityPoint]
    trades_log: list[AiMainlineTradeResponse]
    note: str


class SP500DailyPickResponse(BaseModel):
    symbol: str
    company_name: str
    sector: str
    latest_close: float
    bias: str
    reason: str
    rule_score: int
    ai_score: int | None
    composite_score: int
    technical_score: int
    news_score: int
    fundamental_score: int
    regime_score: int
    backtest_score: int
    daily_score: int
    buy_strength: str
    action_label: str
    buy_urgency: str
    position_sizing: str
    today_action: str
    today_entry_zone: str
    today_note: str
    today_exit_action: str
    today_exit_zone: str
    today_exit_note: str
    expected_return_pct: float
    risk_reward_ratio: float
    holding_days_estimate: int
    holding_window: str
    buy_zone: str
    sell_zone: str
    stop_loss: str
    committee_summary: str | None
    committee_model: str | None
    ai_enabled: bool
    ai_available: bool
    ai_error: str | None
    chart: list[dict]
    agents: list[dict]
    horizons: list[dict]
    headline_count: int
    headline_summary: str
    backtest: BacktestResponse
    sector_score: int
    is_main_line: bool
    is_sector_leader: bool
    sector_boost: int
    is_dark_horse: bool
    dark_horse_boost: int
    ai_chain_layer: str | None = None
    critical_bottleneck: str | None = None
    novice_rating: str | None = None
    investingpro_fair_value: float | None = None
    valuation_gap_pct: float | None = None
    analyst_target_price: float | None = None
    warren_ai_momentum: str | None = None
    investingpro_models: list[dict] | None = None
    cognitive_temperature_gap: str | None = None
    geopolitical_timing_advice: str | None = None
    value_trap_risk: str | None = None
    price_forecast: dict | None = None
    long_term_risk: dict | None = None


class MarketRegimeResponse(BaseModel):
    vix_close: float
    vix_regime: str
    fear_greed_score: int
    fear_greed_label: str
    fear_greed_source: str
    spy_drawdown_pct: float
    spy_distance_ma200_pct: float
    regime_score: int
    action: str
    risk_budget: str
    summary: str
    backtest_win_rate_5d: float
    backtest_avg_return_5d: float
    backtest_win_rate_20d: float
    backtest_avg_return_20d: float


class SectorAnalysisResponse(BaseModel):
    name: str
    score: int
    is_main_line: bool
    avg_return_5d: float
    member_count: int
    top_members: list[str]
    market_role: str


class SP500DailyScanResponse(BaseModel):
    market_regime: MarketRegimeResponse
    picks: list[SP500DailyPickResponse]
    sectors: list[SectorAnalysisResponse] = []
    generated_at: str


def _neutral_regime() -> MarketRegime:
    return MarketRegime(
        vix_close=0.0,
        vix_regime="中性",
        fear_greed_score=50,
        fear_greed_label="中性",
        fear_greed_source="單股回測預設值",
        spy_drawdown_pct=0.0,
        spy_distance_ma200_pct=0.0,
        regime_score=50,
        action="中性偏分批",
        risk_budget="中性部位",
        summary="單股回測採用中性市場環境，不額外放大或縮小訊號。",
        backtest_win_rate_5d=50.0,
        backtest_avg_return_5d=0.0,
        backtest_win_rate_20d=50.0,
        backtest_avg_return_20d=0.0,
    )


def _attach_backtest(report, frame) -> None:
    pass


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_holding_verdict(
    report, cost_basis: float | None
) -> tuple[str, str, str, str]:
    # 依「未來半年（6 個月）統計期望報酬」決定 續抱 / 減碼 / 出場——以「半年內報酬最好」為準：
    #   1) 風險閘門（跌破硬性保護停損 / 長線虧損閘門 / 半年期望報酬明顯為負）→ 賣出期望優於續抱 → 全數出場。
    #   2) 半年上檔有限又短線過熱 → 分批鎖利（減碼比例看過熱程度）。
    #   3) 半年期望報酬仍正向 → 續抱（續抱期望優於賣出）。
    # 不再用均線幾何（MA20<MA50）或帳面獲利% 單獨觸發賣出，避免與「個股分析」結論互相矛盾。
    close = report.latest_close
    ma120 = getattr(report, "ma120", 0.0) or 0.0
    rsi = report.rsi14
    bias = report.bias
    exit_action = getattr(report, "today_exit_action", "") or ""
    pnl_pct = None if cost_basis in (None, 0) else ((close / cost_basis) - 1) * 100

    # 未來半年（126 個交易日）統計期望報酬 %
    pf = report.price_forecast if isinstance(getattr(report, "price_forecast", None), dict) else {}
    exp6: float | None = None
    for h in pf.get("horizons", []) or []:
        if h.get("days") == 126:
            exp6 = _to_float(h.get("expected_return_pct"))
            break

    ltr = report.long_term_risk if isinstance(getattr(report, "long_term_risk", None), dict) else {}
    ltr_blocked = bool(ltr.get("blocked"))
    exp12 = _to_float(ltr.get("expected_return_12m_pct"))

    # 解析硬性保護停損價（防守底線；跌破代表下檔風險已吃掉半年期望報酬）。
    stop_val = None
    try:
        if report.stop_loss and isinstance(report.stop_loss, str):
            stop_val = float(report.stop_loss.split("-")[0].strip())
        elif report.stop_loss:
            stop_val = float(report.stop_loss)
    except Exception:
        stop_val = None
    stop_broken = stop_val is not None and close < stop_val

    overbought = rsi >= 75
    extreme_ob = rsi >= 82

    def _exit(reason_core: str) -> tuple[str, str, str, str]:
        if pnl_pct is not None and pnl_pct < 0:
            return ("停損出場", "高", "100%", "停損出場：" + reason_core)
        return ("獲利了結", "高", "100%", "獲利了結：" + reason_core)

    # ---- 1) 風險閘門 → 全數出場（賣出的期望報酬優於續抱）----
    if stop_broken:
        return _exit(
            f"股價已跌破保護性停損價（{stop_val:.2f} 元），下檔風險已吃掉未來半年的期望報酬，先全數出場保存實力。"
        )
    if ltr_blocked:
        return _exit(
            f"長線虧損閘門觸發（{ltr.get('note', '12 個月統計期望報酬偏負')}），長抱半年以上仍難轉正，建議全數出場。"
        )
    if exp6 is not None and (exp6 <= -5 or (exp6 < 0 and exp12 is not None and exp12 < 0)):
        return _exit(
            f"未來半年統計期望報酬為 {exp6:.1f}%（偏負），續抱期望報酬低於賣出，建議全數出場、資金轉進更強標的。"
        )

    # ---- 2) 半年上檔有限又短線過熱 → 分批鎖利 ----
    weak6 = exp6 is not None and exp6 < 4
    if extreme_ob:
        upside = f"、未來半年期望報酬僅 {exp6:.1f}% 上檔有限" if exp6 is not None else ""
        return (
            "分批減碼", "中", "50%",
            f"RSI {rsi:.0f} 嚴重超買、短線過熱{upside}；先了結一半鎖利、降低急回風險，其餘隨趨勢續抱。",
        )
    if overbought and weak6:
        return (
            "觀察減碼", "中", "30%",
            f"RSI {rsi:.0f} 偏熱且未來半年期望報酬僅 {exp6:.1f}%、上檔空間有限，"
            f"小幅減碼約三成鎖利，核心部位續抱觀察。",
        )

    # ---- 3) 半年期望報酬仍正向 → 續抱（續抱期望優於賣出）----
    caution = ""
    if ma120 > 0 and close < ma120:
        caution = f"　註：股價仍在長線 MA120（{ma120:.2f} 元）之下，屬中段整理，留意而非賣出訊號。"
    if exp6 is not None and exp6 >= 15 and bias != "偏空":
        return (
            "強勢續抱", "低", "0%",
            f"未來半年統計期望報酬高達 {exp6:.1f}%、趨勢未轉空，續抱期望報酬明顯優於賣出，抱牢即可。{caution}",
        )
    if exp6 is not None:
        return (
            "續抱觀察", "低", "0%",
            f"未來半年統計期望報酬為 {exp6:.1f}%（仍正向或持平），續抱期望報酬優於賣出；"
            f"無跌破停損、無長線虧損風險，續抱觀察、跌破保護價再處理。{caution}",
        )

    # ---- 4) 半年預測資料不足 → 回退引擎當日訊號，維持與個股分析一致 ----
    if exit_action == "今天賣出":
        return _exit("引擎當日訊號為『今天賣出』（趨勢偏空且跌破生命線），且無半年預測可佐證續抱。")
    if exit_action == "今天可小量賣":
        return ("觀察減碼", "中", "30%", "引擎當日訊號為『今天可小量賣』（短線偏熱），小幅減碼鎖利、其餘續抱。")
    return ("續抱觀察", "低", "0%", f"無明確賣出訊號（趨勢 {bias}、RSI {rsi:.0f}），續抱觀察、守住保護停損即可。")


# 注意：以下重活全部宣告為「同步 def」而非「async def」。
# FastAPI 會把同步 path operation 丟到外部 threadpool 執行，事件迴圈保持空閒，
# /healthz 才答得出來；否則單一 worker 的事件迴圈會被 yfinance 下載/回測卡死，
# Render 健康檢查逾時 → 重啟容器 → 進行中的請求收到 502。
@router.post("/analyze", response_model=SimpleSignalResponse)
def analyze_simple_signal(request: SimpleSignalRequest) -> SimpleSignalResponse:
    try:
        report, frame = analyze_symbol_with_data(
            request.ticker,
            request.market,
            request.period,
            use_ai_committee=request.use_ai_committee,
            committee_model=request.committee_model,
        )
        _attach_backtest(report, frame)
        return SimpleSignalResponse(**report.__dict__)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"分析 {request.ticker} 失敗：{exc}") from exc


@router.post("/analyze-batch", response_model=list[SimpleSignalResponse])
def analyze_simple_signal_batch(request: SimpleSignalBatchRequest) -> list[SimpleSignalResponse]:
    try:
        reports_with_frames, errors = analyze_symbols_batch_with_data(
            request.tickers,
            request.market,
            request.period,
            use_ai_committee=request.use_ai_committee,
            committee_model=request.committee_model,
        )
        if not reports_with_frames:
            raise ValueError("沒有任何股票分析成功。" + (" | " + " | ".join(errors[:5]) if errors else ""))

        reports: list[SimpleSignalResponse] = []
        for report, frame in reports_with_frames:
            _attach_backtest(report, frame)
            reports.append(SimpleSignalResponse(**report.__dict__))

        return sorted(reports, key=lambda item: item.composite_score, reverse=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"多檔分析失敗：{exc}") from exc


@router.post("/review-holdings", response_model=list[HoldingReviewResponse])
def review_holdings(request: HoldingReviewRequest) -> list[HoldingReviewResponse]:
    try:
        if not request.holdings:
            raise ValueError("請至少提供一筆持股。")

        results: list[HoldingReviewResponse] = []

        def process_holding(item):
            report, frame = analyze_symbol_with_data(
                item.ticker,
                item.market,
                request.period,
                use_ai_committee=request.use_ai_committee,
                committee_model=request.committee_model,
            )
            _attach_backtest(report, frame)
            verdict, urgency, trim_ratio, holding_reason = _build_holding_verdict(report, item.cost_basis)
            pnl_pct = None if item.cost_basis in (None, 0) else round(((report.latest_close / item.cost_basis) - 1) * 100, 2)
            market_value = None if item.shares is None else round(report.latest_close * item.shares, 2)
            unrealized_pnl = None
            if item.cost_basis is not None and item.shares is not None:
                unrealized_pnl = round((report.latest_close - item.cost_basis) * item.shares, 2)

            signal = SimpleSignalResponse(**report.__dict__)
            return HoldingReviewResponse(
                symbol=report.symbol,
                cost_basis=item.cost_basis,
                shares=item.shares,
                latest_close=report.latest_close,
                pnl_pct=pnl_pct,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                verdict=verdict,
                urgency=urgency,
                trim_ratio=trim_ratio,
                holding_reason=holding_reason,
                protective_stop=report.stop_loss,
                signal=signal,
            )

        logger = logging.getLogger("uvicorn.error")
        with ThreadPoolExecutor(max_workers=min(len(request.holdings), 8)) as executor:
            futures = [executor.submit(process_holding, item) for item in request.holdings]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    logger.error(f"持股健檢單個持股處理失敗: {exc}")

        if not results and request.holdings:
            raise ValueError("所有持股的健檢分析均失敗，請確認網路連線或輸入代碼是否正確。")

        # 排序優先級需與 _build_holding_verdict 實際輸出的判語一致：
        #   出場(3) > 減碼(2) > 續抱(1)；同級再依帳面損益絕對值大小排。
        def _verdict_rank(verdict: str) -> int:
            if verdict in {"停損出場", "獲利了結"}:
                return 3
            if verdict in {"分批減碼", "觀察減碼"}:
                return 2
            if verdict in {"強勢續抱", "續抱觀察"}:
                return 1
            return 0

        results.sort(
            key=lambda item: (
                _verdict_rank(item.verdict),
                0 if item.pnl_pct is None else abs(item.pnl_pct),
            ),
            reverse=True,
        )
        return results
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"持股檢查失敗：{exc}") from exc


@router.post("/sp500-daily-top", response_model=SP500DailyScanResponse)
def scan_sp500_daily_top(request: SP500DailyScanRequest) -> SP500DailyScanResponse:
    try:
        payload = get_sp500_daily_top_picks(
            period=request.period,
            limit=request.limit,
            use_ai_committee=request.use_ai_committee,
            committee_model=request.committee_model,
            market=request.market,
            scan_type=request.scan_type,
        )
        return SP500DailyScanResponse(**payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"每日名單掃描失敗：{exc}") from exc


@router.get("/ai-mainline-universe")
async def ai_mainline_universe() -> dict[str, list[str]]:
    """回傳預設 AI 產業鏈主線宇宙，方便前端顯示與快速帶入。"""
    return AI_MAINLINE_UNIVERSE


@router.post("/ai-mainline-backtest", response_model=AiMainlineBacktestResponse)
def ai_mainline_backtest(request: AiMainlineBacktestRequest) -> AiMainlineBacktestResponse:
    try:
        payload = run_ai_mainline_backtest(
            symbols=request.tickers,
            market=request.market,
            period=request.period,
            initial_capital=request.initial_capital,
            max_positions=request.max_positions,
            take_profit_pct=request.take_profit_pct,
            trailing_stop_pct=request.trailing_stop_pct,
            max_holding_days=request.max_holding_days,
        )
        return AiMainlineBacktestResponse(**payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"AI 主線長線回測失敗：{exc}") from exc
