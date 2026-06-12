const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export type SimpleSignalPayload = {
  ticker: string;
  market?: 'us' | 'tw' | '';
  period?: string;
  useAiCommittee?: boolean;
  committeeModel?: string;
};

export type HoldingReviewItemPayload = {
  ticker: string;
  market?: 'us' | 'tw' | '';
  cost_basis?: number;
  shares?: number;
};

export type HoldingReviewPayload = {
  holdings: HoldingReviewItemPayload[];
  period?: string;
  useAiCommittee?: boolean;
  committeeModel?: string;
};

export type HorizonView = {
  horizon: string;
  bias: string;
  entry_zone: string;
  take_profit_zone: string;
  stop_zone: string;
  summary: string;
};

export type AgentView = {
  key: string;
  name: string;
  signal: string;
  confidence: number;
  summary: string;
  historical_edge?: {
    sample_size: number;
    win_rate: number;
    avg_return: number;
    weight: number;
  };
};

export type ChartPoint = {
  date: string;
  close: number;
  ma20: number | null;
  ma50: number | null;
};

export type ForecastHorizon = {
  label: string;
  days: number;
  base: number;
  low: number;
  high: number;
  expected_return_pct: number;
  stance: string;
};

export type PriceForecast = {
  method: string;
  annualized_drift_pct: number;
  annualized_volatility_pct: number;
  verdict: string;
  verdict_reason: string;
  horizons: ForecastHorizon[];
};

export type LongTermRisk = {
  blocked: boolean;
  severity: string;
  note: string;
  expected_return_12m_pct: number | null;
  forecast_base_12m: number | null;
  forecast_high_12m?: number | null;
  history_cumulative_return_pct: number | null;
  history_win_rate_pct: number | null;
  history_trades: number;
};

export type BacktestSummary = {
  sample_size: number;
  win_rate_5d: number;
  avg_return_5d: number;
  win_rate_20d: number;
  avg_return_20d: number;
  win_rate_60d: number;
  avg_return_60d: number;
  max_drawdown_20d: number;
  downside_rate_20d: number;
  confidence_score: number;
  calibration_note: string;
};

export type TimelineTrade = {
  entry_date: string;
  exit_date: string;
  entry_price: number;
  exit_price: number;
  return_pct: number;
  days_held: number;
  outcome: string;
};

export type TimelineBacktestSummary = {
  total_trades: number;
  win_rate: number;
  avg_return: number;
  cumulative_return: number;
  trades_log: TimelineTrade[];
};

export type SimpleSignalResult = {
  symbol: string;
  latest_close: number;
  ma20: number;
  ma50: number;
  rsi14: number;
  atr14: number;
  support: number;
  resistance: number;
  bias: string;
  buy_zone: string;
  sell_zone: string;
  stop_loss: string;
  reason: string;
  rule_score: number;
  ai_score: number | null;
  composite_score: number;
  buy_strength: string;
  today_action: string;
  today_entry_zone: string;
  today_note: string;
  today_exit_action: string;
  today_exit_zone: string;
  today_exit_note: string;
  expected_return_pct: number;
  risk_reward_ratio: number;
  holding_days_estimate: number;
  holding_window: string;
  backtest: BacktestSummary | null;
  committee_summary: string | null;
  committee_model: string | null;
  ai_enabled: boolean;
  ai_available: boolean;
  ai_error: string | null;
  chart: ChartPoint[];
  agents: AgentView[];
  horizons: HorizonView[];
  fundamental_score?: number;
  graham_number?: number | null;
  macd_value?: number;
  macd_signal?: number;
  macd_hist?: number;
  bb_width?: number;
  candlestick_pattern?: string;
  kelly_position_pct?: number;
  decision_assistance: string;
  timeline_backtest: TimelineBacktestSummary | null;
  investingpro_fair_value?: number;
  valuation_gap_pct?: number;
  analyst_target_price?: number;
  warren_ai_momentum?: string;
  investingpro_models?: unknown[];
  cognitive_temperature_gap?: string;
  geopolitical_timing_advice?: string;
  value_trap_risk?: string;
  price_forecast?: PriceForecast | null;
  long_term_risk?: LongTermRisk | null;
};

export type HoldingReviewResult = {
  symbol: string;
  cost_basis: number | null;
  shares: number | null;
  latest_close: number;
  pnl_pct: number | null;
  market_value: number | null;
  unrealized_pnl: number | null;
  verdict: string;
  urgency: string;
  trim_ratio: string;
  holding_reason: string;
  protective_stop: string;
  signal: SimpleSignalResult;
};

export type SP500DailyPick = SimpleSignalResult & {
  company_name: string;
  sector: string;
  technical_score: number;
  news_score: number;
  fundamental_score: number;
  regime_score: number;
  backtest_score: number;
  daily_score: number;
  action_label: string;
  buy_urgency: string;
  position_sizing: string;
  headline_count: number;
  headline_summary: string;
  sector_score: number;
  is_main_line: boolean;
  is_sector_leader: boolean;
  sector_boost: number;
  is_dark_horse: boolean;
  dark_horse_boost: number;
  ai_chain_layer?: string | null;
  critical_bottleneck?: string | null;
  novice_rating?: string | null;
};

export type MarketRegime = {
  vix_close: number;
  vix_regime: string;
  fear_greed_score: number;
  fear_greed_label: string;
  fear_greed_source: number | string;
  spy_drawdown_pct: number;
  spy_distance_ma200_pct: number;
  regime_score: number;
  action: string;
  risk_budget: string;
  summary: string;
  backtest_win_rate_5d: number;
  backtest_avg_return_5d: number;
  backtest_win_rate_20d: number;
  backtest_avg_return_20d: number;
};

export type SectorAnalysis = {
  name: string;
  score: number;
  is_main_line: boolean;
  avg_return_5d: number;
  member_count: number;
  top_members: string[];
  market_role: string;
};

export type SP500DailyScanResponse = {
  market_regime: MarketRegime;
  picks: SP500DailyPick[];
  sectors: SectorAnalysis[];
  generated_at: string;
};

export type AiMainlineTrade = {
  symbol: string;
  layer: string | null;
  entry_date: string;
  exit_date: string;
  entry_price: number;
  exit_price: number;
  return_pct: number;
  days_held: number;
  outcome: string;
  pnl: number;
};

export type AiMainlineLayer = {
  layer: string;
  trades: number;
  win_rate: number;
  avg_return_pct: number;
  net_pnl: number;
  contribution_pct: number;
};

export type AiMainlineEquityPoint = {
  date: string;
  equity: number;
  return_pct: number;
};

export type AiMainlineBacktestResult = {
  market: string;
  start_date: string;
  end_date: string;
  years: number;
  initial_capital: number;
  final_equity: number;
  total_return_pct: number;
  cagr_pct: number;
  max_drawdown_pct: number;
  sharpe_ratio: number;
  win_rate: number;
  total_trades: number;
  avg_holding_days: number;
  avg_trade_return_pct: number;
  benchmark_symbol: string;
  benchmark_return_pct: number;
  excess_return_pct: number;
  universe: string[];
  layer_breakdown: AiMainlineLayer[];
  equity_curve: AiMainlineEquityPoint[];
  trades_log: AiMainlineTrade[];
  note: string;
};

export type AiMainlineBacktestPayload = {
  market?: 'us' | 'tw';
  period?: string;
  tickers?: string[];
  initialCapital?: number;
  maxPositions?: number;
  takeProfitPct?: number;
  trailingStopPct?: number;
  maxHoldingDays?: number;
};

async function parseResponse<T>(response: Response, fallbackMessage: string): Promise<T> {
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || fallbackMessage);
  }
  return data as T;
}

export async function analyzeSimpleSignal(payload: SimpleSignalPayload): Promise<SimpleSignalResult> {
  const response = await fetch(`${API_BASE_URL}/simple-signals/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ticker: payload.ticker,
      market: payload.market || undefined,
      period: payload.period || '3y',
      use_ai_committee: payload.useAiCommittee ?? false,
      committee_model: payload.committeeModel || 'gemma4:e4b',
    }),
  });
  return parseResponse<SimpleSignalResult>(response, '股票分析失敗。');
}

export async function analyzeSimpleSignalBatch(
  payload: Omit<SimpleSignalPayload, 'ticker'> & { tickers: string[] },
): Promise<SimpleSignalResult[]> {
  const response = await fetch(`${API_BASE_URL}/simple-signals/analyze-batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      tickers: payload.tickers,
      market: payload.market || undefined,
      period: payload.period || '3y',
      use_ai_committee: payload.useAiCommittee ?? false,
      committee_model: payload.committeeModel || 'gemma4:e4b',
    }),
  });
  return parseResponse<SimpleSignalResult[]>(response, '批次分析失敗。');
}

export async function reviewHoldings(payload: HoldingReviewPayload): Promise<HoldingReviewResult[]> {
  const response = await fetch(`${API_BASE_URL}/simple-signals/review-holdings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      holdings: payload.holdings,
      period: payload.period || '3y',
      use_ai_committee: payload.useAiCommittee ?? false,
      committee_model: payload.committeeModel || 'gemma4:e4b',
    }),
  });
  return parseResponse<HoldingReviewResult[]>(response, '持股檢視失敗。');
}

export async function fetchSp500DailyTop(payload: {
  period: string;
  limit?: number;
  useAiCommittee?: boolean;
  committeeModel?: string;
  market?: 'us' | 'tw';
  scanType?: 'optimal' | 'lagging_value';
}): Promise<SP500DailyScanResponse> {
  const response = await fetch(`${API_BASE_URL}/simple-signals/sp500-daily-top`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      period: payload.period || '3y',
      limit: payload.limit ?? 50,
      use_ai_committee: payload.useAiCommittee ?? false,
      committee_model: payload.committeeModel || 'gemma4:e4b',
      market: payload.market || 'us',
      scan_type: payload.scanType || 'optimal',
    }),
  });
  return parseResponse<SP500DailyScanResponse>(response, '每日排行掃描失敗。');
}

export async function runAiMainlineBacktest(
  payload: AiMainlineBacktestPayload = {},
): Promise<AiMainlineBacktestResult> {
  const response = await fetch(`${API_BASE_URL}/simple-signals/ai-mainline-backtest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      market: payload.market || 'us',
      period: payload.period || '5y',
      tickers: payload.tickers && payload.tickers.length ? payload.tickers : undefined,
      initial_capital: payload.initialCapital ?? 100000,
      max_positions: payload.maxPositions ?? 8,
      take_profit_pct: payload.takeProfitPct ?? 35,
      trailing_stop_pct: payload.trailingStopPct ?? 18,
      max_holding_days: payload.maxHoldingDays ?? 378,
    }),
  });
  return parseResponse<AiMainlineBacktestResult>(response, 'AI 主線長線回測失敗。');
}
