const API_BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

export type Ignition = {
  ignition_days_ago: number;
  ignition_gain_pct: number;
  ignition_volume_ratio: number;
  ignition_low: number;
};

export type Virattt = {
  score: number;
  signal: 'bullish' | 'neutral' | 'bearish' | string;
  confidence: number;
  trend: number | null;
  momentum: number | null;
  mean_reversion: number | null;
  volatility: number | null;
  stat_arb: number | null;
};

export type Spike = { date: string; gain_pct: number; limit_price: number | null };

export type LowView = {
  state: 'LOW_PRICE' | 'LOW_PRICE_OPPORTUNITY' | 'LOW_PRICE_SPECULATIVE';
  recommendation: 'BUY' | 'BUY_STAGED' | 'BUY_ON_PULLBACK' | 'BUY_ON_CONFIRMATION' | 'WATCH' | 'SPECULATIVE_WATCH' | 'NO_BUY';
  label: string;
  why: string;
  tier: 'A' | 'B' | 'C' | 'D';
  drawdown_type: string;
  confirmations: string[];
};

export type RankRow = {
  symbol: string;
  name: string;
  rank: number;
  zone: 'buy' | 'hold' | 'out';
  score: number;
  close: number;
  ret_6m_pct: number;
  ret_1m_pct: number | null;
  vol_ann_pct: number;
  above_ma200: boolean | null;
  day_change_pct: number;
  ignition?: Ignition;
  dd_52w_pct?: number | null;
  ret_3y_pct?: number | null;
  high_52w?: number;
  low_entry?: boolean;
  low_entry_watch?: boolean;
  low_entry_price?: number | null;
  virattt?: Virattt | null;
  tech_score?: number | null;
  spike?: Spike | null;
  bounce_20d_pct?: number | null;
  quality_tier?: 'A' | 'B' | 'C' | 'D';
  price_location?: string;
  low_view?: LowView | null;
};

export type MarketRanking = {
  universe_size: number;
  low_entry?: string[];
  low_entry_watch?: string[];
  top_n: number;
  keep_n: number;
  rows: RankRow[];
};

export type StrategyReport = {
  generated_at: string;
  strategy: {
    name: string;
    lookback_days: number;
    top_n: Record<string, number>;
    keep_n: Record<string, number>;
    allocation?: { lowentry: number; momentum: number };
    low_entry_rule?: string;
    rule: string;
    next_rebalance: string;
    in_rebalance_window: boolean;
    rebalance?: RebalanceInfo;
  };
  markets: Record<'us' | 'tw', MarketRanking>;
  disclaimer: string;
};

export type HoldingAction = '賣出換股' | '減碼' | '低檔加碼' | '加碼' | '排隊換股' | '續抱' | '核心 ETF' | '資料不足';

export type RebalanceInfo = Record<'us' | 'tw', { every: string; next: string; in_window: boolean; max_swaps: number }>;

export type EvaluatedHolding = {
  symbol: string;
  market: 'us' | 'tw';
  shares: number;
  cost: number;
  price: number | null;
  pnl_pct: number | null;
  value_twd: number;
  weight_pct: number;
  rank: number | null;
  score: number | null;
  ret_6m_pct: number | null;
  target_twd: number | null;
  ignition: Ignition | null;
  action: HoldingAction;
  reason: string;
  today?: string;
  today_reason?: string;
  day_change_pct?: number | null;
  day_pnl_twd?: number;
  day_pnl_local?: number;
  trim_twd?: number;
  trim_shares?: number | null;
  add_twd?: number | null;
  funding_note?: string | null;
  quote_as_of?: string | null;
  quote_source?: 'twse' | 'yfinance' | 'daily_close';
  market_closed?: boolean;
  spike?: Spike | null;
  virattt?: Virattt | null;
  dd_52w_pct?: number | null;
};

export type NewBuy = {
  symbol: string;
  name: string;
  rank: number;
  score: number;
  close: number;
  ret_6m_pct: number;
  ignition?: Ignition | null;
  target_twd: number | null;
};

export type EvaluateResult = {
  generated_at: string;
  next_rebalance: string;
  in_rebalance_window: boolean;
  rebalance?: RebalanceInfo;
  fx_usd_twd: number;
  total_twd: number;
  day_pnl_twd?: number;
  day_change_pct?: number | null;
  day_pnl_by_market_twd?: Record<'us' | 'tw', number>;
  live_quotes?: boolean;
  quote_as_of?: Record<'us' | 'tw', string | null>;
  market_closed?: Record<'us' | 'tw', boolean>;
  cash_twd?: number;
  total_with_cash_twd?: number;
  allocation_plan?: AllocationPlan;
  sleeve_twd: Record<string, number>;
  target_per_name_twd: Record<string, number>;
  holdings: EvaluatedHolding[];
  new_buys: Record<'us' | 'tw', NewBuy[]>;
  low_entry_buys?: Record<'us' | 'tw', { symbol: string; name: string; rank: number; close: number; dd_52w_pct: number; ret_3y_pct: number; high_52w: number; target_twd: number | null; low_label?: string; quality_tier?: string }[]>;
  low_entry_target_twd?: Record<string, number>;
  allocation?: { lowentry: number; momentum: number };
};

export type AllocationStep = {
  kind: 'sell' | 'trim' | 'fx' | 'buy' | 'bond';
  symbol: string;
  name?: string | null;
  market: 'us' | 'tw';
  amount_twd: number;
  shares?: number | null;
  shares_note?: string | null;
  when: string;
  why: string;
};

export type AllocationPlan = {
  error?: string;
  total_twd: number;
  current: Record<'stock' | 'bond' | 'cash' | 'other', { twd: number; pct: number }>;
  target: Record<'stock' | 'bond' | 'cash', { twd: number; pct: number }>;
  regime: { label: string; reasons: string[]; fear_greed: number | null; vix: number | null };
  steps: AllocationStep[];
  notes: string[];
  disclaimer: string;
};

export type PeriodStats = { total_return_pct: number; cagr_pct: number; max_drawdown_pct: number; sharpe: number };

export type StrategyBacktest = {
  market: 'us' | 'tw';
  benchmark_symbol: string;
  start_date: string;
  end_date: string;
  universe_size: number;
  trades_per_year: number;
  periods: Record<string, { strategy: PeriodStats; benchmark: PeriodStats; compare?: Record<string, PeriodStats> }>;
  equity_curve: { date: string; strategy: number; benchmark: number }[];
  current_holdings: string[];
  recent_rebalances: { date: string; buy: string[]; sell: string[] }[];
  rules: { lookback_days: number; top_n: number; keep_n: number; rebalance_days: number; cost_per_trade: number };
  caveat: string;
  tracks?: Record<string, { name: string; periods: Record<string, { strategy: PeriodStats; benchmark: PeriodStats }>; win_rate_pct?: number; trades?: number;
    recent_trades?: { ticker: string; entry: string; exit: string; return_pct: number }[] }>;
  combo_curve?: { date: string; combo: number; lowentry: number; benchmark: number }[];
};

async function parse<T>(response: Response, fallback: string): Promise<T> {
  if (!response.ok) {
    let detail = fallback;
    try {
      const body = await response.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export async function fetchStrategyReport(): Promise<StrategyReport> {
  const response = await fetch(`${API_BASE_URL}/strategy/report`);
  return parse<StrategyReport>(response, '策略排名讀取失敗。');
}

export async function evaluateHoldings(
  holdings: { ticker: string; cost: number; shares: number }[],
  cash: { twd: number; usd: number } = { twd: 0, usd: 0 },
): Promise<EvaluateResult> {
  const response = await fetch(`${API_BASE_URL}/strategy/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ holdings, cash_twd: cash.twd || 0, cash_usd: cash.usd || 0 }),
  });
  return parse<EvaluateResult>(response, '持股評估失敗。');
}

export async function fetchStrategyBacktest(): Promise<{ generated_at: string; allocation?: { lowentry: number; momentum: number }; markets: Record<'us' | 'tw', StrategyBacktest> }> {
  const response = await fetch(`${API_BASE_URL}/daily/data/strategy_backtest.json?_=${Date.now()}`);
  return parse(response, '回測結果讀取失敗。');
}

export type LookupResult = {
  symbol: string;
  market: 'us' | 'tw';
  universe_size: number | null;
  keep_n: number;
  top_n: number;
  row: (RankRow & { outside_universe?: boolean }) | null;
  verdict: string;
  detail: string;
  closes: { date: string; close: number }[];
};

export async function lookupSymbols(symbols: string[]): Promise<LookupResult[]> {
  const response = await fetch(`${API_BASE_URL}/strategy/lookup?symbols=${encodeURIComponent(symbols.join(','))}`);
  return parse<LookupResult[]>(response, '查詢失敗。');
}

export type OppRow = {
  ticker: string;
  company?: string;
  price: number | null;
  held: boolean;
  theme: string | null;
  sub_theme: string | null;
  valuation_class: string | null;
  opportunity: number | null;
  opportunity_detail?: { coverage: number; validated_share?: number; components_used?: string[] };
  batch_rank?: number;
  status: string;
  flags: string[];
  status_why: string;
  scores: Record<string, number | null>;
  ai: { exposure: number | null; demand_sensitivity: number | null; cycle_position: string | null };
  entry: {
    zone_1?: { low: number; high: number; basis: string; pct_from_price: number };
    zone_2?: { low: number; high: number; basis: string; pct_from_price: number };
    zone_3?: { low: number; high: number; basis: string; pct_from_price: number };
    avoid_above?: number;
    notes?: string[];
  };
  narrative: Record<string, string>;
  rotation_view?: { better_than: string[]; worse_than: string[] };
  shock?: { shock: { date: string; return: number; causes: string[] }; fundamental_damage_score: number; verdict: string | null; evidence: string[] } | null;
  catalyst?: { score: number | null; key_catalyst?: string; events?: { title: string; category: string; direction: number; priced_in: number | null; published: string }[] } | null;
  target_weight?: number;
  sec_valuation?: { ps: number; ps_3y_median: number; relative_to_own_3y: number | null; pe: number | null } | null;
};

export type OpportunityReport = {
  generated_at: string;
  mode: string;
  regime: { regime: string; metrics: Record<string, number | null> };
  ranking: OppRow[];
  rotations: { sell: string; buy: string; rotation_score: number; type: string; fraction_of_A: number; why: string }[];
  target_weights: Record<string, number>;
  notes: string[];
};

export async function fetchOpportunity(): Promise<OpportunityReport> {
  const response = await fetch(`${API_BASE_URL}/strategy/opportunity`);
  return parse<OpportunityReport>(response, '機會評分讀取失敗。');
}

export async function opportunityLive(symbols: string[], holdings: { ticker: string; cost: number; shares: number }[]): Promise<OpportunityReport> {
  const response = await fetch(`${API_BASE_URL}/strategy/opportunity/live`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbols, holdings }),
  });
  return parse<OpportunityReport>(response, '即時評分失敗。');
}

export async function fetchRankingValidation(market: 'us' | 'tw'): Promise<Record<string, unknown>> {
  const response = await fetch(`${API_BASE_URL}/daily/data/ranking_validation_${market}.json?_=${Date.now()}`);
  return parse(response, '驗證結果讀取失敗。');
}

/* ============================== 每日建議（市場情緒） ============================== */

export type SectorMood = {
  key: string;
  name: string;
  kind: 'etf' | 'group';
  ret_1m_pct: number;
  ret_3m_pct: number;
  ret_6m_pct: number;
  rel_spy_3m_pct: number | null;
  rsi14: number;
  dist_ma50_pct: number;
  breadth_above_ma50_pct: number | null;
  mood: '過熱' | '強勢' | '中性' | '弱勢' | '超賣';
  members?: number;
};

export type MarketSentiment = {
  asof: string | null;
  fear_greed: {
    score: number | null;
    label: string;
    prev_1d: number | null;
    prev_1w: number | null;
    prev_1m: number | null;
    components: { key: string; name: string; score: number }[];
    history: { date: string; fg: number }[];
    note: string;
  };
  cnn: { score: number; rating: string; timestamp: string; previous_close: number | null; previous_1_week: number | null; previous_1_month: number | null } | null;
  cnn_validation: { n_days: number; corr: number; mean_abs_diff: number } | null;
  vix: { value: number | null; prev_1d: number | null; ma50: number | null; vix3m: number | null; backwardation: boolean; panic: boolean };
  taiex: { close: number; vs_ma200_pct: number; rsi14: number; ret_1m_pct: number; dd_52w_pct: number } | null;
  sectors: SectorMood[];
};

export type AdviceItem = {
  type: 'low_entry_new' | 'low_entry_limit' | 'low_entry_holding' | 'near_trigger' | 'rebalance_buy' | 'rebalance_wait' | 'panic_hold' | 'ignition';
  spike?: Spike | null;
  bounce_20d_pct?: number | null;
  virattt?: Virattt | null;
  symbol: string;
  name?: string;
  close?: number;
  rank?: number;
  dd_52w_pct?: number | null;
  ret_3y_pct?: number | null;
  high_52w?: number;
  low_entry_price?: number | null;
  action: string;
  reason: string;
};

export type MarketAdvice = {
  level: 'action' | 'watch' | 'hold';
  summary: string;
  actions: AdviceItem[];
  watch: AdviceItem[];
  hot_groups: string[];
  oversold_groups: string[];
  momentum_share?: number;
  virattt?: {
    bullish: { symbol: string; name?: string; close: number; rank: number; day_change_pct?: number; virattt: Virattt }[];
    bearish: { symbol: string; name?: string; close: number; rank: number; day_change_pct?: number; virattt: Virattt }[];
    counts: Record<string, number>;
    used_in_ranking: boolean;
  };
};


export type DailyAdvice = {
  generated_at: string;
  date: string;
  headline: string;
  panic_no_sell: boolean;
  in_rebalance_window: boolean;
  next_rebalance: string;
  rebalance?: RebalanceInfo;
  sentiment_guidance: string[];
  markets: Partial<Record<'us' | 'tw', MarketAdvice>>;
  rules: { rule: string; check: string }[];
  disclaimer: string;
  sentiment: MarketSentiment | null;
  history: { date: string; headline: string; panic_no_sell: boolean; markets: Record<string, { level: string; summary: string; buys: string[] }> }[];
  evidence: Record<
    'us' | 'tw',
    {
      sentiment_study: { since: string; note: string; rows: { bucket: string; days: number; pool_21?: number; pool_63?: number; pool_126?: number; bench_63?: number }[] } | null;
      sentiment_ablation: ({ name: string } & Record<string, PeriodStats | string>)[] | null;
    }
  > | null;
};

export async function fetchDailyAdvice(): Promise<DailyAdvice> {
  const response = await fetch(`${API_BASE_URL}/strategy/advice`);
  return parse<DailyAdvice>(response, '每日建議讀取失敗');
}

/* ============================== AI 問答 ============================== */

export type AskResult = {
  prompt: string;
  answer?: string;
  model?: string;
  error?: string;
  cash_plan?: { order: number; symbol: string; why: string; value_twd: number; suggest_trim_twd: number | null }[];
};

export async function fetchAskPresets(): Promise<{ questions: string[]; llm_available: boolean }> {
  const response = await fetch(`${API_BASE_URL}/strategy/ask/presets`);
  return parse(response, '讀取預設問題失敗');
}

export async function askQuestion(
  question: string,
  holdings: { ticker: string; cost: number; shares: number }[],
  callLlm: boolean,
): Promise<AskResult> {
  const response = await fetch(`${API_BASE_URL}/strategy/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, holdings, call_llm: callLlm }),
  });
  return parse<AskResult>(response, '問答失敗');
}
