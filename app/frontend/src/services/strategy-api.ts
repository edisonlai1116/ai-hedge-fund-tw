const API_BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

export type Ignition = {
  ignition_days_ago: number;
  ignition_gain_pct: number;
  ignition_volume_ratio: number;
  ignition_low: number;
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
};

export type MarketRanking = {
  universe_size: number;
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
    rule: string;
    next_rebalance: string;
    in_rebalance_window: boolean;
  };
  markets: Record<'us' | 'tw', MarketRanking>;
  disclaimer: string;
};

export type HoldingAction = '賣出換股' | '減碼' | '加碼' | '續抱' | '核心 ETF' | '資料不足';

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
  fx_usd_twd: number;
  total_twd: number;
  sleeve_twd: Record<string, number>;
  target_per_name_twd: Record<string, number>;
  holdings: EvaluatedHolding[];
  new_buys: Record<'us' | 'tw', NewBuy[]>;
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

export async function evaluateHoldings(holdings: { ticker: string; cost: number; shares: number }[]): Promise<EvaluateResult> {
  const response = await fetch(`${API_BASE_URL}/strategy/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ holdings }),
  });
  return parse<EvaluateResult>(response, '持股評估失敗。');
}

export async function fetchStrategyBacktest(): Promise<{ generated_at: string; markets: Record<'us' | 'tw', StrategyBacktest> }> {
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
