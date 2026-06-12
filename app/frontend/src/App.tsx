import type { ErrorInfo, FormEvent, ReactNode } from 'react';
import { Component, useState } from 'react';
import {
  ArrowDownToLine,
  ArrowUpFromLine,
  BriefcaseBusiness,
  Clock3,
  LineChart,
  Search,
  ShieldAlert,
  Sparkles,
  TrendingUp,
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Toaster } from './components/ui/sonner';
import {
  analyzeSimpleSignal,
  analyzeSimpleSignalBatch,
  fetchSp500DailyTop,
  reviewHoldings,
  runAiMainlineBacktest,
  type AgentView,
  type AiMainlineBacktestResult,
  type ChartPoint,
  type HoldingReviewItemPayload,
  type HoldingReviewResult,
  type HorizonView,
  type LongTermRisk,
  type MarketRegime,
  type PriceForecast,
  type SP500DailyPick,
  type SP500DailyScanResponse,
  type SimpleSignalResult,
} from './services/simple-signal-api';

type DetailResult = SimpleSignalResult | SP500DailyPick;

class ErrorBoundary extends Component<{ children: ReactNode }, { hasError: boolean; message: string }> {
  constructor(props: { children: ReactNode }) {
    super(props);
    this.state = { hasError: false, message: '' };
  }

  static getDerivedStateFromError(error: unknown): { hasError: boolean; message: string } {
    return { hasError: true, message: error instanceof Error ? error.message : String(error) };
  }

  componentDidCatch(error: unknown, info: ErrorInfo): void {
    console.error('結果渲染發生錯誤：', error, info);
  }

  render(): ReactNode {
    if (this.state.hasError) {
      return (
        <div className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">
          <div className="font-semibold">這檔結果顯示時發生問題，已略過以保持頁面正常。</div>
          <div className="mt-1 text-xs text-rose-500">請改選其他標的，或重新掃描。{this.state.message ? `（${this.state.message}）` : ''}</div>
        </div>
      );
    }
    return this.props.children;
  }
}

function safeFixed(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value) || !Number.isFinite(value)) {
    return '-';
  }
  return value.toFixed(digits);
}

function parseTickers(raw: string): string[] {
  return raw
    .split(/[\s,\n，、]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseHoldingsInput(raw: string, market: 'us' | 'tw' | ''): HoldingReviewItemPayload[] {
  return raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const parts = line.split(/[\s,，、]+/).filter(Boolean);
      const [ticker, costBasisText, sharesText] = parts;
      return {
        ticker,
        market: market || undefined,
        cost_basis: costBasisText ? Number(costBasisText) : undefined,
        shares: sharesText ? Number(sharesText) : undefined,
      };
    })
    .filter((item) => item.ticker);
}

function actionTone(action: string): string {
  if (action.includes('買') || action.includes('偏多')) return 'text-emerald-700';
  if (action.includes('賣') || action.includes('偏空')) return 'text-rose-700';
  if (action.includes('續抱')) return 'text-sky-700';
  if (action.includes('回檔') || action.includes('觀察')) return 'text-amber-700';
  return 'text-slate-700';
}

function aiScoreText(result: DetailResult): string {
  if (result.ai_score !== null) return `${result.ai_score}`;
  if (result.ai_enabled && !result.ai_available) return 'AI 不可用';
  if (result.ai_enabled) return 'AI 未回傳';
  return '未啟用';
}

function finalVerdict(result: DetailResult): string {
  if (result.today_exit_action.includes('賣')) return '偏賣出，先處理風險或獲利';
  if (result.today_action.includes('可買')) return '偏買進，今天可評估掛單';
  if (result.today_action.includes('回檔')) return '等回檔，今天不追高';
  return '先觀察，等待更明確位置';
}

type TabKey = 'analyze' | 'daily' | 'holdings' | 'backtest';

const TABS: { key: TabKey; label: string; icon: typeof Search; hint: string }[] = [
  { key: 'analyze', label: '個股分析', icon: Search, hint: '單股或多股技術＋AI 評分' },
  { key: 'daily', label: '每日掃描', icon: Sparkles, hint: '當日最佳買點 Top 50' },
  { key: 'holdings', label: '持股健檢', icon: BriefcaseBusiness, hint: '判斷續抱 / 減碼 / 賣出' },
  { key: 'backtest', label: 'AI 主線長線回測', icon: LineChart, hint: '產業鏈投組投報率驗證' },
];

export default function App() {
  const [ticker, setTicker] = useState('');
  const [holdingsText, setHoldingsText] = useState('');
  const [market, setMarket] = useState<'us' | 'tw' | ''>('');
  const [scanMarket, setScanMarket] = useState<'us' | 'tw'>('us');
  const [scanType, setScanType] = useState<'optimal' | 'lagging_value'>('optimal');
  const [useAiCommittee, setUseAiCommittee] = useState(false);
  const [committeeModel, setCommitteeModel] = useState('gemma4:e4b');

  const [backtestMarket, setBacktestMarket] = useState<'us' | 'tw'>('us');
  const [backtestPeriod, setBacktestPeriod] = useState('5y');

  const [activeTab, setActiveTab] = useState<TabKey>('analyze');

  const [loading, setLoading] = useState(false);
  const [holdingsLoading, setHoldingsLoading] = useState(false);
  const [scanLoading, setScanLoading] = useState(false);
  const [backtestLoading, setBacktestLoading] = useState(false);
  const [error, setError] = useState('');
  const [holdingsError, setHoldingsError] = useState('');
  const [dailyError, setDailyError] = useState('');
  const [backtestError, setBacktestError] = useState('');

  const [result, setResult] = useState<DetailResult | null>(null);
  const [ranking, setRanking] = useState<SimpleSignalResult[]>([]);
  const [holdings, setHoldings] = useState<HoldingReviewResult[]>([]);
  const [selectedHolding, setSelectedHolding] = useState<HoldingReviewResult | null>(null);
  const [dailyScan, setDailyScan] = useState<SP500DailyScanResponse | null>(null);
  const [backtest, setBacktest] = useState<AiMainlineBacktestResult | null>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const tickers = parseTickers(ticker);
    if (tickers.length === 0) {
      setError('請先輸入股票代碼。');
      return;
    }

    setLoading(true);
    setError('');
    setSelectedHolding(null);
    try {
      if (tickers.length > 1) {
        const response = await analyzeSimpleSignalBatch({ tickers, market, useAiCommittee, committeeModel });
        setRanking(response);
        setResult(response[0] ?? null);
      } else {
        const response = await analyzeSimpleSignal({ ticker: tickers[0], market, useAiCommittee, committeeModel });
        setRanking([response]);
        setResult(response);
      }
    } catch (submitError) {
      setRanking([]);
      setResult(null);
      setError(submitError instanceof Error ? submitError.message : '分析失敗。');
    } finally {
      setLoading(false);
    }
  }

  async function handleReviewHoldings() {
    const parsed = parseHoldingsInput(holdingsText, market);
    if (parsed.length === 0) {
      setHoldingsError('請至少輸入一筆持股，例如 AAPL 185 20。');
      return;
    }

    setHoldingsLoading(true);
    setHoldingsError('');
    try {
      const response = await reviewHoldings({ holdings: parsed, useAiCommittee, committeeModel });
      setHoldings(response);
      if (response[0]) {
        setSelectedHolding(response[0]);
        setResult(response[0].signal);
      }
    } catch (reviewError) {
      setHoldings([]);
      setSelectedHolding(null);
      setHoldingsError(reviewError instanceof Error ? reviewError.message : '持股健檢失敗。');
    } finally {
      setHoldingsLoading(false);
    }
  }

  async function handleDailyScan() {
    setScanLoading(true);
    setDailyError('');
    try {
      const response = await fetchSp500DailyTop({
        period: '3y',
        limit: 50,
        useAiCommittee,
        committeeModel,
        market: scanMarket,
        scanType,
      });
      setDailyScan(response);
      if (response.picks[0]) {
        setSelectedHolding(null);
        setResult(response.picks[0]);
      }
    } catch (scanError) {
      setDailyScan(null);
      setDailyError(scanError instanceof Error ? scanError.message : '每日掃描失敗。');
    } finally {
      setScanLoading(false);
    }
  }

  async function handleAiMainlineBacktest() {
    setBacktestLoading(true);
    setBacktestError('');
    try {
      const response = await runAiMainlineBacktest({
        market: backtestMarket,
        period: backtestPeriod,
      });
      setBacktest(response);
    } catch (btError) {
      setBacktest(null);
      setBacktestError(btError instanceof Error ? btError.message : 'AI 主線長線回測失敗。');
    } finally {
      setBacktestLoading(false);
    }
  }

  return (
    <>
      <main className="min-h-screen bg-[#f5f7fb] text-slate-900">
        <header className="border-b border-slate-200 bg-white">
          <div className="mx-auto flex w-full max-w-7xl flex-col gap-3 px-4 py-3 md:flex-row md:items-center md:justify-between md:px-6">
            <div className="flex items-center gap-2">
              <span className="flex h-8 w-8 items-center justify-center rounded-md bg-slate-950 text-white">
                <TrendingUp className="h-4 w-4" />
              </span>
              <div>
                <div className="text-sm font-semibold leading-tight text-slate-900">AI 主線投資儀表板</div>
                <div className="text-xs text-slate-500">技術 + AI 評分 ・ 每日掃描 ・ 持股健檢 ・ 長線回測</div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <label className="flex h-9 items-center gap-2 rounded-md border border-slate-200 bg-white px-3 text-sm">
                <input checked={useAiCommittee} onChange={(event) => setUseAiCommittee(event.target.checked)} type="checkbox" />
                AI 加權
              </label>
              <Input
                value={committeeModel}
                onChange={(event) => setCommitteeModel(event.target.value)}
                className="h-9 w-40 bg-white"
                title="AI 委員會模型名稱"
              />
            </div>
          </div>
        </header>

        <section className="mx-auto w-full max-w-7xl px-4 py-5 md:px-6">
          <nav className="mb-5 flex flex-wrap gap-2">
            {TABS.map((tab) => {
              const Icon = tab.icon;
              const active = activeTab === tab.key;
              return (
                <button
                  key={tab.key}
                  type="button"
                  onClick={() => setActiveTab(tab.key)}
                  className={`flex items-center gap-2 rounded-md border px-3 py-2 text-sm transition ${
                    active
                      ? 'border-slate-900 bg-slate-950 text-white shadow-sm'
                      : 'border-slate-200 bg-white text-slate-600 hover:border-slate-400'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span className="font-medium">{tab.label}</span>
                  <span className={`hidden text-xs lg:inline ${active ? 'text-slate-300' : 'text-slate-400'}`}>{tab.hint}</span>
                </button>
              );
            })}
          </nav>

          {activeTab === 'analyze' ? (
            <div className="space-y-5">
              <form className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm" onSubmit={handleSubmit}>
                <div className="grid gap-3 lg:grid-cols-[1fr_120px_auto] lg:items-end">
                  <label className="block">
                    <span className="mb-1 block text-xs font-semibold text-slate-600">單股 / 多股查詢（可用逗號或空白分隔）</span>
                    <div className="relative">
                      <Search className="pointer-events-none absolute left-3 top-3 h-4 w-4 text-slate-400" />
                      <Input value={ticker} onChange={(event) => setTicker(event.target.value)} placeholder="AAPL, MSFT, NVDA, 2330" className="h-10 bg-white pl-9" />
                    </div>
                  </label>
                  <label className="block">
                    <span className="mb-1 block text-xs font-semibold text-slate-600">市場</span>
                    <select value={market} onChange={(event) => setMarket(event.target.value as 'us' | 'tw' | '')} className="h-10 w-full rounded-md border border-slate-200 bg-white px-3 text-sm">
                      <option value="">自動</option>
                      <option value="us">美股</option>
                      <option value="tw">台股</option>
                    </select>
                  </label>
                  <Button className="h-10 bg-slate-950 text-white hover:bg-slate-800 lg:w-32" disabled={loading} type="submit">
                    {loading ? '分析中' : '分析'}
                  </Button>
                </div>
                {error ? <p className="mt-2 text-sm text-rose-600">{error}</p> : null}
              </form>

              <ErrorBoundary key={`single-${result?.symbol ?? 'none'}`}>
                <SignalInsight
                  result={result}
                  lists={
                    ranking.length > 1 ? (
                      <SignalList
                        title="多檔分析排名"
                        items={ranking}
                        onSelect={(item) => {
                          setSelectedHolding(null);
                          setResult(item);
                        }}
                        selectedSymbol={result?.symbol}
                      />
                    ) : null
                  }
                />
              </ErrorBoundary>
            </div>
          ) : null}

          {activeTab === 'daily' ? (
            <div className="space-y-5">
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <div className="mb-3 flex items-center justify-between">
                  <div>
                    <div className="text-sm font-semibold text-slate-900">每日候選掃描</div>
                    <div className="text-xs text-slate-500">以 agent 投票、回測、風控排序，挑出當日最佳買點</div>
                  </div>
                  <Sparkles className="h-5 w-5 text-slate-400" />
                </div>
                <div className="grid gap-2 sm:grid-cols-[160px_160px_auto] sm:items-end">
                  <label className="block">
                    <span className="mb-1 block text-xs font-semibold text-slate-600">市場</span>
                    <select value={scanMarket} onChange={(event) => setScanMarket(event.target.value as 'us' | 'tw')} className="h-10 w-full rounded-md border border-slate-200 bg-white px-3 text-sm">
                      <option value="us">美股</option>
                      <option value="tw">台股</option>
                    </select>
                  </label>
                  <label className="block">
                    <span className="mb-1 block text-xs font-semibold text-slate-600">掃描模式</span>
                    <select value={scanType} onChange={(event) => setScanType(event.target.value as 'optimal' | 'lagging_value')} className="h-10 w-full rounded-md border border-slate-200 bg-white px-3 text-sm">
                      <option value="optimal">最佳買點</option>
                      <option value="lagging_value">低估補漲</option>
                    </select>
                  </label>
                  <Button className="h-10 bg-slate-950 text-white hover:bg-slate-800 sm:w-44" disabled={scanLoading} onClick={handleDailyScan} type="button">
                    {scanLoading ? '掃描中' : '掃描 Top 50'}
                  </Button>
                </div>
                {dailyError ? <p className="mt-2 text-sm text-rose-600">{dailyError}</p> : null}
              </div>

              {dailyScan?.market_regime ? <MarketRegimeCard regime={dailyScan.market_regime} /> : null}

              <ErrorBoundary key={`daily-${result?.symbol ?? 'none'}`}>
                <SignalInsight
                  result={result}
                  lists={
                    dailyScan?.picks?.length ? (
                      <SignalList
                        title={`${scanMarket === 'us' ? 'S&P 500' : '台股'} 今日 Top 50`}
                        items={dailyScan.picks}
                        onSelect={(item) => {
                          setSelectedHolding(null);
                          setResult(item);
                        }}
                        selectedSymbol={result?.symbol}
                      />
                    ) : null
                  }
                />
              </ErrorBoundary>
            </div>
          ) : null}

          {activeTab === 'holdings' ? (
            <div className="space-y-5">
              <Card className="rounded-lg border-slate-200 bg-white shadow-sm">
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2 text-base">
                    <BriefcaseBusiness className="h-4 w-4" />
                    持股健檢
                  </CardTitle>
                  <CardDescription>每行一檔：代碼 成本 股數（例：AAPL 185 20）。系統會判斷賣出、減碼或續抱。</CardDescription>
                </CardHeader>
                <CardContent className="grid gap-3 lg:grid-cols-[1fr_200px] lg:items-start">
                  <textarea
                    className="min-h-[120px] w-full rounded-md border border-slate-200 bg-white p-3 text-sm outline-none focus:border-slate-400"
                    placeholder={'AAPL 185 20\nNVDA 109 10\n2330 785 1'}
                    value={holdingsText}
                    onChange={(event) => setHoldingsText(event.target.value)}
                  />
                  <Button className="h-10 w-full bg-slate-950 text-white hover:bg-slate-800" disabled={holdingsLoading} onClick={handleReviewHoldings} type="button">
                    {holdingsLoading ? '檢查中' : '檢查持股'}
                  </Button>
                </CardContent>
                {holdingsError ? <p className="px-6 pb-4 text-sm text-rose-600">{holdingsError}</p> : null}
              </Card>

              {selectedHolding ? <HoldingSummaryCard holding={selectedHolding} /> : null}

              <ErrorBoundary key={`holdings-${result?.symbol ?? 'none'}`}>
                <SignalInsight
                  result={result}
                  lists={
                    holdings.length > 0 ? (
                      <ResultList
                        title="持股健檢結果"
                        items={holdings}
                        onSelectHolding={(item) => {
                          setSelectedHolding(item);
                          setResult(item.signal);
                        }}
                        selectedSymbol={result?.symbol}
                      />
                    ) : null
                  }
                />
              </ErrorBoundary>
            </div>
          ) : null}

          {activeTab === 'backtest' ? (
            <div className="space-y-5">
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <div className="mb-3 flex items-center justify-between">
                  <div>
                    <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                      <LineChart className="h-4 w-4 text-slate-500" />
                      AI 主線投組長線回測
                    </div>
                    <div className="text-xs text-slate-500">
                      以 AI 產業鏈主線股、6-18 個月波段策略，驗證「AI 主線長線投資」整體投報率與對標超額報酬。
                    </div>
                  </div>
                </div>
                <div className="grid gap-2 sm:grid-cols-[160px_160px_auto] sm:items-end">
                  <label className="block">
                    <span className="mb-1 block text-xs font-semibold text-slate-600">市場</span>
                    <select value={backtestMarket} onChange={(event) => setBacktestMarket(event.target.value as 'us' | 'tw')} className="h-10 w-full rounded-md border border-slate-200 bg-white px-3 text-sm">
                      <option value="us">美股 AI 主線</option>
                      <option value="tw">台股 AI 主線</option>
                    </select>
                  </label>
                  <label className="block">
                    <span className="mb-1 block text-xs font-semibold text-slate-600">回看期間</span>
                    <select value={backtestPeriod} onChange={(event) => setBacktestPeriod(event.target.value)} className="h-10 w-full rounded-md border border-slate-200 bg-white px-3 text-sm">
                      <option value="3y">3 年</option>
                      <option value="5y">5 年</option>
                      <option value="10y">10 年</option>
                    </select>
                  </label>
                  <Button className="h-10 bg-slate-950 text-white hover:bg-slate-800 sm:w-44" disabled={backtestLoading} onClick={handleAiMainlineBacktest} type="button">
                    {backtestLoading ? '回測中' : '執行長線回測'}
                  </Button>
                </div>
                {backtestError ? <p className="mt-2 text-sm text-rose-600">{backtestError}</p> : null}
              </div>

              {backtest ? (
                <AiMainlineBacktestPanel data={backtest} />
              ) : (
                <div className="flex min-h-[180px] items-center justify-center rounded-lg border border-dashed border-slate-300 bg-white text-sm text-slate-500">
                  {backtestLoading ? '回測進行中，需下載多檔歷史資料，請稍候…' : '選擇市場與期間後，按「執行長線回測」檢視整體投報率與分層貢獻。'}
                </div>
              )}
            </div>
          ) : null}
        </section>
      </main>
      <Toaster />
    </>
  );
}

function SignalInsight({ result, lists }: { result: DetailResult | null; lists?: ReactNode }) {
  if (!result) {
    return lists ? <div className="space-y-4">{lists}</div> : <EmptyState />;
  }
  return (
    <div className="space-y-5">
      <StockOverviewCard result={result} />
      {result.long_term_risk ? <LongTermRiskBanner risk={result.long_term_risk} /> : null}
      {result.price_forecast ? <PriceForecastCard forecast={result.price_forecast} latestClose={result.latest_close} /> : null}
      {result.horizons?.length ? <HorizonPlanCard horizons={result.horizons} /> : null}
      <div className="grid gap-4 lg:grid-cols-[1fr_360px]">
        <div className="space-y-4">{lists}</div>
        <div className="space-y-4">{result.backtest ? <BacktestCard result={result} /> : null}</div>
      </div>
      <AgentVoteCard result={result} />
      <PriceChartCard result={result} />
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex h-full min-h-[220px] items-center justify-center rounded-lg border border-dashed border-slate-300 bg-white text-sm text-slate-500">
      輸入股票或持股後，這裡會顯示今天的買、賣、續抱結論。
    </div>
  );
}

function HoldingSummaryCard({ holding }: { holding: HoldingReviewResult }) {
  return (
    <Card className="rounded-lg border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">{holding.symbol} 持股判斷</CardTitle>
        <CardDescription>{holding.holding_reason}</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-3 sm:grid-cols-3 xl:grid-cols-6">
        <MetricCard label="建議動作" value={holding.verdict} valueClassName={actionTone(holding.verdict)} />
        <MetricCard label="賣出比例" value={holding.trim_ratio} />
        <MetricCard label="未實現報酬" value={holding.pnl_pct === null ? '-' : `${safeFixed(holding.pnl_pct, 2)}%`} />
        <MetricCard label="最新收盤" value={safeFixed(holding.latest_close, 2)} />
        <MetricCard label="保護停損" value={holding.protective_stop} />
        <MetricCard label="緊急程度" value={holding.urgency} />
      </CardContent>
    </Card>
  );
}

function StockOverviewCard({ result }: { result: DetailResult }) {
  return (
    <Card className="rounded-lg border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">{result.symbol} 結論與核心資料</CardTitle>
        <CardDescription>{finalVerdict(result)}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">今日操作結論</div>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard label="買進動作" value={result.today_action} valueClassName={actionTone(result.today_action)} icon={<ArrowDownToLine className="h-4 w-4" />} />
            <MetricCard label="今天買點" value={result.today_entry_zone} />
            <MetricCard label="賣出動作" value={result.today_exit_action} valueClassName={actionTone(result.today_exit_action)} icon={<ArrowUpFromLine className="h-4 w-4" />} />
            <MetricCard label="今天賣點" value={result.today_exit_zone} />
            <MetricCard label="預期報酬" value={`${safeFixed(result.expected_return_pct, 1)}%`} icon={<TrendingUp className="h-4 w-4" />} />
            <MetricCard label="風險報酬比" value={safeFixed(result.risk_reward_ratio, 2)} icon={<ShieldAlert className="h-4 w-4" />} />
            <MetricCard label="預估持有" value={result.holding_days_estimate > 0 ? `${result.holding_days_estimate} 天` : '暫不建倉'} icon={<Clock3 className="h-4 w-4" />} />
            <MetricCard label="綜合分數" value={`${result.composite_score}`} />
          </div>
        </div>
        <div>
          <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">核心技術資料</div>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard label="趨勢" value={result.bias} valueClassName={actionTone(result.bias)} />
            <MetricCard label="最新收盤" value={safeFixed(result.latest_close, 2)} />
            <MetricCard label="規則分數" value={`${result.rule_score}`} />
            <MetricCard label="AI 分數" value={aiScoreText(result)} />
            <MetricCard label="建議買點區" value={result.buy_zone} />
            <MetricCard label="建議賣點區" value={result.sell_zone} />
            <MetricCard label="停損區" value={result.stop_loss} />
            <MetricCard label="RSI14" value={safeFixed(result.rsi14, 1)} />
          </div>
          <p className="mt-2 text-xs leading-5 text-slate-500">{result.reason}</p>
        </div>
      </CardContent>
    </Card>
  );
}

function forecastVerdictTone(verdict: string): string {
  if (verdict.includes('賣') || verdict.includes('減碼')) return 'text-rose-700';
  if (verdict.includes('暫不') || verdict.includes('觀望')) return 'text-amber-700';
  if (verdict.includes('買進') || verdict.includes('布局')) return 'text-emerald-700';
  return 'text-slate-700';
}

function forecastVerdictBg(verdict: string): string {
  if (verdict.includes('賣') || verdict.includes('減碼')) return 'border-rose-200 bg-rose-50';
  if (verdict.includes('暫不') || verdict.includes('觀望')) return 'border-amber-200 bg-amber-50';
  if (verdict.includes('買進') || verdict.includes('布局')) return 'border-emerald-200 bg-emerald-50';
  return 'border-slate-200 bg-slate-50';
}

function stanceTone(stance: string): string {
  if (stance.includes('多')) return 'text-emerald-700';
  if (stance.includes('空')) return 'text-rose-700';
  return 'text-slate-500';
}

function LongTermRiskBanner({ risk }: { risk: LongTermRisk }) {
  const blocked = risk.blocked;
  const tone = blocked
    ? 'border-rose-300 bg-rose-50 text-rose-800'
    : risk.severity === 'medium'
      ? 'border-amber-300 bg-amber-50 text-amber-800'
      : 'border-emerald-200 bg-emerald-50 text-emerald-800';
  const title = blocked ? '🔴 長線恐虧損：不建議買進' : risk.severity === 'medium' ? '🟡 長線需謹慎' : '🟢 長線虧損風險評估';
  return (
    <div className={`rounded-lg border p-4 ${tone}`}>
      <div className="text-sm font-semibold">{title}</div>
      <div className="mt-1 text-xs leading-5">{risk.note}</div>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px]">
        {risk.expected_return_12m_pct !== null ? (
          <span>12 個月預測：{risk.expected_return_12m_pct >= 0 ? '+' : ''}{risk.expected_return_12m_pct.toFixed(1)}%</span>
        ) : null}
        {risk.history_cumulative_return_pct !== null && risk.history_trades > 0 ? (
          <span>
            個股波段歷史：累積 {risk.history_cumulative_return_pct >= 0 ? '+' : ''}
            {risk.history_cumulative_return_pct.toFixed(1)}%／勝率 {risk.history_win_rate_pct?.toFixed(0)}%（{risk.history_trades} 筆）
          </span>
        ) : null}
      </div>
    </div>
  );
}

function PriceForecastCard({ forecast, latestClose }: { forecast: PriceForecast; latestClose: number }) {
  return (
    <Card className="rounded-lg border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <TrendingUp className="h-4 w-4" />
          3 / 6 / 9 / 12 個月股價預測
        </CardTitle>
        <CardDescription>{forecast.method}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className={`rounded-md border p-3 ${forecastVerdictBg(forecast.verdict)}`}>
          <div className={`text-sm font-semibold ${forecastVerdictTone(forecast.verdict)}`}>現在建議：{forecast.verdict}</div>
          <div className="mt-0.5 text-xs leading-5 text-slate-600">{forecast.verdict_reason}</div>
          <div className="mt-1 text-xs text-slate-500">
            現價 {latestClose.toFixed(2)} ・ 年化漂移 {forecast.annualized_drift_pct.toFixed(1)}% ・ 年化波動 {forecast.annualized_volatility_pct.toFixed(1)}%
          </div>
        </div>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {forecast.horizons.map((horizon) => {
            const up = horizon.expected_return_pct >= 0;
            return (
              <div key={horizon.days} className="rounded-md border border-slate-200 bg-white p-3">
                <div className="mb-1 flex items-center justify-between">
                  <span className="text-sm font-semibold text-slate-900">{horizon.label}</span>
                  <span className={`text-xs font-semibold ${stanceTone(horizon.stance)}`}>{horizon.stance}</span>
                </div>
                <div className="text-lg font-semibold text-slate-900">{horizon.base.toFixed(2)}</div>
                <div className={`text-xs font-medium ${up ? 'text-emerald-700' : 'text-rose-700'}`}>
                  預期 {up ? '+' : ''}{horizon.expected_return_pct.toFixed(1)}%
                </div>
                <div className="mt-1 text-xs text-slate-500">區間 {horizon.low.toFixed(2)} ~ {horizon.high.toFixed(2)}</div>
              </div>
            );
          })}
        </div>
        <p className="text-xs leading-5 text-slate-400">
          預測為統計推估，僅供評估買賣時機參考，非保證未來表現；實際走勢受財報、消息面與大盤影響。
        </p>
      </CardContent>
    </Card>
  );
}

function HorizonPlanCard({ horizons }: { horizons: HorizonView[] }) {
  return (
    <Card className="rounded-lg border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <Clock3 className="h-4 w-4" />
          短 / 中 / 長線買賣點
        </CardTitle>
        <CardDescription>各時間維度的進場區、停利區與停損區，搭配上方股價預測判斷買進時機。</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-3 md:grid-cols-3">
        {horizons.map((horizon) => (
          <div key={horizon.horizon} className="rounded-md border border-slate-200 bg-white p-3">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-sm font-semibold text-slate-900">{horizon.horizon}</span>
              <span className={`text-xs font-semibold ${actionTone(horizon.bias)}`}>{horizon.bias}</span>
            </div>
            <div className="grid gap-1.5 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-slate-500">進場區</span>
                <span className="font-medium text-emerald-700">{horizon.entry_zone}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-500">停利區</span>
                <span className="font-medium text-slate-900">{horizon.take_profit_zone}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-500">停損區</span>
                <span className="font-medium text-rose-700">{horizon.stop_zone}</span>
              </div>
            </div>
            <p className="mt-2 text-xs leading-5 text-slate-500">{horizon.summary}</p>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

function AgentVoteCard({ result }: { result: DetailResult }) {
  return (
    <Card className="mb-5 rounded-lg border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Agents 投票評分</CardTitle>
        <CardDescription>參考原 repo 的 agent 架構，使用 Technical、Fundamentals、Valuation、Sentiment、Risk 的 signal 與 confidence 組合總分。</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
        {result.agents.map((agent) => (
          <AgentVoteItem key={agent.key} agent={agent} />
        ))}
      </CardContent>
    </Card>
  );
}

function AgentVoteItem({ agent }: { agent: AgentView }) {
  const edge = agent.historical_edge;
  const hasEdge = edge && edge.sample_size >= 8;
  const trustLabel = hasEdge
    ? edge.weight >= 1.2
      ? '高信任'
      : edge.weight <= 0.8
        ? '低信任'
        : '中性信任'
    : null;
  const trustTone = hasEdge
    ? edge.weight >= 1.2
      ? 'bg-emerald-100 text-emerald-700'
      : edge.weight <= 0.8
        ? 'bg-rose-100 text-rose-700'
        : 'bg-slate-200 text-slate-600'
    : '';
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <div className="text-sm font-semibold text-slate-900">{agent.name}</div>
          {trustLabel ? (
            <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${trustTone}`}>
              {trustLabel} ×{edge!.weight.toFixed(2)}
            </span>
          ) : null}
        </div>
        <div className={`text-sm font-semibold ${actionTone(agent.signal)}`}>{agent.signal}</div>
      </div>
      <div className="mb-2 h-1.5 overflow-hidden rounded-full bg-slate-200">
        <div className="h-full rounded-full bg-slate-900" style={{ width: `${Math.max(5, Math.min(100, agent.confidence))}%` }} />
      </div>
      <div className="text-xs leading-5 text-slate-600">{agent.summary}</div>
      {hasEdge ? (
        <div className="mt-2 text-[11px] leading-4 text-slate-500">
          個股歷史回測：此訊號出現後 20 日勝率 {edge!.win_rate.toFixed(0)}%、平均報酬 {edge!.avg_return.toFixed(1)}%（樣本 {edge!.sample_size} 次），據此調整信任權重。
        </div>
      ) : null}
    </div>
  );
}

function SignalList({
  title,
  items,
  selectedSymbol,
  onSelect,
}: {
  title: string;
  items: DetailResult[];
  selectedSymbol?: string;
  onSelect: (item: DetailResult) => void;
}) {
  return (
    <Card className="rounded-lg border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent className="grid gap-2">
        {items.map((item, index) => (
          <button
            key={`${item.symbol}-${index}`}
            className={`grid w-full gap-2 rounded-md border p-3 text-left text-sm hover:border-slate-400 md:grid-cols-[44px_1fr_120px_120px_90px] md:items-center ${
              selectedSymbol === item.symbol ? 'border-slate-900 bg-slate-50' : 'border-slate-200 bg-white'
            }`}
            onClick={() => onSelect(item)}
            type="button"
          >
            <div className="font-mono text-xs text-slate-500">#{index + 1}</div>
            <div>
              <div className="font-semibold text-slate-900">{item.symbol}</div>
              <div className="text-xs text-slate-500">{'company_name' in item ? item.company_name : finalVerdict(item)}</div>
            </div>
            <div className={`font-medium ${actionTone(item.today_action)}`}>{item.today_action}</div>
            <div className={`font-medium ${actionTone(item.today_exit_action)}`}>{item.today_exit_action}</div>
            <div className="text-slate-700">{'daily_score' in item ? item.daily_score : item.composite_score}</div>
          </button>
        ))}
      </CardContent>
    </Card>
  );
}

function ResultList({
  title,
  items,
  selectedSymbol,
  onSelectHolding,
}: {
  title: string;
  items: HoldingReviewResult[];
  selectedSymbol?: string;
  onSelectHolding: (item: HoldingReviewResult) => void;
}) {
  return (
    <Card className="rounded-lg border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent className="grid gap-2">
        {items.map((item) => (
          <button
            key={`${item.symbol}-${item.cost_basis ?? 'na'}`}
            className={`grid w-full gap-2 rounded-md border p-3 text-left text-sm hover:border-slate-400 md:grid-cols-[1fr_110px_90px_90px] md:items-center ${
              selectedSymbol === item.symbol ? 'border-slate-900 bg-slate-50' : 'border-slate-200 bg-white'
            }`}
            onClick={() => onSelectHolding(item)}
            type="button"
          >
            <div>
              <div className="font-semibold text-slate-900">{item.symbol}</div>
              <div className="text-xs text-slate-500">{item.holding_reason}</div>
            </div>
            <div className={`font-medium ${actionTone(item.verdict)}`}>{item.verdict}</div>
            <div>{item.trim_ratio}</div>
            <div>{item.pnl_pct === null ? '-' : `${item.pnl_pct.toFixed(1)}%`}</div>
          </button>
        ))}
      </CardContent>
    </Card>
  );
}

function BacktestCard({ result }: { result: DetailResult }) {
  if (!result.backtest) return null;

  return (
    <Card className="rounded-lg border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">回測校正</CardTitle>
        <CardDescription>{result.backtest.calibration_note}</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-3">
        <MetricCard label="樣本數" value={`${result.backtest.sample_size}`} />
        <MetricCard label="20日勝率" value={`${safeFixed(result.backtest.win_rate_20d, 1)}%`} />
        <MetricCard label="20日平均報酬" value={`${safeFixed(result.backtest.avg_return_20d, 2)}%`} />
        <MetricCard label="20日下跌比例" value={`${safeFixed(result.backtest.downside_rate_20d, 1)}%`} />
        <MetricCard label="回測可信度" value={`${result.backtest.confidence_score}`} />
      </CardContent>
    </Card>
  );
}

function MarketRegimeCard({ regime }: { regime: MarketRegime }) {
  return (
    <Card className="mb-5 rounded-lg border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">今日市場大環境</CardTitle>
        <CardDescription>{regime.summary}</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        <MetricCard label="市場動作" value={regime.action} />
        <MetricCard label="VIX" value={`${safeFixed(regime.vix_close, 2)} / ${regime.vix_regime}`} />
        <MetricCard label="Fear & Greed" value={`${regime.fear_greed_score} / ${regime.fear_greed_label}`} />
        <MetricCard label="SPY 回撤" value={`${safeFixed(regime.spy_drawdown_pct, 2)}%`} />
        <MetricCard label="建議部位" value={regime.risk_budget} />
      </CardContent>
    </Card>
  );
}

function PriceChartCard({ result }: { result: DetailResult }) {
  return (
    <Card className="mb-5 rounded-lg border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">{result.symbol} 走勢參考</CardTitle>
        <CardDescription>最近 60 個交易日，含收盤、MA20、MA50、買點區與賣點區。</CardDescription>
      </CardHeader>
      <CardContent>
        <PriceChart chart={result.chart} buyZone={result.buy_zone} sellZone={result.sell_zone} />
      </CardContent>
    </Card>
  );
}

function PriceChart({ chart, buyZone, sellZone }: { chart: ChartPoint[]; buyZone: string; sellZone: string }) {
  if (!chart || chart.length < 2) {
    return <div className="text-sm text-slate-500">目前沒有足夠圖表資料。</div>;
  }

  const width = 860;
  const height = 260;
  const padding = 18;
  const [buyLow, buyHigh] = parseRange(buyZone);
  const [sellLow, sellHigh] = parseRange(sellZone);
  const values = chart.flatMap((point) => [point.close, point.ma20 ?? point.close, point.ma50 ?? point.close]).concat([buyLow, buyHigh, sellLow, sellHigh]);
  const minValue = Math.min(...values) * 0.96;
  const maxValue = Math.max(...values) * 1.04;
  const toX = (index: number) => padding + (index / Math.max(chart.length - 1, 1)) * (width - padding * 2);
  const toY = (value: number) => height - padding - ((value - minValue) / Math.max(maxValue - minValue, 1)) * (height - padding * 2);

  const closePath = buildPath(chart.map((point, index) => ({ x: toX(index), y: toY(point.close) })));
  const ma20Points = chart.map((point, index) => (point.ma20 === null ? null : { x: toX(index), y: toY(point.ma20) })).filter(Boolean) as Array<{ x: number; y: number }>;
  const ma50Points = chart.map((point, index) => (point.ma50 === null ? null : { x: toX(index), y: toY(point.ma50) })).filter(Boolean) as Array<{ x: number; y: number }>;

  return (
    <div className="space-y-2">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full rounded-md border border-slate-200 bg-white">
        <rect x={padding} y={toY(sellHigh)} width={width - padding * 2} height={Math.max(toY(sellLow) - toY(sellHigh), 4)} fill="rgba(248,113,113,0.12)" />
        <rect x={padding} y={toY(buyHigh)} width={width - padding * 2} height={Math.max(toY(buyLow) - toY(buyHigh), 4)} fill="rgba(34,197,94,0.12)" />
        <path d={closePath} fill="none" stroke="#0f172a" strokeWidth="2.2" />
        <path d={buildPath(ma20Points)} fill="none" stroke="#0284c7" strokeWidth="1.6" strokeDasharray="5 4" />
        <path d={buildPath(ma50Points)} fill="none" stroke="#f97316" strokeWidth="1.6" strokeDasharray="4 4" />
      </svg>
      <div className="grid gap-2 text-xs text-slate-500 md:grid-cols-4">
        <div>黑線：收盤價</div>
        <div>藍線：MA20</div>
        <div>橘線：MA50</div>
        <div>綠區：買點 / 紅區：賣點</div>
      </div>
    </div>
  );
}

function buildPath(points: Array<{ x: number; y: number }>): string {
  if (points.length === 0) return '';
  return points.map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`).join(' ');
}

function parseRange(rangeText: string): [number, number] {
  const [low, high] = rangeText.split('-').map((part) => Number(part.trim()));
  return [low, high];
}

function AiMainlineBacktestPanel({ data }: { data: AiMainlineBacktestResult }) {
  const positiveTone = (value: number) => (value >= 0 ? 'text-emerald-700' : 'text-rose-700');
  const fmtPct = (value: number) => `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`;
  return (
    <Card className="mb-5 rounded-lg border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <LineChart className="h-4 w-4" />
          AI 主線長線回測結果
        </CardTitle>
        <CardDescription>{data.note}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <MetricCard label="累積報酬" value={fmtPct(data.total_return_pct)} valueClassName={positiveTone(data.total_return_pct)} icon={<TrendingUp className="h-4 w-4" />} />
          <MetricCard label="年化報酬 (CAGR)" value={fmtPct(data.cagr_pct)} valueClassName={positiveTone(data.cagr_pct)} />
          <MetricCard label="最大回撤" value={`${data.max_drawdown_pct.toFixed(1)}%`} valueClassName="text-rose-700" icon={<ShieldAlert className="h-4 w-4" />} />
          <MetricCard label="Sharpe" value={data.sharpe_ratio.toFixed(2)} />
          <MetricCard label="勝率" value={`${data.win_rate.toFixed(1)}%`} />
          <MetricCard label="交易次數" value={`${data.total_trades} 筆`} />
          <MetricCard label="平均持有" value={`${data.avg_holding_days.toFixed(0)} 天`} icon={<Clock3 className="h-4 w-4" />} />
          <MetricCard
            label={`對標 ${data.benchmark_symbol}`}
            value={`${fmtPct(data.benchmark_return_pct)} / 超額 ${fmtPct(data.excess_return_pct)}`}
            valueClassName={positiveTone(data.excess_return_pct)}
          />
        </div>

        <div>
          <div className="mb-1 flex items-center justify-between text-xs text-slate-500">
            <span>權益曲線（初始 {data.initial_capital.toLocaleString()} → 期末 {data.final_equity.toLocaleString()}）</span>
            <span>{data.start_date} ~ {data.end_date}（{data.years} 年）</span>
          </div>
          <EquityCurve points={data.equity_curve} />
        </div>

        {data.layer_breakdown.length ? (
          <div>
            <div className="mb-2 text-sm font-semibold text-slate-900">AI 產業鏈分層貢獻</div>
            <div className="grid gap-2">
              {data.layer_breakdown.map((layer) => (
                <div key={layer.layer} className="grid items-center gap-2 rounded-md border border-slate-200 bg-slate-50 p-3 text-sm md:grid-cols-[1fr_90px_90px_110px_90px]">
                  <div className="font-medium text-slate-900">{layer.layer}</div>
                  <div className="text-xs text-slate-500">{layer.trades} 筆</div>
                  <div className="text-xs text-slate-500">勝率 {layer.win_rate.toFixed(0)}%</div>
                  <div className={`text-xs ${positiveTone(layer.net_pnl)}`}>損益 {layer.net_pnl.toLocaleString()}</div>
                  <div className={`text-xs font-semibold ${positiveTone(layer.contribution_pct)}`}>貢獻 {layer.contribution_pct.toFixed(0)}%</div>
                </div>
              ))}
            </div>
          </div>
        ) : null}

        {data.trades_log.length ? (
          <div>
            <div className="mb-2 text-sm font-semibold text-slate-900">近期交易紀錄</div>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[640px] border-collapse text-left text-xs">
                <thead>
                  <tr className="text-slate-500">
                    <th className="py-1.5 pr-2 font-medium">標的</th>
                    <th className="py-1.5 pr-2 font-medium">分層</th>
                    <th className="py-1.5 pr-2 font-medium">進場</th>
                    <th className="py-1.5 pr-2 font-medium">出場</th>
                    <th className="py-1.5 pr-2 font-medium">報酬</th>
                    <th className="py-1.5 pr-2 font-medium">持有</th>
                    <th className="py-1.5 pr-2 font-medium">結果</th>
                  </tr>
                </thead>
                <tbody>
                  {[...data.trades_log].reverse().map((trade, index) => (
                    <tr key={`${trade.symbol}-${trade.exit_date}-${index}`} className="border-t border-slate-100">
                      <td className="py-1.5 pr-2 font-semibold text-slate-900">{trade.symbol}</td>
                      <td className="py-1.5 pr-2 text-slate-500">{trade.layer ?? '-'}</td>
                      <td className="py-1.5 pr-2 text-slate-600">{trade.entry_date}</td>
                      <td className="py-1.5 pr-2 text-slate-600">{trade.exit_date}</td>
                      <td className={`py-1.5 pr-2 font-medium ${positiveTone(trade.return_pct)}`}>{fmtPct(trade.return_pct)}</td>
                      <td className="py-1.5 pr-2 text-slate-600">{trade.days_held} 天</td>
                      <td className="py-1.5 pr-2 text-slate-600">{trade.outcome}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

function EquityCurve({ points }: { points: AiMainlineBacktestResult['equity_curve'] }) {
  if (!points || points.length < 2) {
    return <div className="text-sm text-slate-500">沒有足夠的權益曲線資料。</div>;
  }
  const width = 860;
  const height = 220;
  const padding = 18;
  const values = points.map((p) => p.equity);
  const minValue = Math.min(...values) * 0.98;
  const maxValue = Math.max(...values) * 1.02;
  const toX = (index: number) => padding + (index / Math.max(points.length - 1, 1)) * (width - padding * 2);
  const toY = (value: number) => height - padding - ((value - minValue) / Math.max(maxValue - minValue, 1)) * (height - padding * 2);
  const path = buildPath(points.map((point, index) => ({ x: toX(index), y: toY(point.equity) })));
  const baselineY = toY(points[0].equity);
  const lastUp = points[points.length - 1].equity >= points[0].equity;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full rounded-md border border-slate-200 bg-white">
      <line x1={padding} y1={baselineY} x2={width - padding} y2={baselineY} stroke="#cbd5e1" strokeWidth="1" strokeDasharray="4 4" />
      <path d={path} fill="none" stroke={lastUp ? '#059669' : '#e11d48'} strokeWidth="2.2" />
    </svg>
  );
}

function MetricCard({ label, value, icon, valueClassName }: { label: string; value: string; icon?: ReactNode; valueClassName?: string }) {
  return (
    <div className="rounded-md border border-slate-200 bg-white p-3">
      <div className="mb-1 flex items-center gap-2 text-xs text-slate-500">
        {icon}
        <span>{label}</span>
      </div>
      <div className={`break-words text-sm font-semibold text-slate-900 ${valueClassName ?? ''}`}>{value}</div>
    </div>
  );
}
