import type { ChangeEvent, ErrorInfo, FormEvent, ReactNode } from 'react';
import { Component, Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  ArrowDownToLine,
  ArrowUpFromLine,
  BriefcaseBusiness,
  ChevronDown,
  Clock3,
  Gauge,
  LineChart,
  Mic,
  RadioTower,
  Search,
  ShieldAlert,
  Sparkles,
  TrendingUp,
  Wallet,
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Toaster } from './components/ui/sonner';
import {
  analyzeSimpleSignal,
  analyzeSimpleSignalBatch,
  fetchGooayeOpinions,
  fetchMarketRegime,
  fetchQuotes,
  fetchSp500DailyTop,
  fetchSystemStatus,
  reviewHoldings,
  runAiMainlineBacktest,
  type SystemStatus,
  type AgentView,
  type AiMainlineBacktestResult,
  type ChartPoint,
  type GooayeOpinion,
  type HoldingReviewItemPayload,
  type HoldingReviewResult,
  type HorizonView,
  type LongTermRisk,
  type MarketRegime,
  type MarketRegimeSuggestion,
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

// 統一比對鍵：去 .TW/.TWO 後綴並大寫（股癌點名對照、持股健檢比對共用）。
function tickerBaseKey(ticker: string): string {
  return (ticker || '').trim().toUpperCase().replace(/\.(TW|TWO)$/, '');
}
function buildGooayeMap(opinions: GooayeOpinion[]): Record<string, GooayeOpinion> {
  const map: Record<string, GooayeOpinion> = {};
  for (const op of opinions) {
    if (op?.target_ticker) map[tickerBaseKey(op.target_ticker)] = op;
  }
  return map;
}
function gooayeStanceTone(label: string): string {
  if (/bull|多|買/i.test(label)) return 'bg-emerald-100 text-emerald-700';
  if (/bear|空|賣|減/i.test(label)) return 'bg-rose-100 text-rose-700';
  return 'bg-slate-200 text-slate-600';
}

// 真實持股（持股健檢）持久化：記住輸入＋上次健檢日期，避免每次重 key、並可每天自動健檢。
const REAL_HOLDINGS_KEY = 'real_holdings_v1';
function loadRealHoldings(): { text: string; lastReviewed: string } {
  try {
    const raw = localStorage.getItem(REAL_HOLDINGS_KEY);
    if (raw) {
      const d = JSON.parse(raw);
      return { text: typeof d.text === 'string' ? d.text : '', lastReviewed: d.lastReviewed ?? '' };
    }
  } catch {
    /* ignore */
  }
  return { text: '', lastReviewed: '' };
}
function saveRealHoldings(text: string, lastReviewed: string): void {
  try {
    localStorage.setItem(REAL_HOLDINGS_KEY, JSON.stringify({ text, lastReviewed }));
  } catch {
    /* ignore */
  }
}
function localDateStr(): string {
  return new Date().toLocaleDateString('en-CA');
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

type TabKey = 'analyze' | 'daily' | 'holdings' | 'backtest' | 'portfolio';

const TABS: { key: TabKey; label: string; icon: typeof Search; hint: string }[] = [
  { key: 'analyze', label: '個股分析', icon: Search, hint: '單股或多股技術＋AI 評分' },
  { key: 'daily', label: '每日掃描', icon: Sparkles, hint: '當日最佳買點 Top 50' },
  { key: 'holdings', label: '持股健檢', icon: BriefcaseBusiness, hint: '判斷續抱 / 減碼 / 賣出' },
  { key: 'backtest', label: 'AI 主線長線回測', icon: LineChart, hint: '產業鏈投組投報率驗證' },
  { key: 'portfolio', label: '跟單對帳本', icon: Wallet, hint: '5 萬美金實單跟蹤 vs 大盤' },
];

export default function App() {
  const [ticker, setTicker] = useState('');
  const [holdingsText, setHoldingsText] = useState(() => loadRealHoldings().text);
  const [holdingsLastReviewed, setHoldingsLastReviewed] = useState(() => loadRealHoldings().lastReviewed);
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
  const [gooayeMap, setGooayeMap] = useState<Record<string, GooayeOpinion>>({});

  useEffect(() => {
    fetchGooayeOpinions()
      .then((ops) => setGooayeMap(buildGooayeMap(ops)))
      .catch(() => setGooayeMap({}));
  }, []);


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

  async function handleReviewHoldings(opts?: { auto?: boolean }) {
    const parsed = parseHoldingsInput(holdingsText, market);
    if (parsed.length === 0) {
      if (!opts?.auto) setHoldingsError('請至少輸入一筆持股，例如 AAPL 185 20。');
      return;
    }

    setHoldingsLoading(true);
    setHoldingsError('');
    try {
      const response = await reviewHoldings({ holdings: parsed, useAiCommittee, committeeModel });
      setHoldings(response);
      setHoldingsLastReviewed(localDateStr()); // 記錄今日已健檢（每天首次開啟自動健檢用）
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
              <SystemStatusBadge />
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
                        gooayeMap={gooayeMap}
                      />
                    ) : null
                  }
                />
              </ErrorBoundary>
            </div>
          ) : null}

          {activeTab === 'holdings' ? (
            <HoldingsManager useAiCommittee={useAiCommittee} committeeModel={committeeModel} gooayeMap={gooayeMap} />
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
                      以 AI 產業鏈主線股、寬鬆停利(35%/移損18%)讓獲利奔跑、持有上限 6 個月的波段策略，驗證整體投報率與對標超額報酬。經 5/10 年回測，此波段優於 3–6 月緊縮打法。
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

          {activeTab === 'portfolio' ? (
            <PortfolioTab
              recommendations={dailyScan?.picks ?? []}
              useAiCommittee={useAiCommittee}
              committeeModel={committeeModel}
              gooayeMap={gooayeMap}
            />
          ) : null}
        </section>
      </main>
      <Toaster />
    </>
  );
}

function fmtStamp(value: string | null | undefined): string {
  if (!value) return '—';
  // ISO（2026-06-13T06:30:00+08:00）取到分鐘；其他格式（RSS 日期）原樣顯示。
  const m = value.match(/^(\d{4}-\d{2}-\d{2})[T\s](\d{2}:\d{2})/);
  return m ? `${m[1]} ${m[2]}` : value;
}

function SystemStatusBadge() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [open, setOpen] = useState(false);
  const [err, setErr] = useState(false);

  useEffect(() => {
    let alive = true;
    const load = () =>
      fetchSystemStatus()
        .then((s) => {
          if (alive) {
            setStatus(s);
            setErr(false);
          }
        })
        .catch(() => {
          if (alive) setErr(true);
        });
    load();
    const id = window.setInterval(load, 5 * 60 * 1000); // 每 5 分鐘刷新
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, []);

  const ep = status?.gooaye.episode_title;
  const label = err ? '狀態取得失敗' : ep ? `股癌 ${ep}` : '載入中…';

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex h-9 max-w-[220px] items-center gap-1.5 rounded-md border border-slate-200 bg-white px-3 text-xs text-slate-600 hover:border-slate-400"
        title="系統更新狀態"
      >
        <span className={`h-2 w-2 shrink-0 rounded-full ${err ? 'bg-rose-500' : 'bg-emerald-500'}`} />
        <RadioTower className="h-3.5 w-3.5 shrink-0 text-slate-400" />
        <span className="truncate font-medium text-slate-900">{label}</span>
        <ChevronDown className="h-3.5 w-3.5 shrink-0 text-slate-400" />
      </button>
      {open ? (
        <div className="absolute right-0 z-50 mt-1 w-80 rounded-md border border-slate-200 bg-white p-3 text-xs shadow-lg">
          <div className="mb-2 flex items-center gap-1.5 font-semibold text-slate-900">
            <RadioTower className="h-4 w-4 text-slate-500" />
            系統更新狀態
          </div>
          {err ? (
            <div className="text-rose-600">無法取得狀態（後端 /status 未連線）。</div>
          ) : !status ? (
            <div className="text-slate-500">載入中…</div>
          ) : (
            <div className="space-y-2">
              <StatusRow label="股癌最新集數" value={status.gooaye.episode_title ?? '—'} strong />
              <StatusRow label="集數發布時間" value={fmtStamp(status.gooaye.published_date)} />
              <StatusRow
                label="累積點名觀點"
                value={status.gooaye.opinion_count != null ? `${status.gooaye.opinion_count} 則` : '—'}
              />
              <StatusRow
                label="背景掃描最後檢查"
                value={status.gooaye.last_checked ? fmtStamp(status.gooaye.last_checked) : '（每 2 小時自動掃描）'}
              />
              <div className="my-1 border-t border-slate-100" />
              <StatusRow
                label="每日 Top 50 報告"
                value={fmtStamp(status.daily_report.generated_at) !== '—' ? fmtStamp(status.daily_report.generated_at) : status.daily_report.generated_date ?? '—'}
                strong
              />
              <StatusRow label="報告標的數" value={status.daily_report.top_n != null ? `${status.daily_report.top_n} 檔` : '—'} />
              <div className="my-1 border-t border-slate-100" />
              <StatusRow label="伺服器時間" value={fmtStamp(status.server_time)} />
              <div className="pt-1 text-[11px] leading-4 text-slate-400">
                股癌集數每 2 小時自動掃描；每日 Top 50 報告由排程每日重建。時間有變動代表系統有在更新。
              </div>
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}

function StatusRow({ label, value, strong }: { label: string; value: string; strong?: boolean }) {
  return (
    <div className="flex items-start justify-between gap-3">
      <span className="text-slate-500">{label}</span>
      <span className={`text-right ${strong ? 'font-semibold text-slate-900' : 'text-slate-700'}`}>{value}</span>
    </div>
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
            <MetricCard label="長線 MA120" value={result.ma120 ? safeFixed(result.ma120, 2) : '-'} />
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
  gooayeMap,
}: {
  title: string;
  items: DetailResult[];
  selectedSymbol?: string;
  onSelect: (item: DetailResult) => void;
  gooayeMap?: Record<string, GooayeOpinion>;
}) {
  return (
    <Card className="rounded-lg border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent className="grid gap-2">
        {items.map((item, index) => {
          const op = gooayeMap?.[tickerBaseKey(item.symbol)];
          return (
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
                <div className="flex items-center gap-1.5 font-semibold text-slate-900">
                  {item.symbol}
                  {op ? (
                    <span className={`inline-flex items-center gap-0.5 rounded-full px-1.5 py-0.5 text-[10px] font-semibold ${gooayeStanceTone(op.sentiment_label)}`} title={op.core_logic}>
                      <Mic className="h-2.5 w-2.5" />股癌點名
                    </span>
                  ) : null}
                </div>
                <div className="text-xs text-slate-500">{'company_name' in item ? item.company_name : finalVerdict(item)}</div>
                {op ? <div className="mt-0.5 text-[11px] leading-4 text-slate-400">🎙️ {op.core_logic}</div> : null}
              </div>
              <div className={`font-medium ${actionTone(item.today_action)}`}>{item.today_action}</div>
              <div className={`font-medium ${actionTone(item.today_exit_action)}`}>{item.today_exit_action}</div>
              <div className="text-slate-700">{'daily_score' in item ? item.daily_score : item.composite_score}</div>
            </button>
          );
        })}
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

/* ===================== 跟單對帳本（紙上實單 vs 大盤） ===================== */

const PORTFOLIO_KEY = 'fund_paper_v1';
const PORTFOLIO_START_CAPITAL = 50000;

type PaperCurrency = 'USD' | 'TWD';
// avgCost：原幣每股成本（顯示用）；avgCostUsd：美金每股成本（帳戶結算用，買進當下以匯率換算）
type PaperPosition = { shares: number; avgCost: number; avgCostUsd: number; currency: PaperCurrency; name?: string };
type PaperTrade = {
  id: string;
  date: string;
  type: 'buy' | 'sell';
  ticker: string;
  shares: number;
  price: number; // 原幣每股價格
  priceUsd: number; // 美金每股價格（成交當下換算；帳戶以此結算）
  currency: PaperCurrency;
  name?: string;
  amount: number; // 美金成交金額（= shares × priceUsd）
  realized: number | null; // 美金已實現損益
  note: string;
};
type PaperBenchmark = { symbol: string; price: number; date: string; shares: number };
type PaperEquityPoint = { date: string; equity: number; benchmark: number | null };
type PaperAccount = {
  startCapital: number;
  startDate: string;
  startBenchmark: PaperBenchmark | null;
  cash: number;
  positions: Record<string, PaperPosition>;
  trades: PaperTrade[];
  realized: number;
  equityHistory: PaperEquityPoint[];
};

function paperToday(): string {
  return new Date().toLocaleDateString('en-CA');
}
function paperUid(): string {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}
function paperUsd(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '-';
  return '$' + Number(value).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
// 台股代號偵測（純數字，或 .TW/.TWO 結尾）。
function isTwTicker(ticker: string): boolean {
  const t = ticker.trim().toUpperCase();
  return /^\d+(\.TW[O]?)?$/.test(t) || t.endsWith('.TW') || t.endsWith('.TWO');
}
function paperCurrencyOf(ticker: string): PaperCurrency {
  return isTwTicker(ticker) ? 'TWD' : 'USD';
}
// 統一比對鍵：去掉 .TW/.TWO 後綴（持股健檢回傳常為正規化後代號，如 2330 → 2330.TW）。
function paperBaseKey(ticker: string): string {
  return ticker.trim().toUpperCase().replace(/\.(TW|TWO)$/, '');
}
// quotes['TWD=X'] = 1 美金可換多少台幣（≈32）；回傳「1 台幣 = 多少美金」。
function usdPerTwd(quotes: Record<string, number>): number | null {
  const r = quotes['TWD=X'];
  return r && r > 0 ? 1 / r : null;
}
// 把原幣每股價格換算成美金；台股需要匯率，缺匯率回 null。
function toUsdPrice(priceNative: number, currency: PaperCurrency, quotes: Record<string, number>): number | null {
  if (currency === 'USD') return priceNative;
  const fx = usdPerTwd(quotes);
  return fx == null ? null : priceNative * fx;
}
// 原幣金額顯示（台股加 NT$ 與 TWD 標記）。
function paperNative(value: number | null | undefined, currency: PaperCurrency): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '-';
  const n = Number(value).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  return currency === 'TWD' ? `NT$${n}` : `$${n}`;
}
function paperPct(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '-';
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
}
function toneOf(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return 'text-slate-900';
  return value > 0 ? 'text-emerald-700' : value < 0 ? 'text-rose-700' : 'text-slate-900';
}
function paperTrimNum(text?: string): number {
  const n = parseFloat(String(text ?? '').replace('%', ''));
  return Number.isNaN(n) ? 0 : n;
}
function verdictBadgeClass(rv: HoldingReviewResult): string {
  const n = paperTrimNum(rv.trim_ratio);
  if (n >= 100 || /賣出|停損|了結/.test(rv.verdict)) return 'border-rose-200 bg-rose-100 text-rose-700';
  if (n > 0) return 'border-amber-200 bg-amber-100 text-amber-700';
  return 'border-emerald-200 bg-emerald-100 text-emerald-700';
}
function blankPaperAccount(): PaperAccount {
  return {
    startCapital: PORTFOLIO_START_CAPITAL,
    startDate: paperToday(),
    startBenchmark: null,
    cash: PORTFOLIO_START_CAPITAL,
    positions: {},
    trades: [],
    realized: 0,
    equityHistory: [],
  };
}
function loadPaperAccount(): PaperAccount {
  try {
    const raw = localStorage.getItem(PORTFOLIO_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as PaperAccount;
      if (parsed && parsed.positions && Array.isArray(parsed.trades)) {
        // 舊資料相容：先前版本只有美股(USD)、未存 currency/avgCostUsd/priceUsd → 補上（USD: priceUsd=price）。
        for (const t of Object.keys(parsed.positions)) {
          const p = parsed.positions[t] as PaperPosition;
          if (p.currency == null) p.currency = paperCurrencyOf(t);
          if (p.avgCostUsd == null) p.avgCostUsd = p.avgCost;
        }
        parsed.trades = parsed.trades.map((t) => ({
          ...t,
          currency: t.currency ?? paperCurrencyOf(t.ticker),
          priceUsd: t.priceUsd ?? t.price,
        }));
        return parsed;
      }
    }
  } catch {
    /* ignore corrupt storage */
  }
  return blankPaperAccount();
}
// 原幣現價（無報價時回退原幣均價）。
function paperPriceNative(account: PaperAccount, quotes: Record<string, number>, ticker: string): number | null {
  if (quotes[ticker] != null) return quotes[ticker];
  const pos = account.positions[ticker];
  return pos ? pos.avgCost : null;
}
// 美金現價（台股以即時匯率換算；缺報價/匯率時回退美金成本）。
function paperPriceUsdOf(account: PaperAccount, quotes: Record<string, number>, ticker: string): number | null {
  const pos = account.positions[ticker];
  const native = paperPriceNative(account, quotes, ticker);
  const currency = pos ? pos.currency : paperCurrencyOf(ticker);
  if (native != null) {
    const usd = toUsdPrice(native, currency, quotes);
    if (usd != null) return usd;
  }
  return pos ? pos.avgCostUsd : null;
}
function paperMarketValue(account: PaperAccount, quotes: Record<string, number>): number {
  return Object.keys(account.positions).reduce((sum, t) => {
    const p = paperPriceUsdOf(account, quotes, t);
    return sum + (p ?? 0) * account.positions[t].shares;
  }, 0);
}
function paperEquity(account: PaperAccount, quotes: Record<string, number>): number {
  return account.cash + paperMarketValue(account, quotes);
}
// 以美金重放所有交易（用每筆成交當下的 priceUsd 結算，故重放具決定性）。
function rebuildPaperAccount(trades: PaperTrade[], startCapital: number): Pick<PaperAccount, 'cash' | 'positions' | 'realized' | 'trades'> {
  let cash = startCapital;
  let realized = 0;
  const positions: Record<string, PaperPosition> = {};
  const ordered = [...trades].sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
  const rebuilt: PaperTrade[] = [];
  for (const t of ordered) {
    const currency = t.currency ?? paperCurrencyOf(t.ticker);
    const usdAmount = t.shares * t.priceUsd;
    if (t.type === 'buy') {
      const pos = positions[t.ticker] ?? { shares: 0, avgCost: 0, avgCostUsd: 0, currency, name: t.name };
      const ns = pos.shares + t.shares;
      pos.avgCost = ns > 0 ? (pos.avgCost * pos.shares + t.shares * t.price) / ns : 0;
      pos.avgCostUsd = ns > 0 ? (pos.avgCostUsd * pos.shares + usdAmount) / ns : 0;
      pos.shares = ns;
      pos.currency = currency;
      if (t.name) pos.name = t.name;
      positions[t.ticker] = pos;
      cash -= usdAmount;
      rebuilt.push({ ...t, realized: null });
    } else {
      const pos = positions[t.ticker] ?? { shares: 0, avgCost: 0, avgCostUsd: 0, currency, name: t.name };
      const realizedTrade = (t.priceUsd - pos.avgCostUsd) * t.shares;
      pos.shares -= t.shares;
      cash += usdAmount;
      realized += realizedTrade;
      if (pos.shares <= 1e-6) delete positions[t.ticker];
      else positions[t.ticker] = pos;
      rebuilt.push({ ...t, realized: realizedTrade });
    }
  }
  return { cash, positions, realized, trades: rebuilt };
}

function PortfolioTab({
  recommendations,
  useAiCommittee,
  committeeModel,
  gooayeMap,
}: {
  recommendations: SP500DailyPick[];
  useAiCommittee: boolean;
  committeeModel: string;
  gooayeMap: Record<string, GooayeOpinion>;
}) {
  const [account, setAccount] = useState<PaperAccount>(() => loadPaperAccount());
  const accountRef = useRef(account);
  const [quotes, setQuotes] = useState<Record<string, number>>({});
  const [quoteMeta, setQuoteMeta] = useState<Record<string, { name?: string; currency?: string }>>({});
  const [quotesOk, setQuotesOk] = useState(false);
  const [quoteMsg, setQuoteMsg] = useState('');
  const [regime, setRegime] = useState<MarketRegimeSuggestion | null>(null);
  const [reviews, setReviews] = useState<Record<string, HoldingReviewResult>>({});
  const [reviewMsg, setReviewMsg] = useState('');
  const [reviewing, setReviewing] = useState(false);
  const [openRows, setOpenRows] = useState<Record<string, boolean>>({});

  const [tradeType, setTradeType] = useState<'buy' | 'sell'>('buy');
  const [fTicker, setFTicker] = useState('');
  const [fShares, setFShares] = useState('');
  const [fPrice, setFPrice] = useState('');
  const [fDate, setFDate] = useState(paperToday());
  const [fNote, setFNote] = useState('');
  const [formError, setFormError] = useState('');

  useEffect(() => {
    accountRef.current = account;
    localStorage.setItem(PORTFOLIO_KEY, JSON.stringify(account));
  }, [account]);

  const commit = useCallback((next: PaperAccount) => {
    accountRef.current = next;
    setAccount(next);
  }, []);

  const refreshQuotes = useCallback(async () => {
    const acc = accountRef.current;
    // 一律帶 SPY（大盤對照）與 TWD=X（美金/台幣匯率，台股換算用）。
    const syms = Array.from(new Set<string>([...Object.keys(acc.positions), 'SPY', 'TWD=X']));
    try {
      const items = await fetchQuotes(syms);
      const q: Record<string, number> = {};
      const meta: Record<string, { name?: string; currency?: string }> = {};
      items.forEach((it) => {
        if (it.ok && typeof it.price === 'number') {
          q[it.symbol] = it.price;
          meta[it.symbol] = { name: it.name, currency: it.currency };
        }
      });
      const ok = items.some((it) => it.ok);
      setQuotes(q);
      setQuoteMeta(meta);
      setQuotesOk(ok);
      setQuoteMsg(ok ? '' : '目前取不到即時報價，未實現損益暫以成本價估算。');
      setAccount((prev) => {
        let next = prev;
        if (!prev.startBenchmark && q.SPY) {
          next = {
            ...next,
            startBenchmark: { symbol: 'SPY', price: q.SPY, date: paperToday(), shares: prev.startCapital / q.SPY },
          };
        }
        // 報價回來後，補上持倉缺少的中文名（例：手動買入台股時尚未有報價）。
        const needName = Object.keys(next.positions).some((t) => !next.positions[t].name && meta[t]?.name && meta[t]?.name !== t);
        if (needName) {
          const patched: Record<string, PaperPosition> = {};
          for (const t of Object.keys(next.positions)) {
            const p = next.positions[t];
            patched[t] = !p.name && meta[t]?.name && meta[t]?.name !== t ? { ...p, name: meta[t]!.name } : p;
          }
          next = { ...next, positions: patched };
        }
        if (ok) {
          const eq = paperEquity(next, q);
          const bm = next.startBenchmark && q.SPY ? next.startBenchmark.shares * q.SPY : null;
          const d = paperToday();
          const hist = [...next.equityHistory];
          const last = hist[hist.length - 1];
          if (last && last.date === d) hist[hist.length - 1] = { date: d, equity: eq, benchmark: bm };
          else hist.push({ date: d, equity: eq, benchmark: bm });
          next = { ...next, equityHistory: hist };
        }
        accountRef.current = next;
        return next;
      });
    } catch {
      setQuotesOk(false);
      setQuoteMsg('此網站取不到即時報價（/quotes 無法連線）。仍可記錄交易，未實現損益以成本價估算。');
    }
  }, []);

  const reviewHoldingsNow = useCallback(async () => {
    const acc = accountRef.current;
    const tickers = Object.keys(acc.positions);
    if (!tickers.length) {
      setReviews({});
      setReviewMsg('');
      return;
    }
    setReviewing(true);
    try {
      const res = await reviewHoldings({
        holdings: tickers.map((t) => ({ ticker: t, cost_basis: acc.positions[t].avgCost, shares: acc.positions[t].shares })),
        period: '2y',
        useAiCommittee,
        committeeModel,
      });
      const map: Record<string, HoldingReviewResult> = {};
      res.forEach((r) => {
        if (r && r.symbol) map[paperBaseKey(r.symbol)] = r;
      });
      setReviews(map);
      setReviewMsg(`策略評估更新：${paperToday()}`);
    } catch {
      setReviewMsg('策略評估暫時無法取得（需後端 /simple-signals）。');
    } finally {
      setReviewing(false);
    }
  }, [useAiCommittee, committeeModel]);

  useEffect(() => {
    refreshQuotes();
    if (Object.keys(accountRef.current.positions).length) reviewHoldingsNow();
    fetchMarketRegime('us')
      .then((r) => setRegime(r.ok ? r : null))
      .catch(() => setRegime(null));
    // 僅在掛載時跑一次
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function submitTrade() {
    const ticker = fTicker.trim().toUpperCase();
    const shares = parseFloat(fShares);
    const price = parseFloat(fPrice); // 原幣每股
    const date = fDate || paperToday();
    const note = fNote.trim();
    setFormError('');
    if (!ticker) return setFormError('請輸入標的代號。');
    if (!(shares > 0)) return setFormError('股數需大於 0。');
    if (!(price > 0)) return setFormError('價格需大於 0。');

    const currency = paperCurrencyOf(ticker);
    const priceUsd = toUsdPrice(price, currency, quotes);
    if (priceUsd == null) {
      return setFormError('尚未取得美金/台幣匯率，無法換算台股。請先按「重新整理報價」後再記錄。');
    }
    const usdAmount = shares * priceUsd; // 帳戶以美金結算
    const name = quoteMeta[ticker]?.name && quoteMeta[ticker]?.name !== ticker ? quoteMeta[ticker]?.name : accountRef.current.positions[ticker]?.name;
    const acc = accountRef.current;

    if (tradeType === 'buy') {
      if (usdAmount > acc.cash + 1e-6) return setFormError(`現金不足：需 ${paperUsd(usdAmount)}，可用 ${paperUsd(acc.cash)}。`);
      const pos = acc.positions[ticker] ?? { shares: 0, avgCost: 0, avgCostUsd: 0, currency, name };
      const ns = pos.shares + shares;
      const next: PaperAccount = {
        ...acc,
        cash: acc.cash - usdAmount,
        positions: {
          ...acc.positions,
          [ticker]: {
            shares: ns,
            avgCost: (pos.avgCost * pos.shares + shares * price) / ns,
            avgCostUsd: (pos.avgCostUsd * pos.shares + usdAmount) / ns,
            currency,
            name: name ?? pos.name,
          },
        },
        trades: [...acc.trades, { id: paperUid(), date, type: 'buy', ticker, shares, price, priceUsd, currency, name, amount: usdAmount, realized: null, note }],
      };
      commit(next);
    } else {
      const pos = acc.positions[ticker];
      if (!pos || pos.shares < shares - 1e-6) return setFormError(`持股不足：目前持有 ${pos ? pos.shares : 0} 股。`);
      const realizedTrade = (priceUsd - pos.avgCostUsd) * shares; // 美金已實現
      const positions = { ...acc.positions };
      const remaining = pos.shares - shares;
      if (remaining <= 1e-6) delete positions[ticker];
      else positions[ticker] = { ...pos, shares: remaining };
      const next: PaperAccount = {
        ...acc,
        cash: acc.cash + usdAmount,
        realized: acc.realized + realizedTrade,
        positions,
        trades: [...acc.trades, { id: paperUid(), date, type: 'sell', ticker, shares, price, priceUsd, currency, name: name ?? pos.name, amount: usdAmount, realized: realizedTrade, note }],
      };
      commit(next);
    }
    setFShares('');
    setFPrice('');
    setFNote('');
    refreshQuotes();
    reviewHoldingsNow();
  }

  function deleteTrade(id: string) {
    if (!window.confirm('刪除這筆交易？系統會依剩餘交易重算整個帳戶。')) return;
    const acc = accountRef.current;
    const remainingTrades = acc.trades.filter((t) => t.id !== id);
    const rebuilt = rebuildPaperAccount(remainingTrades, acc.startCapital);
    commit({ ...acc, ...rebuilt });
    refreshQuotes();
    reviewHoldingsNow();
  }

  function prefillBuy(ticker: string, price?: number) {
    setTradeType('buy');
    setFTicker(ticker);
    if (price != null) setFPrice(String(price));
    setFDate(paperToday());
    setFormError('');
  }

  function exportData() {
    const blob = new Blob([JSON.stringify(accountRef.current, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `paper-fund-${paperToday()}.json`;
    a.click();
  }
  function importData(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = JSON.parse(String(reader.result)) as PaperAccount;
        if (!parsed.positions || !Array.isArray(parsed.trades)) throw new Error('bad');
        commit(parsed);
        refreshQuotes();
        reviewHoldingsNow();
      } catch {
        window.alert('檔案格式不正確。');
      }
    };
    reader.readAsText(file);
    event.target.value = '';
  }
  function resetAll() {
    if (!window.confirm('確定要清空帳戶、回到 $50,000 起始狀態？建議先「匯出備份」。')) return;
    commit(blankPaperAccount());
    setQuotes({});
    setReviews({});
    refreshQuotes();
  }

  const equity = useMemo(() => paperEquity(account, quotes), [account, quotes]);
  const marketValue = useMemo(() => paperMarketValue(account, quotes), [account, quotes]);
  const totalReturnPct = ((equity - account.startCapital) / account.startCapital) * 100;
  const benchmarkEquity = account.startBenchmark && quotes.SPY ? account.startBenchmark.shares * quotes.SPY : null;
  const benchmarkReturnPct = benchmarkEquity != null ? ((benchmarkEquity - account.startCapital) / account.startCapital) * 100 : null;
  const vsDeltaPct = benchmarkReturnPct != null ? totalReturnPct - benchmarkReturnPct : null;

  const sellSignals = Object.keys(account.positions)
    .map((t) => ({ ticker: t, rv: reviews[paperBaseKey(t)] }))
    .filter((x): x is { ticker: string; rv: HoldingReviewResult } => Boolean(x.rv) && paperTrimNum(x.rv!.trim_ratio) > 0);

  const positionTickers = Object.keys(account.positions);
  const recList = recommendations.slice(0, 20);
  const fxTwdPerUsd = quotes['TWD=X']; // 1 美金 = ? 台幣
  const formCurrency = paperCurrencyOf(fTicker || 'US');

  return (
    <div className="space-y-5">
      <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
        <div className="flex items-center gap-2">
          <Wallet className="h-5 w-5 text-slate-500" />
          <div>
            <div className="text-sm font-semibold text-slate-900">跟單對帳本</div>
            <div className="text-xs text-slate-500">
              起始本金 {paperUsd(account.startCapital)} ・ 開帳日 {account.startDate} ・ 依推薦自行操作，驗證能否贏過大盤(SPY)。資料只存在本機瀏覽器。
              {fxTwdPerUsd ? <>　匯率 1 USD ≈ {fxTwdPerUsd.toFixed(2)} TWD（台股自動換算為美金）。</> : null}
            </div>
          </div>
        </div>
      </div>

      {quoteMsg ? (
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-700">ℹ️ {quoteMsg}</div>
      ) : null}

      {sellSignals.length ? (
        <div className="rounded-lg border border-rose-300 bg-rose-50 p-4 text-sm text-rose-800">
          <div className="flex items-center gap-2 font-semibold">
            <AlertTriangle className="h-4 w-4" />
            策略對你的持股發出減碼／賣出訊號（請自行判斷是否執行）
          </div>
          <div className="mt-1.5 space-y-1 text-xs">
            {sellSignals.map((s) => (
              <div key={s.ticker}>
                <span className="font-semibold">{s.ticker}</span>：{s.rv.verdict}・建議出場 {s.rv.trim_ratio}・強度 {s.rv.urgency}
              </div>
            ))}
          </div>
        </div>
      ) : null}

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-6">
        <MetricCard label="總資產（現金＋持倉）" value={paperUsd(equity)} icon={<Wallet className="h-4 w-4" />} />
        <MetricCard label="總報酬" value={paperPct(totalReturnPct)} valueClassName={toneOf(totalReturnPct)} />
        <MetricCard label="vs 大盤(SPY)" value={vsDeltaPct == null ? '-' : paperPct(vsDeltaPct)} valueClassName={toneOf(vsDeltaPct)} />
        <MetricCard label="現金" value={paperUsd(account.cash)} />
        <MetricCard label="持倉市值" value={paperUsd(marketValue)} />
        <MetricCard label="已實現損益" value={paperUsd(account.realized)} valueClassName={toneOf(account.realized)} />
      </div>

      {benchmarkReturnPct != null ? (
        <div className="text-xs text-slate-500">
          大盤(SPY)同期報酬 {paperPct(benchmarkReturnPct)}（基準價 {paperUsd(account.startBenchmark?.price)} @ {account.startBenchmark?.date}）。
          {vsDeltaPct != null && vsDeltaPct >= 0 ? '　🎉 目前贏過大盤。' : '　目前落後大盤。'}
        </div>
      ) : (
        <div className="text-xs text-slate-500">等取得 SPY 報價後鎖定大盤對照基準。</div>
      )}

      {regime ? (
        <Card className="rounded-lg border-slate-200 bg-white shadow-sm">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <Gauge className="h-4 w-4 text-slate-500" />
              今日市場情緒 → 建議現金 / 股票比例
            </CardTitle>
            <CardDescription>依 VIX 與貪婪指數推估的風險預算，供你模擬調整持股水位參考。</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <MetricCard label="VIX 波動率" value={`${regime.vix_close.toFixed(2)}（${regime.vix_regime}）`} />
              <MetricCard label="貪婪指數" value={`${regime.fear_greed_score}（${regime.fear_greed_label}）`} />
              <MetricCard label="建議股票水位" value={`${regime.suggested_stock_pct}%`} valueClassName="text-emerald-700" />
              <MetricCard label="建議現金水位" value={`${regime.suggested_cash_pct}%`} valueClassName="text-sky-700" />
            </div>
            <div className="overflow-hidden rounded-full bg-slate-100">
              <div className="flex h-3 w-full">
                <div className="h-full bg-emerald-500" style={{ width: `${regime.suggested_stock_pct}%` }} />
                <div className="h-full bg-sky-400" style={{ width: `${regime.suggested_cash_pct}%` }} />
              </div>
            </div>
            <div className="text-xs text-slate-500">
              以目前總資產 {paperUsd(equity)} 換算 ≈ 股票 {paperUsd((equity * regime.suggested_stock_pct) / 100)} ／ 現金 {paperUsd((equity * regime.suggested_cash_pct) / 100)}。
              <span className="ml-1">house 觀點：{regime.action}（{regime.risk_budget}）。{regime.summary}</span>
            </div>
          </CardContent>
        </Card>
      ) : null}

      <Card className="rounded-lg border-slate-200 bg-white shadow-sm">
        <CardHeader className="pb-3">
          <CardTitle className="text-base">資產曲線 vs 大盤</CardTitle>
          <CardDescription>每次開啟此分頁（成功取得報價時）記錄一個每日資產點。藍線＝你的帳戶，橘線＝SPY 買進持有。</CardDescription>
        </CardHeader>
        <CardContent>
          <PaperEquityChart history={account.equityHistory} startCapital={account.startCapital} />
        </CardContent>
      </Card>

      <Card className="rounded-lg border-slate-200 bg-white shadow-sm">
        <CardHeader className="pb-3">
          <CardTitle className="text-base">記一筆交易</CardTitle>
          <CardDescription>依推薦或自己的判斷記錄買入 / 賣出。</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="inline-flex overflow-hidden rounded-md border border-slate-200">
            <button
              type="button"
              onClick={() => setTradeType('buy')}
              className={`px-4 py-2 text-sm font-medium ${tradeType === 'buy' ? 'bg-emerald-600 text-white' : 'bg-white text-slate-600'}`}
            >
              買入
            </button>
            <button
              type="button"
              onClick={() => setTradeType('sell')}
              className={`px-4 py-2 text-sm font-medium ${tradeType === 'sell' ? 'bg-rose-600 text-white' : 'bg-white text-slate-600'}`}
            >
              賣出
            </button>
          </div>
          <div className="grid gap-3 md:grid-cols-[1fr_110px_130px_150px_1fr_auto] md:items-end">
            <label className="block">
              <span className="mb-1 block text-xs font-semibold text-slate-600">標的代號</span>
              <Input list="paper-rec-tickers" value={fTicker} onChange={(e) => setFTicker(e.target.value)} placeholder="AAPL / 2330" className="h-10 bg-white" />
              <datalist id="paper-rec-tickers">
                {recList.map((r) => (
                  <option key={r.symbol} value={r.symbol} label={r.company_name && r.company_name !== r.symbol ? r.company_name : undefined} />
                ))}
              </datalist>
            </label>
            <label className="block">
              <span className="mb-1 block text-xs font-semibold text-slate-600">股數</span>
              <Input type="number" min="0" step="any" value={fShares} onChange={(e) => setFShares(e.target.value)} className="h-10 bg-white" />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs font-semibold text-slate-600">價格({formCurrency === 'TWD' ? 'TWD' : 'USD'})</span>
              <Input type="number" min="0" step="any" value={fPrice} onChange={(e) => setFPrice(e.target.value)} className="h-10 bg-white" />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs font-semibold text-slate-600">日期</span>
              <Input type="date" value={fDate} onChange={(e) => setFDate(e.target.value)} className="h-10 bg-white" />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs font-semibold text-slate-600">備註（選填）</span>
              <Input value={fNote} onChange={(e) => setFNote(e.target.value)} placeholder="依推薦 / 停損…" className="h-10 bg-white" />
            </label>
            <Button
              type="button"
              onClick={submitTrade}
              className={`h-10 md:w-28 ${tradeType === 'buy' ? 'bg-emerald-600 hover:bg-emerald-500' : 'bg-rose-600 hover:bg-rose-500'} text-white`}
            >
              {tradeType === 'buy' ? '記錄買入' : '記錄賣出'}
            </Button>
          </div>
          {formError ? <p className="text-sm text-rose-600">{formError}</p> : null}
        </CardContent>
      </Card>

      <Card className="rounded-lg border-slate-200 bg-white shadow-sm">
        <CardHeader className="pb-3">
          <CardTitle className="text-base">今日推薦（點「買入」自動帶入表單）</CardTitle>
          <CardDescription>來自「每日掃描」分頁的結果。請先到該分頁掃出 Top 50，這裡才會帶出可跟單的清單。</CardDescription>
        </CardHeader>
        <CardContent>
          {recList.length ? (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[560px] border-collapse text-left text-sm">
                <thead>
                  <tr className="text-slate-500">
                    <th className="py-1.5 pr-2 font-medium">標的</th>
                    <th className="py-1.5 pr-2 font-medium">每日分數</th>
                    <th className="py-1.5 pr-2 font-medium">建議</th>
                    <th className="py-1.5 pr-2 font-medium">最新收盤</th>
                    <th className="py-1.5 pr-2 font-medium" />
                  </tr>
                </thead>
                <tbody>
                  {recList.map((r) => {
                    const ccy = paperCurrencyOf(r.symbol);
                    const op = gooayeMap[tickerBaseKey(r.symbol)];
                    return (
                      <tr key={r.symbol} className="border-t border-slate-100">
                        <td className="py-1.5 pr-2 font-semibold text-slate-900">
                          <div className="flex items-center gap-1.5">
                            {r.symbol}
                            {r.company_name && r.company_name !== r.symbol ? (
                              <span className="font-normal text-slate-500">{r.company_name}</span>
                            ) : null}
                            {op ? (
                              <span className={`inline-flex items-center gap-0.5 rounded-full px-1.5 py-0.5 text-[10px] font-semibold ${gooayeStanceTone(op.sentiment_label)}`} title={op.core_logic}>
                                <Mic className="h-2.5 w-2.5" />股癌點名
                              </span>
                            ) : null}
                          </div>
                          {op ? <div className="mt-0.5 max-w-[260px] text-[11px] font-normal leading-4 text-slate-400">🎙️ {op.core_logic}</div> : null}
                        </td>
                        <td className="py-1.5 pr-2 text-slate-700">{r.daily_score}</td>
                        <td className={`py-1.5 pr-2 font-medium ${actionTone(r.action_label ?? r.today_action)}`}>{r.action_label ?? r.today_action}</td>
                        <td className="py-1.5 pr-2 text-slate-700">{paperNative(r.latest_close, ccy)}</td>
                        <td className="py-1.5 pr-2">
                          <Button type="button" onClick={() => prefillBuy(r.symbol, r.latest_close)} className="h-8 bg-emerald-600 px-3 text-xs text-white hover:bg-emerald-500">
                            買入
                          </Button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-sm text-slate-500">尚無推薦。請先到「每日掃描」分頁按「掃描 Top 50」（可切美股 / 台股）。</div>
          )}
        </CardContent>
      </Card>

      <Card className="rounded-lg border-slate-200 bg-white shadow-sm">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between gap-2">
            <div>
              <CardTitle className="text-base">目前持倉</CardTitle>
              <CardDescription>點一列看策略理由與買/賣強度。策略建議減碼/賣出時會在最上方紅框提醒。</CardDescription>
            </div>
            <div className="flex items-center gap-2">
              {reviewMsg ? <span className="text-xs text-slate-400">{reviewMsg}</span> : null}
              <Button type="button" onClick={reviewHoldingsNow} disabled={reviewing} className="h-9 border border-slate-200 bg-white text-slate-700 hover:border-slate-400">
                {reviewing ? '評估中…' : '🔍 重新評估持股'}
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {positionTickers.length ? (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[640px] border-collapse text-left text-sm">
                <thead>
                  <tr className="text-slate-500">
                    <th className="py-1.5 pr-2 font-medium">標的</th>
                    <th className="py-1.5 pr-2 font-medium">股數</th>
                    <th className="py-1.5 pr-2 font-medium">均價(原幣)</th>
                    <th className="py-1.5 pr-2 font-medium">現價(原幣)</th>
                    <th className="py-1.5 pr-2 font-medium">市值(USD)</th>
                    <th className="py-1.5 pr-2 font-medium">未實現(USD)</th>
                    <th className="py-1.5 pr-2 font-medium">報酬率</th>
                    <th className="py-1.5 pr-2 font-medium">策略訊號</th>
                  </tr>
                </thead>
                <tbody>
                  {positionTickers.map((t) => {
                    const pos = account.positions[t];
                    const curNative = paperPriceNative(account, quotes, t);
                    const curUsd = paperPriceUsdOf(account, quotes, t);
                    const mvUsd = curUsd != null ? curUsd * pos.shares : null;
                    const upl = curUsd != null ? (curUsd - pos.avgCostUsd) * pos.shares : null;
                    const uplPct = curUsd != null && pos.avgCostUsd > 0 ? ((curUsd - pos.avgCostUsd) / pos.avgCostUsd) * 100 : null;
                    const rv = reviews[paperBaseKey(t)];
                    const open = openRows[t];
                    return (
                      <Fragment key={t}>
                        <tr className="cursor-pointer border-t border-slate-100 hover:bg-slate-50" onClick={() => setOpenRows((prev) => ({ ...prev, [t]: !prev[t] }))}>
                          <td className="py-1.5 pr-2 font-semibold text-slate-900">
                            {t}
                            {pos.name && pos.name !== t ? <span className="ml-1 font-normal text-slate-500">{pos.name}</span> : null}
                          </td>
                          <td className="py-1.5 pr-2 text-slate-700">{pos.shares}</td>
                          <td className="py-1.5 pr-2 text-slate-700">{paperNative(pos.avgCost, pos.currency)}</td>
                          <td className="py-1.5 pr-2 text-slate-700">
                            {curNative != null ? paperNative(curNative, pos.currency) : '-'}
                            {quotes[t] == null ? <span className="ml-1 text-xs text-slate-400">(成本估)</span> : null}
                          </td>
                          <td className="py-1.5 pr-2 text-slate-700">{mvUsd != null ? paperUsd(mvUsd) : '-'}</td>
                          <td className={`py-1.5 pr-2 font-medium ${toneOf(upl)}`}>{upl != null ? paperUsd(upl) : '-'}</td>
                          <td className={`py-1.5 pr-2 font-medium ${toneOf(uplPct)}`}>{uplPct != null ? paperPct(uplPct) : '-'}</td>
                          <td className="py-1.5 pr-2">
                            {rv ? (
                              <span className="inline-flex items-center gap-1">
                                <span className={`rounded-full border px-2 py-0.5 text-xs font-semibold ${verdictBadgeClass(rv)}`}>{rv.verdict}</span>
                                <span className="text-xs text-slate-500">出{rv.trim_ratio}·強度{rv.urgency}</span>
                              </span>
                            ) : (
                              <span className="text-xs text-slate-400">—</span>
                            )}
                          </td>
                        </tr>
                        {open ? (
                          <tr className="border-t border-slate-100 bg-slate-50">
                            <td colSpan={8} className="px-2 py-3 text-xs leading-5 text-slate-600">
                              {rv ? (
                                <div className="space-y-1">
                                  <div>
                                    <span className="font-semibold">策略結論：</span>
                                    <span className={`ml-1 rounded-full border px-2 py-0.5 font-semibold ${verdictBadgeClass(rv)}`}>{rv.verdict}</span>
                                    <span className="ml-1">強度 {rv.urgency}・建議出場 {rv.trim_ratio}</span>
                                  </div>
                                  {rv.holding_reason ? <div>{rv.holding_reason}</div> : null}
                                  <div>
                                    <span className="font-semibold">操作區間：</span>趨勢 {rv.signal?.bias ?? '—'}　買進區 {rv.signal?.buy_zone ?? '—'}　賣出區 {rv.signal?.sell_zone ?? '—'}　停損 {rv.protective_stop || rv.signal?.stop_loss || '—'}
                                  </div>
                                  <div>
                                    <span className="font-semibold">今日動作：</span>買進 {rv.signal?.today_action ?? '—'}
                                    {rv.signal?.buy_strength ? `（強度 ${rv.signal.buy_strength}）` : ''}　賣出 {rv.signal?.today_exit_action ?? '—'}
                                  </div>
                                </div>
                              ) : (
                                <div className="text-slate-400">尚未評估。點上方「🔍 重新評估持股」取得策略訊號。</div>
                              )}
                            </td>
                          </tr>
                        ) : null}
                      </Fragment>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-sm text-slate-500">尚無持倉。</div>
          )}
        </CardContent>
      </Card>

      <Card className="rounded-lg border-slate-200 bg-white shadow-sm">
        <CardHeader className="pb-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <CardTitle className="text-base">交易紀錄</CardTitle>
            <div className="flex flex-wrap items-center gap-2">
              <Button type="button" onClick={refreshQuotes} className="h-9 border border-slate-200 bg-white text-slate-700 hover:border-slate-400">🔄 重新整理報價</Button>
              <Button type="button" onClick={exportData} className="h-9 border border-slate-200 bg-white text-slate-700 hover:border-slate-400">⬇ 匯出備份</Button>
              <label className="inline-flex h-9 cursor-pointer items-center rounded-md border border-slate-200 bg-white px-3 text-sm font-medium text-slate-700 hover:border-slate-400">
                ⬆ 匯入
                <input type="file" accept="application/json" className="hidden" onChange={importData} />
              </label>
              <Button type="button" onClick={resetAll} className="h-9 border border-slate-200 bg-white text-slate-700 hover:border-slate-400">♻ 重設帳戶</Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {account.trades.length ? (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[640px] border-collapse text-left text-sm">
                <thead>
                  <tr className="text-slate-500">
                    <th className="py-1.5 pr-2 font-medium">日期</th>
                    <th className="py-1.5 pr-2 font-medium">動作</th>
                    <th className="py-1.5 pr-2 font-medium">標的</th>
                    <th className="py-1.5 pr-2 font-medium">股數</th>
                    <th className="py-1.5 pr-2 font-medium">價格(原幣)</th>
                    <th className="py-1.5 pr-2 font-medium">金額(USD)</th>
                    <th className="py-1.5 pr-2 font-medium">已實現(USD)</th>
                    <th className="py-1.5 pr-2 font-medium">備註</th>
                    <th className="py-1.5 pr-2 font-medium" />
                  </tr>
                </thead>
                <tbody>
                  {[...account.trades]
                    .sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : 0))
                    .map((t) => {
                      const ccy = t.currency ?? paperCurrencyOf(t.ticker);
                      return (
                      <tr key={t.id} className="border-t border-slate-100">
                        <td className="py-1.5 pr-2 text-slate-600">{t.date}</td>
                        <td className="py-1.5 pr-2">
                          <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${t.type === 'buy' ? 'bg-emerald-100 text-emerald-700' : 'bg-rose-100 text-rose-700'}`}>
                            {t.type === 'buy' ? '買入' : '賣出'}
                          </span>
                        </td>
                        <td className="py-1.5 pr-2 font-semibold text-slate-900">
                          {t.ticker}
                          {t.name && t.name !== t.ticker ? <span className="ml-1 font-normal text-slate-500">{t.name}</span> : null}
                        </td>
                        <td className="py-1.5 pr-2 text-slate-700">{t.shares}</td>
                        <td className="py-1.5 pr-2 text-slate-700">{paperNative(t.price, ccy)}</td>
                        <td className="py-1.5 pr-2 text-slate-700">{paperUsd(t.amount)}</td>
                        <td className={`py-1.5 pr-2 font-medium ${toneOf(t.realized)}`}>{t.realized != null ? paperUsd(t.realized) : '-'}</td>
                        <td className="py-1.5 pr-2 text-slate-500">{t.note}</td>
                        <td className="py-1.5 pr-2">
                          <button type="button" onClick={() => deleteTrade(t.id)} className="text-rose-600 hover:text-rose-700" title="刪除">
                            ✕
                          </button>
                        </td>
                      </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-sm text-slate-500">尚無交易。</div>
          )}
        </CardContent>
      </Card>

      <p className="text-xs leading-5 text-slate-400">
        ⚠️ 本頁為紙上跟單模擬：資料只存在你這台瀏覽器（localStorage），不會上傳，換裝置請先「匯出備份」。報價來自 yfinance 最近收盤；大盤對照以「開帳當天把全部本金買進並持有 SPY」計算。非投資建議。
      </p>
    </div>
  );
}

function PaperEquityChart({ history, startCapital }: { history: PaperEquityPoint[]; startCapital: number }) {
  const points = history.filter((p) => p.equity != null);
  if (points.length < 2) {
    return <div className="text-sm text-slate-500">累積 2 天以上資料後顯示曲線。</div>;
  }
  const width = 860;
  const height = 220;
  const padding = 18;
  const equities = points.map((p) => p.equity);
  const benchmarks = points.map((p) => p.benchmark).filter((v): v is number => v != null);
  const all = equities.concat(benchmarks, [startCapital]);
  const minValue = Math.min(...all) * 0.99;
  const maxValue = Math.max(...all) * 1.01;
  const toX = (index: number) => padding + (index / Math.max(points.length - 1, 1)) * (width - padding * 2);
  const toY = (value: number) => height - padding - ((value - minValue) / Math.max(maxValue - minValue, 1)) * (height - padding * 2);
  const equityPath = buildPath(points.map((p, i) => ({ x: toX(i), y: toY(p.equity) })));
  const benchPath = buildPath(points.map((p, i) => (p.benchmark == null ? null : { x: toX(i), y: toY(p.benchmark) })).filter(Boolean) as Array<{ x: number; y: number }>);
  const baselineY = toY(startCapital);
  return (
    <div className="space-y-2">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full rounded-md border border-slate-200 bg-white">
        <line x1={padding} y1={baselineY} x2={width - padding} y2={baselineY} stroke="#cbd5e1" strokeWidth="1" strokeDasharray="4 4" />
        {benchPath ? <path d={benchPath} fill="none" stroke="#f97316" strokeWidth="2" /> : null}
        <path d={equityPath} fill="none" stroke="#2563eb" strokeWidth="2.2" />
      </svg>
      <div className="grid gap-2 text-xs text-slate-500 md:grid-cols-3">
        <div>藍線：你的帳戶</div>
        <div>橘線：SPY 買進持有</div>
        <div>灰虛線：起始本金 {paperUsd(startCapital)}</div>
      </div>
    </div>
  );
}

/* ===================== 我的真實持股健檢（結構化、記憶、每日自動） ===================== */

const REAL_HOLDINGS_V2_KEY = 'real_holdings_v2';
type RealHolding = { ticker: string; cost: number; shares: number };

function parseHoldingsLines(text: string): RealHolding[] {
  return text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => {
      const p = l.split(/[\s,，、]+/).filter(Boolean);
      return { ticker: (p[0] || '').toUpperCase(), cost: Number(p[1]) || 0, shares: Number(p[2]) || 0 };
    })
    .filter((h) => h.ticker);
}
function loadRealHoldingsV2(): { holdings: RealHolding[]; lastReviewed: string } {
  try {
    const raw = localStorage.getItem(REAL_HOLDINGS_V2_KEY);
    if (raw) {
      const d = JSON.parse(raw);
      if (Array.isArray(d.holdings)) return { holdings: d.holdings, lastReviewed: d.lastReviewed ?? '' };
    }
    // 從舊版（v1 文字框）遷移
    const v1 = localStorage.getItem('real_holdings_v1');
    if (v1) {
      const d = JSON.parse(v1);
      return { holdings: parseHoldingsLines(d.text || ''), lastReviewed: d.lastReviewed ?? '' };
    }
  } catch {
    /* ignore */
  }
  return { holdings: [], lastReviewed: '' };
}
function saveRealHoldingsV2(holdings: RealHolding[], lastReviewed: string): void {
  try {
    localStorage.setItem(REAL_HOLDINGS_V2_KEY, JSON.stringify({ holdings, lastReviewed }));
  } catch {
    /* ignore */
  }
}

function HoldingsManager({
  useAiCommittee,
  committeeModel,
  gooayeMap,
}: {
  useAiCommittee: boolean;
  committeeModel: string;
  gooayeMap: Record<string, GooayeOpinion>;
}) {
  const initial = loadRealHoldingsV2();
  const [holdings, setHoldings] = useState<RealHolding[]>(initial.holdings);
  const [lastReviewed, setLastReviewed] = useState(initial.lastReviewed);
  const holdingsRef = useRef(holdings);
  const [reviews, setReviews] = useState<Record<string, HoldingReviewResult>>({});
  const [names, setNames] = useState<Record<string, string>>({});
  const [reviewing, setReviewing] = useState(false);
  const [error, setError] = useState('');
  const [openRows, setOpenRows] = useState<Record<string, boolean>>({});
  const [fTicker, setFTicker] = useState('');
  const [fCost, setFCost] = useState('');
  const [fShares, setFShares] = useState('');

  useEffect(() => {
    holdingsRef.current = holdings;
    saveRealHoldingsV2(holdings, lastReviewed);
  }, [holdings, lastReviewed]);

  const runReview = useCallback(async () => {
    const list = holdingsRef.current;
    if (!list.length) {
      setReviews({});
      return;
    }
    setReviewing(true);
    setError('');
    try {
      const res = await reviewHoldings({
        holdings: list.map((h) => ({ ticker: h.ticker, cost_basis: h.cost, shares: h.shares })),
        useAiCommittee,
        committeeModel,
      });
      const map: Record<string, HoldingReviewResult> = {};
      res.forEach((r) => {
        if (r && r.symbol) map[tickerBaseKey(r.symbol)] = r;
      });
      setReviews(map);
      setLastReviewed(localDateStr());
      try {
        const q = await fetchQuotes(list.map((h) => h.ticker));
        const nm: Record<string, string> = {};
        q.forEach((it) => {
          if (it.ok && it.name && it.name !== it.symbol) nm[tickerBaseKey(it.symbol)] = it.name;
        });
        setNames(nm);
      } catch {
        /* 名稱非必要 */
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : '持股健檢失敗。');
    } finally {
      setReviewing(false);
    }
  }, [useAiCommittee, committeeModel]);

  useEffect(() => {
    // 每天首次開啟、已有持股時自動健檢一次
    if (holdingsRef.current.length && initial.lastReviewed !== localDateStr()) runReview();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function addHolding() {
    const t = fTicker.trim().toUpperCase();
    const c = parseFloat(fCost);
    const s = parseFloat(fShares);
    setError('');
    if (!t) return setError('請輸入代碼。');
    if (!(c > 0)) return setError('平均成本需大於 0。');
    if (!(s > 0)) return setError('股數需大於 0。');
    const next = [...holdingsRef.current.filter((h) => h.ticker !== t), { ticker: t, cost: c, shares: s }];
    holdingsRef.current = next;
    setHoldings(next);
    setFTicker('');
    setFCost('');
    setFShares('');
    runReview();
  }
  function removeHolding(t: string) {
    const next = holdingsRef.current.filter((h) => h.ticker !== t);
    holdingsRef.current = next;
    setHoldings(next);
    runReview();
  }

  const sellSignals = holdings
    .map((h) => ({ ticker: h.ticker, rv: reviews[tickerBaseKey(h.ticker)] }))
    .filter((x): x is { ticker: string; rv: HoldingReviewResult } => Boolean(x.rv) && paperTrimNum(x.rv!.trim_ratio) > 0);

  return (
    <div className="space-y-5">
      <Card className="rounded-lg border-slate-200 bg-white shadow-sm">
        <CardHeader className="pb-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <CardTitle className="flex items-center gap-2 text-base">
                <BriefcaseBusiness className="h-4 w-4" />
                我的真實持股健檢
              </CardTitle>
              <CardDescription>新增持股後系統會記住；每天首次開啟此頁自動健檢，依波段策略判斷續抱／減碼／賣出，不用每次重 key。</CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400">{lastReviewed ? `上次健檢：${lastReviewed}` : '尚未健檢'}</span>
              <Button type="button" onClick={() => runReview()} disabled={reviewing || !holdings.length} className="h-9 border border-slate-200 bg-white text-slate-700 hover:border-slate-400">
                {reviewing ? '健檢中…' : '🔍 重新健檢'}
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid gap-3 md:grid-cols-[1fr_140px_120px_auto] md:items-end">
            <label className="block">
              <span className="mb-1 block text-xs font-semibold text-slate-600">代碼（美股 AAPL／台股 2330）</span>
              <Input value={fTicker} onChange={(e) => setFTicker(e.target.value)} placeholder="AAPL / 2330" className="h-10 bg-white" />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs font-semibold text-slate-600">平均成本（原幣）</span>
              <Input type="number" min="0" step="any" value={fCost} onChange={(e) => setFCost(e.target.value)} className="h-10 bg-white" />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs font-semibold text-slate-600">股數</span>
              <Input type="number" min="0" step="any" value={fShares} onChange={(e) => setFShares(e.target.value)} className="h-10 bg-white" />
            </label>
            <Button type="button" onClick={addHolding} className="h-10 bg-slate-950 text-white hover:bg-slate-800 md:w-28">
              新增持股
            </Button>
          </div>
          {error ? <p className="text-sm text-rose-600">{error}</p> : null}
        </CardContent>
      </Card>

      {sellSignals.length ? (
        <div className="rounded-lg border border-rose-300 bg-rose-50 p-4 text-sm text-rose-800">
          <div className="flex items-center gap-2 font-semibold">
            <AlertTriangle className="h-4 w-4" />
            策略對你的真實持股發出減碼／賣出訊號（請自行判斷是否執行）
          </div>
          <div className="mt-1.5 space-y-1 text-xs">
            {sellSignals.map((s) => (
              <div key={s.ticker}>
                <span className="font-semibold">{s.ticker}</span>：{s.rv.verdict}・建議出場 {s.rv.trim_ratio}・強度 {s.rv.urgency}
              </div>
            ))}
          </div>
        </div>
      ) : null}

      <Card className="rounded-lg border-slate-200 bg-white shadow-sm">
        <CardHeader className="pb-3">
          <CardTitle className="text-base">持股健檢結果</CardTitle>
          <CardDescription>點一列看策略理由與操作區間。報酬率以你的成本對最新收盤計算（原幣）。</CardDescription>
        </CardHeader>
        <CardContent>
          {holdings.length ? (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[680px] border-collapse text-left text-sm">
                <thead>
                  <tr className="text-slate-500">
                    <th className="py-1.5 pr-2 font-medium">標的</th>
                    <th className="py-1.5 pr-2 font-medium">股數</th>
                    <th className="py-1.5 pr-2 font-medium">成本(原幣)</th>
                    <th className="py-1.5 pr-2 font-medium">現價(原幣)</th>
                    <th className="py-1.5 pr-2 font-medium">報酬率</th>
                    <th className="py-1.5 pr-2 font-medium">策略訊號</th>
                    <th className="py-1.5 pr-2 font-medium" />
                  </tr>
                </thead>
                <tbody>
                  {holdings.map((h) => {
                    const key = tickerBaseKey(h.ticker);
                    const rv = reviews[key];
                    const ccy = paperCurrencyOf(h.ticker);
                    const name = names[key];
                    const cur = rv ? rv.latest_close : null;
                    const pnl = rv ? rv.pnl_pct : null;
                    const op = gooayeMap[key];
                    const open = openRows[h.ticker];
                    return (
                      <Fragment key={h.ticker}>
                        <tr className="cursor-pointer border-t border-slate-100 hover:bg-slate-50" onClick={() => setOpenRows((prev) => ({ ...prev, [h.ticker]: !prev[h.ticker] }))}>
                          <td className="py-1.5 pr-2 font-semibold text-slate-900">
                            <div className="flex items-center gap-1.5">
                              {h.ticker}
                              {name ? <span className="font-normal text-slate-500">{name}</span> : null}
                              {op ? (
                                <span className={`inline-flex items-center gap-0.5 rounded-full px-1.5 py-0.5 text-[10px] font-semibold ${gooayeStanceTone(op.sentiment_label)}`} title={op.core_logic}>
                                  <Mic className="h-2.5 w-2.5" />股癌點名
                                </span>
                              ) : null}
                            </div>
                          </td>
                          <td className="py-1.5 pr-2 text-slate-700">{h.shares}</td>
                          <td className="py-1.5 pr-2 text-slate-700">{paperNative(h.cost, ccy)}</td>
                          <td className="py-1.5 pr-2 text-slate-700">{cur != null ? paperNative(cur, ccy) : '-'}</td>
                          <td className={`py-1.5 pr-2 font-medium ${toneOf(pnl)}`}>{pnl != null ? paperPct(pnl) : '-'}</td>
                          <td className="py-1.5 pr-2">
                            {rv ? (
                              <span className="inline-flex items-center gap-1">
                                <span className={`rounded-full border px-2 py-0.5 text-xs font-semibold ${verdictBadgeClass(rv)}`}>{rv.verdict}</span>
                                <span className="text-xs text-slate-500">出{rv.trim_ratio}·強度{rv.urgency}</span>
                              </span>
                            ) : (
                              <span className="text-xs text-slate-400">{reviewing ? '健檢中…' : '—'}</span>
                            )}
                          </td>
                          <td className="py-1.5 pr-2">
                            <button type="button" onClick={(e) => { e.stopPropagation(); removeHolding(h.ticker); }} className="text-rose-600 hover:text-rose-700" title="刪除持股">
                              ✕
                            </button>
                          </td>
                        </tr>
                        {open ? (
                          <tr className="border-t border-slate-100 bg-slate-50">
                            <td colSpan={7} className="px-2 py-3 text-xs leading-5 text-slate-600">
                              {rv ? (
                                <div className="space-y-1">
                                  <div>
                                    <span className="font-semibold">策略結論：</span>
                                    <span className={`ml-1 rounded-full border px-2 py-0.5 font-semibold ${verdictBadgeClass(rv)}`}>{rv.verdict}</span>
                                    <span className="ml-1">強度 {rv.urgency}・建議出場 {rv.trim_ratio}</span>
                                  </div>
                                  {rv.holding_reason ? <div>{rv.holding_reason}</div> : null}
                                  <div>
                                    <span className="font-semibold">操作區間：</span>趨勢 {rv.signal?.bias ?? '—'}　買進區 {rv.signal?.buy_zone ?? '—'}　賣出區 {rv.signal?.sell_zone ?? '—'}　停損 {rv.protective_stop || rv.signal?.stop_loss || '—'}
                                  </div>
                                  {op ? <div className="text-slate-500">🎙️ 股癌：{op.core_logic}</div> : null}
                                </div>
                              ) : (
                                <div className="text-slate-400">尚未健檢，點「🔍 重新健檢」。</div>
                              )}
                            </td>
                          </tr>
                        ) : null}
                      </Fragment>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-sm text-slate-500">尚無持股。用上方表單新增（例：AAPL 185 20、台股 2330 785 1），系統會記住並每天自動健檢。</div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
