import type { ChangeEvent, ReactNode } from 'react';
import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  BriefcaseBusiness,
  ChevronDown,
  Gauge,
  LineChart,
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
  fetchMarketRegime,
  fetchQuotes,
  fetchSystemStatus,
  type SystemStatus,
  type MarketRegimeSuggestion,
} from './services/simple-signal-api';
import { evaluateHoldings, fetchStrategyReport, type EvaluatedHolding, type RankRow } from './services/strategy-api';
import { MyHoldingsPanel, StockLookupPanel, StrategyBacktestPanel, StrategyPicksPanel } from './StrategyViews';

type TabKey = 'analyze' | 'daily' | 'holdings' | 'backtest' | 'portfolio';

const TABS: { key: TabKey; label: string; icon: typeof Search; hint: string }[] = [
  { key: 'analyze', label: '個股查詢', icon: Search, hint: '查任一檔的策略排名' },
  { key: 'daily', label: '策略精選', icon: Sparkles, hint: '本月買進名單（單一策略）' },
  { key: 'holdings', label: '我的持股', icon: BriefcaseBusiness, hint: '加碼 / 減碼 / 換股提醒' },
  { key: 'backtest', label: '策略回測', icon: LineChart, hint: '對標 VOO / 0050' },
  { key: 'portfolio', label: '跟單對帳本', icon: Wallet, hint: '5 萬美金實單跟蹤 vs 大盤' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<TabKey>('holdings');

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
                <div className="text-sm font-semibold leading-tight text-slate-900">AI 科技股動能策略</div>
                <div className="text-xs text-slate-500">單一策略 ・ 我的持股提醒 ・ 對標 VOO / 0050</div>
              </div>
            </div>
            <div className="flex items-center gap-2">
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

          {activeTab === 'analyze' ? <StockLookupPanel /> : null}
          {activeTab === 'daily' ? <StrategyPicksPanel /> : null}
          {activeTab === 'holdings' ? <MyHoldingsPanel /> : null}
          {activeTab === 'backtest' ? <StrategyBacktestPanel /> : null}
          {activeTab === 'portfolio' ? <PortfolioTab /> : null}
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
  const codeAt = fmtStamp(status?.code?.updated_at);

  return (
    <div className="relative flex items-center gap-2">
      {codeAt !== '—' ? (
        <span
          className="hidden whitespace-nowrap text-[11px] text-slate-400 md:inline"
          title="後端程式最後更新（部署/建置）時間"
        >
          程式更新：{codeAt}
        </span>
      ) : null}
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
              <StatusRow label="程式更新時間" value={fmtStamp(status.code?.updated_at)} strong />
              {status.code?.commit ? <StatusRow label="程式版本 (commit)" value={status.code.commit} /> : null}
              <div className="my-1 border-t border-slate-100" />
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
              {status.nicolas ? (
                <>
                  <div className="my-1 border-t border-slate-100" />
                  <StatusRow label="尼可拉斯楊最新一集" value={status.nicolas.episode_title ?? '—'} strong />
                  <StatusRow label="發布時間" value={fmtStamp(status.nicolas.published_date)} />
                  <StatusRow label="觀點覆蓋" value={status.nicolas.opinion_count != null ? `${status.nicolas.opinion_count} 則` : '—'} />
                  {status.nicolas.url ? (
                    <a href={status.nicolas.url} target="_blank" rel="noreferrer" className="block text-right text-[11px] text-violet-600 hover:underline">▶ 看這集</a>
                  ) : null}
                </>
              ) : null}
              <div className="my-1 border-t border-slate-100" />
              <StatusRow label="伺服器時間" value={fmtStamp(status.server_time)} />
              <div className="pt-1 text-[11px] leading-4 text-slate-400">
                股癌集數每 2 小時自動掃描；尼可拉斯楊由其 YouTube 頻道每 2 小時自動追蹤最新一集；每日 Top 50 報告由排程每日重建。時間有變動代表系統有在更新。
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

function buildPath(points: Array<{ x: number; y: number }>): string {
  if (points.length === 0) return '';
  return points.map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`).join(' ');
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
  startBenchmarkQQQ?: PaperBenchmark | null;
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
function verdictBadgeClass(rv: EvaluatedHolding): string {
  if (rv.action === '賣出換股') return 'border-rose-200 bg-rose-100 text-rose-700';
  if (rv.action === '減碼') return 'border-amber-200 bg-amber-100 text-amber-700';
  if (rv.action === '加碼') return 'border-emerald-200 bg-emerald-100 text-emerald-700';
  return 'border-sky-200 bg-sky-50 text-sky-700';
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

function PortfolioTab() {
  const [account, setAccount] = useState<PaperAccount>(() => loadPaperAccount());
  const accountRef = useRef(account);
  const [quotes, setQuotes] = useState<Record<string, number>>({});
  const [quoteMeta, setQuoteMeta] = useState<Record<string, { name?: string; currency?: string }>>({});
  const [, setQuotesOk] = useState(false);
  const [quoteMsg, setQuoteMsg] = useState('');
  const [regime, setRegime] = useState<MarketRegimeSuggestion | null>(null);
  const [reviews, setReviews] = useState<Record<string, EvaluatedHolding>>({});
  const [stratRows, setStratRows] = useState<(RankRow & { market: 'us' | 'tw' })[]>([]);
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
    // 一律帶 SPY、QQQ（大盤對照）與 TWD=X（美金/台幣匯率，台股換算用）。
    const syms = Array.from(new Set<string>([...Object.keys(acc.positions), 'SPY', 'QQQ', 'TWD=X']));
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
        if (!next.startBenchmarkQQQ && q.QQQ) {
          next = {
            ...next,
            startBenchmarkQQQ: { symbol: 'QQQ', price: q.QQQ, date: paperToday(), shares: prev.startCapital / q.QQQ },
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
      const res = await evaluateHoldings(
        tickers.map((t) => ({ ticker: t, cost: acc.positions[t].avgCost, shares: acc.positions[t].shares })),
      );
      const map: Record<string, EvaluatedHolding> = {};
      res.holdings.forEach((r) => {
        if (r && r.symbol) map[paperBaseKey(r.symbol)] = r;
      });
      setReviews(map);
      setReviewMsg(`策略評估更新：${paperToday()}`);
    } catch {
      setReviewMsg('策略評估暫時無法取得（需後端 /strategy）。');
    } finally {
      setReviewing(false);
    }
  }, []);

  useEffect(() => {
    fetchStrategyReport()
      .then((rep) =>
        setStratRows(
          (['us', 'tw'] as const).flatMap((m) =>
            (rep.markets[m]?.rows ?? []).slice(0, rep.markets[m]?.top_n ?? 10).map((r) => ({ ...r, market: m })),
          ),
        ),
      )
      .catch(() => setStratRows([]));
  }, []);

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
  // 那斯達克 100（QQQ）對照——科技/AI 部位重，QQQ 是更貼切也更難贏的標竿。
  const benchmarkQQQEquity = account.startBenchmarkQQQ && quotes.QQQ ? account.startBenchmarkQQQ.shares * quotes.QQQ : null;
  const benchmarkQQQReturnPct = benchmarkQQQEquity != null ? ((benchmarkQQQEquity - account.startCapital) / account.startCapital) * 100 : null;
  const vsQQQDeltaPct = benchmarkQQQReturnPct != null ? totalReturnPct - benchmarkQQQReturnPct : null;

  // 持股集中度風險：每檔佔總資產比重、單一過大 / 類股(台股) 過重 / 現金未布署 警示。
  const concentration = useMemo(() => {
    if (equity <= 0) return null;
    const rows = Object.keys(account.positions).map((t) => {
      const p = account.positions[t];
      const px = quotes[t] ?? p.avgCost;
      const usd = toUsdPrice(px, p.currency, quotes);
      const mv = usd != null ? usd * p.shares : p.avgCostUsd * p.shares;
      return { ticker: t, name: p.name, currency: p.currency, mv, pct: (mv / equity) * 100 };
    }).sort((a, b) => b.mv - a.mv);
    const twPct = rows.filter((r) => r.currency === 'TWD').reduce((s, r) => s + r.pct, 0);
    const cashPct = (account.cash / equity) * 100;
    const warnings: Array<{ level: 'danger' | 'warn'; text: string }> = [];
    const SINGLE_CAP = 15;
    rows.filter((r) => r.pct > SINGLE_CAP).forEach((r) =>
      warnings.push({ level: r.pct >= 25 ? 'danger' : 'warn', text: `${r.ticker}${r.name ? `（${r.name}）` : ''} 佔 ${r.pct.toFixed(0)}%，超過單一持股上限 ${SINGLE_CAP}%${r.pct >= 25 ? '——過度集中，強烈建議減碼分散' : '，考慮減碼'}` }),
    );
    if (twPct >= 40) warnings.push({ level: twPct >= 55 ? 'danger' : 'warn', text: `台股合計佔 ${twPct.toFixed(0)}%，單一市場過度集中` });
    if (cashPct >= 30) warnings.push({ level: 'warn', text: `現金佔 ${cashPct.toFixed(0)}% 未布署，漲勢中會拖累相對大盤表現` });
    return { rows, twPct, cashPct, warnings, singleCap: SINGLE_CAP };
  }, [account.positions, account.cash, quotes, equity]);

  const sellSignals = Object.keys(account.positions)
    .map((t) => ({ ticker: t, rv: reviews[paperBaseKey(t)] }))
    .filter((x): x is { ticker: string; rv: EvaluatedHolding } => Boolean(x.rv) && ['賣出換股', '減碼'].includes(x.rv!.action));

  const positionTickers = Object.keys(account.positions);
  const recList = stratRows;
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
              起始本金 {paperUsd(account.startCapital)} ・ 開帳日 {account.startDate} ・ 依策略名單自行操作，驗證能否贏過大盤(SPY)。資料只存在本機瀏覽器。
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
                <span className="font-semibold">{s.ticker}</span>：{s.rv.action}・排名 {s.rv.rank ?? '—'}・{s.rv.reason}
              </div>
            ))}
          </div>
        </div>
      ) : null}

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-7">
        <MetricCard label="總資產（現金＋持倉）" value={paperUsd(equity)} icon={<Wallet className="h-4 w-4" />} />
        <MetricCard label="總報酬" value={paperPct(totalReturnPct)} valueClassName={toneOf(totalReturnPct)} />
        <MetricCard label="vs 大盤(SPY)" value={vsDeltaPct == null ? '-' : paperPct(vsDeltaPct)} valueClassName={toneOf(vsDeltaPct)} />
        <MetricCard label="vs 那斯達克(QQQ)" value={vsQQQDeltaPct == null ? '-' : paperPct(vsQQQDeltaPct)} valueClassName={toneOf(vsQQQDeltaPct)} />
        <MetricCard label="現金" value={paperUsd(account.cash)} />
        <MetricCard label="持倉市值" value={paperUsd(marketValue)} />
        <MetricCard label="已實現損益" value={paperUsd(account.realized)} valueClassName={toneOf(account.realized)} />
      </div>

      {concentration && (concentration.warnings.length > 0 || concentration.rows.length > 0) ? (
        <div className={`rounded-lg border p-4 ${concentration.warnings.some((w) => w.level === 'danger') ? 'border-rose-300 bg-rose-50' : concentration.warnings.length ? 'border-amber-300 bg-amber-50' : 'border-slate-200 bg-white'}`}>
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-800">
            <ShieldAlert className="h-4 w-4" />持股集中度 / 風險檢查
          </div>
          {concentration.warnings.length ? (
            <div className="mt-2 space-y-1 text-xs">
              {concentration.warnings.map((w, i) => (
                <div key={i} className={w.level === 'danger' ? 'font-semibold text-rose-700' : 'text-amber-700'}>
                  {w.level === 'danger' ? '🔴' : '🟠'} {w.text}
                </div>
              ))}
            </div>
          ) : (
            <div className="mt-2 text-xs text-emerald-700">🟢 配置分散度健康：無單一持股超過 {concentration.singleCap}%、無單一市場過度集中。</div>
          )}
          <div className="mt-3 flex flex-wrap gap-1.5">
            {concentration.rows.map((r) => (
              <span key={r.ticker} className={`rounded-full border px-2 py-0.5 text-[11px] ${r.pct >= 25 ? 'border-rose-200 bg-rose-100 text-rose-700' : r.pct > concentration.singleCap ? 'border-amber-200 bg-amber-100 text-amber-700' : 'border-slate-200 bg-slate-50 text-slate-600'}`}>
                {r.ticker} {r.pct.toFixed(0)}%
              </span>
            ))}
            <span className="rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-[11px] text-slate-500">現金 {concentration.cashPct.toFixed(0)}%</span>
          </div>
          <div className="mt-2 text-[11px] text-slate-500">原則：單一持股 ≤ {concentration.singleCap}%、單一市場不過半、現金別長期閒置。集中度是賺賠的最大變數之一，賺到的獲利記得分散保護。</div>
        </div>
      ) : null}

      {benchmarkReturnPct != null ? (
        <div className="text-xs text-slate-500">
          SPY 同期 {paperPct(benchmarkReturnPct)}（基準 @ {account.startBenchmark?.date}）
          {benchmarkQQQReturnPct != null ? <>　·　QQQ 同期 {paperPct(benchmarkQQQReturnPct)}（基準 @ {account.startBenchmarkQQQ?.date}）</> : null}
          。{vsDeltaPct != null && vsDeltaPct >= 0 ? '🎉 贏 SPY' : '落後 SPY'}
          {vsQQQDeltaPct != null ? (vsQQQDeltaPct >= 0 ? '、贏 QQQ。' : '、落後 QQQ。') : '。'}
          {account.startBenchmarkQQQ && account.startBenchmark && account.startBenchmarkQQQ.date !== account.startBenchmark.date ? '　（QQQ 對照較晚啟用，基準日不同、僅供參考）' : ''}
        </div>
      ) : (
        <div className="text-xs text-slate-500">等取得 SPY／QQQ 報價後鎖定大盤對照基準。</div>
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
                  <option key={r.symbol} value={r.symbol} label={r.name || undefined} />
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
          <CardTitle className="text-base">策略買進名單（點「買入」自動帶入表單）</CardTitle>
          <CardDescription>來自「策略精選」：AI 科技股池中動能排名在買進區的標的（每月初調整）。</CardDescription>
        </CardHeader>
        <CardContent>
          {recList.length ? (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[560px] border-collapse text-left text-sm">
                <thead>
                  <tr className="text-slate-500">
                    <th className="py-1.5 pr-2 font-medium">標的</th>
                    <th className="py-1.5 pr-2 font-medium">市場</th>
                    <th className="py-1.5 pr-2 font-medium">排名</th>
                    <th className="py-1.5 pr-2 font-medium">近 6 月</th>
                    <th className="py-1.5 pr-2 font-medium">最新收盤</th>
                    <th className="py-1.5 pr-2 font-medium" />
                  </tr>
                </thead>
                <tbody>
                  {recList.map((r) => {
                    const ccy = paperCurrencyOf(r.symbol);
                    return (
                      <tr key={r.symbol} className="border-t border-slate-100">
                        <td className="py-1.5 pr-2 font-semibold text-slate-900">
                          {r.symbol}
                          {r.name ? <span className="ml-1 font-normal text-slate-500">{r.name}</span> : null}
                        </td>
                        <td className="py-1.5 pr-2 text-slate-600">{r.market === 'us' ? '美股' : '台股'}</td>
                        <td className="py-1.5 pr-2 text-slate-700">{r.rank}</td>
                        <td className={`py-1.5 pr-2 ${toneOf(r.ret_6m_pct)}`}>{paperPct(r.ret_6m_pct)}</td>
                        <td className="py-1.5 pr-2 text-slate-700">{paperNative(r.close, ccy)}</td>
                        <td className="py-1.5 pr-2">
                          <Button type="button" onClick={() => prefillBuy(r.symbol, r.close)} className="h-8 bg-emerald-600 px-3 text-xs text-white hover:bg-emerald-500">
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
            <div className="text-sm text-slate-500">策略名單讀取中（或後端暫時無法連線）。</div>
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
                                <span className={`rounded-full border px-2 py-0.5 text-xs font-semibold ${verdictBadgeClass(rv)}`}>{rv.action}</span>
                                <span className="text-xs text-slate-500">排名 {rv.rank ?? '—'}</span>
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
                                    <span className={`ml-1 rounded-full border px-2 py-0.5 font-semibold ${verdictBadgeClass(rv)}`}>{rv.action}</span>
                                    <span className="ml-1">動能排名 {rv.rank ?? '—'}・近 6 月 {rv.ret_6m_pct != null ? paperPct(rv.ret_6m_pct) : '—'}</span>
                                  </div>
                                  <div>{rv.reason}</div>
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
