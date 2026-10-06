import { useCallback, useEffect, useMemo, useState } from 'react';
import { BellRing, Copy, Flame, Plus, RefreshCw, Trash2 } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  evaluateHoldings,
  fetchStrategyBacktest,
  fetchStrategyReport,
  lookupSymbols,
  type EvaluateResult,
  type LookupResult,
  type HoldingAction,
  type Ignition,
  type StrategyBacktest,
  type StrategyReport,
} from './services/strategy-api';

/* ============================== 共用 ============================== */

type Market = 'us' | 'tw';
type Holding = { ticker: string; cost: number; shares: number };

// 與舊「持股健檢」同一個儲存鍵，既有輸入的持股直接沿用。
const HOLDINGS_KEY = 'real_holdings_v2';
const NOTIFY_KEY = 'strategy_notified_v1';

function loadHoldings(): Holding[] {
  try {
    const raw = localStorage.getItem(HOLDINGS_KEY);
    if (raw) {
      const d = JSON.parse(raw);
      if (Array.isArray(d.holdings)) return d.holdings;
    }
  } catch {
    /* ignore */
  }
  return [];
}

function saveHoldings(holdings: Holding[]): void {
  try {
    localStorage.setItem(HOLDINGS_KEY, JSON.stringify({ holdings, lastReviewed: new Date().toISOString().slice(0, 10) }));
  } catch {
    /* ignore */
  }
}

function parseLines(text: string): Holding[] {
  return text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith('#'))
    .map((l) => {
      const p = l.split(/[\s,，、]+/).filter(Boolean);
      return { ticker: (p[0] || '').toUpperCase(), cost: Number(p[1]) || 0, shares: Number(p[2]) || 0 };
    })
    .filter((h) => h.ticker);
}

const ntd = (v: number | null | undefined) => (v == null ? '—' : `NT$${Math.round(v).toLocaleString()}`);
const pct = (v: number | null | undefined, d = 1) => (v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(d)}%`);
const tone = (v: number | null | undefined) => (v == null ? 'text-slate-500' : v >= 0 ? 'text-emerald-700' : 'text-rose-700');

const ACTION_STYLE: Record<HoldingAction, string> = {
  賣出換股: 'bg-rose-100 text-rose-800 border-rose-200',
  減碼: 'bg-orange-100 text-orange-800 border-orange-200',
  低檔加碼: 'bg-emerald-200 text-emerald-900 border-emerald-300',
  加碼: 'bg-emerald-100 text-emerald-800 border-emerald-200',
  續抱: 'bg-sky-50 text-sky-800 border-sky-200',
  '核心 ETF': 'bg-slate-100 text-slate-700 border-slate-200',
  資料不足: 'bg-slate-50 text-slate-500 border-slate-200',
};

function IgnitionBadge({ ign }: { ign?: Ignition | null }) {
  if (!ign) return null;
  return (
    <span className="ml-1 inline-flex items-center gap-0.5 rounded bg-amber-100 px-1.5 py-0.5 text-[11px] font-medium text-amber-800" title="近 3 日爆量長紅點火">
      <Flame className="h-3 w-3" />
      點火 +{ign.ignition_gain_pct}%
    </span>
  );
}

function MarketToggle({ value, onChange }: { value: Market; onChange: (m: Market) => void }) {
  return (
    <div className="inline-flex rounded-md border border-slate-200 bg-white p-0.5 text-sm">
      {(['us', 'tw'] as Market[]).map((m) => (
        <button
          key={m}
          type="button"
          onClick={() => onChange(m)}
          className={`rounded px-3 py-1.5 ${value === m ? 'bg-slate-950 text-white' : 'text-slate-600 hover:bg-slate-100'}`}
        >
          {m === 'us' ? '美股' : '台股'}
        </button>
      ))}
    </div>
  );
}

function StrategyRuleNote({ report }: { report: StrategyReport | null }) {
  return (
    <div className="text-xs leading-relaxed text-slate-500">
      AI 科技股池，兩條策略：<b>長線低檔布局 {Math.round((report?.strategy.allocation?.lowentry ?? 0.7) * 100)}%</b>（3 年報酬為正的長線贏家，
      自 52 週高點回落 ≥30% 就分批買、持有 12 個月）＋ <b>動能輪動 {Math.round((report?.strategy.allocation?.momentum ?? 0.3) * 100)}%</b>
      （美股前 {report?.strategy.top_n.us ?? 10}／台股前 {report?.strategy.top_n.tw ?? 20} 名）。持股只有在動能轉弱「且」3 年長線趨勢破壞時才建議換股。
      回測見「策略回測」分頁（含倖存者偏差說明）。規則化訊號，非投資建議。
    </div>
  );
}

/* ============================== 策略精選（取代每日掃描三種模式） ============================== */

export function StrategyPicksPanel() {
  const [report, setReport] = useState<StrategyReport | null>(null);
  const [market, setMarket] = useState<Market>('us');
  const [error, setError] = useState('');
  const [query, setQuery] = useState('');

  useEffect(() => {
    fetchStrategyReport().then(setReport).catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  const data = report?.markets[market];
  const rows = data?.rows ?? [];
  const keep = data?.keep_n ?? 20;
  const top = data?.top_n ?? 10;
  const found = useMemo(() => {
    const q = query.trim().toUpperCase();
    if (!q) return null;
    return rows.find((r) => r.symbol.toUpperCase() === q || r.symbol.split('.')[0] === q) ?? 'none';
  }, [query, rows]);

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="text-sm font-semibold text-slate-900">策略精選 · 本月買進名單</div>
            <div className="text-xs text-slate-500">
              {report ? `排名更新：${report.generated_at.replace('T', ' ').slice(0, 16)} ・ 下次月調：${report.strategy.next_rebalance}` : '讀取中…'}
              {report?.strategy.in_rebalance_window ? ' ・ 🔔 本週為月調窗口' : ''}
            </div>
          </div>
          <MarketToggle value={market} onChange={setMarket} />
        </div>
        <div className="mt-2">
          <StrategyRuleNote report={report} />
        </div>
        {error ? <p className="mt-2 text-sm text-rose-600">{error}</p> : null}
      </div>

      {data ? (
        <>
          <LowEntrySection rows={rows} low={data.low_entry ?? []} watch={data.low_entry_watch ?? []} />
          <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <div className="mb-2 text-sm font-semibold text-slate-900">
              動能輪動（{Math.round((report?.strategy.allocation?.momentum ?? 0.3) * 100)}% 資金）· 前 {top} 名（買進區）<span className="ml-2 text-xs font-normal text-slate-500">共排名 {data.universe_size} 檔</span>
            </div>
            <RankTable rows={rows.slice(0, top)} />
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <div className="text-sm font-semibold text-slate-900">
                查排名<span className="ml-2 text-xs font-normal text-slate-500">第 {top + 1}~{keep} 名為續抱區（已持有可續抱、不新買）；{keep} 名之後為賣出區</span>
              </div>
              <Input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="輸入代號，例：NVDA / 2330" className="h-9 w-56" />
            </div>
            {found === 'none' ? <p className="text-sm text-slate-500">不在股票池內（可到「我的持股」輸入，系統會即時計算等效排名）。</p> : null}
            {found && found !== 'none' ? <RankTable rows={[found]} /> : null}
            {!found ? <RankTable rows={rows.slice(top, keep)} /> : null}
          </div>
        </>
      ) : null}
    </div>
  );
}

function LowEntrySection({ rows, low, watch }: { rows: StrategyReport['markets']['us']['rows']; low: string[]; watch: string[] }) {
  const by = Object.fromEntries(rows.map((r) => [r.symbol, r]));
  const list = (syms: string[]) => syms.map((s) => by[s]).filter(Boolean);
  const Row = ({ r }: { r: (typeof rows)[number] }) => (
    <tr className="border-b border-slate-100">
      <td className="py-1.5 pr-2">
        <span className="font-medium text-slate-900">{r.symbol.replace(/\.TWO?$/, '')}</span>
        {r.name ? <span className="ml-1 text-xs text-slate-500">{r.name}</span> : null}
      </td>
      <td className="py-1.5 pr-2 text-right">{r.close}</td>
      <td className="py-1.5 pr-2 text-right text-rose-700">{pct(r.dd_52w_pct ?? null, 0)}</td>
      <td className="py-1.5 pr-2 text-right">{r.high_52w}</td>
      <td className={`py-1.5 pr-2 text-right ${tone(r.ret_3y_pct ?? null)}`}>{pct(r.ret_3y_pct ?? null, 0)}</td>
      <td className="py-1.5 text-right text-slate-600">{r.low_entry_price ?? '—'}</td>
    </tr>
  );
  const head = (
    <thead>
      <tr className="border-b border-slate-200 text-left text-xs text-slate-500">
        <th className="py-1.5 pr-2">標的</th>
        <th className="py-1.5 pr-2 text-right">現價</th>
        <th className="py-1.5 pr-2 text-right">距 52 週高</th>
        <th className="py-1.5 pr-2 text-right">52 週高</th>
        <th className="py-1.5 pr-2 text-right">3 年報酬</th>
        <th className="py-1.5 text-right">低檔價（−30%）</th>
      </tr>
    </thead>
  );
  return (
    <div className="rounded-lg border border-emerald-200 bg-white p-4 shadow-sm">
      <div className="text-sm font-semibold text-slate-900">長線低檔布局（主力資金）· 現在可分批買進</div>
      <div className="mb-2 text-xs text-slate-500">長線贏家（3 年報酬為正）自 52 週高點回落 ≥30%：回測勝率 81%（美股）。每檔約一成低檔資金，持有 12 個月。</div>
      {list(low).length ? (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[520px] text-sm">
            {head}
            <tbody>{list(low).map((r) => <Row key={r.symbol} r={r} />)}</tbody>
          </table>
        </div>
      ) : (
        <p className="text-sm text-slate-500">目前沒有股票落到低檔區（AI 族群普遍在高檔）。耐心等待正是低檔布局的一部分。</p>
      )}
      {list(watch).length ? (
        <>
          <div className="mb-1 mt-3 text-xs font-semibold text-slate-700">觀察名單（已回落 20~30%，接近低檔區）</div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[520px] text-sm">
              {head}
              <tbody>{list(watch).map((r) => <Row key={r.symbol} r={r} />)}</tbody>
            </table>
          </div>
        </>
      ) : null}
    </div>
  );
}

function RankTable({ rows }: { rows: StrategyReport['markets']['us']['rows'] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[560px] text-sm">
        <thead>
          <tr className="border-b border-slate-200 text-left text-xs text-slate-500">
            <th className="py-1.5 pr-2">排名</th>
            <th className="py-1.5 pr-2">標的</th>
            <th className="py-1.5 pr-2 text-right">現價</th>
            <th className="py-1.5 pr-2 text-right">近 6 月</th>
            <th className="py-1.5 pr-2 text-right">近 1 月</th>
            <th className="py-1.5 pr-2 text-right">年化波動</th>
            <th className="py-1.5 pr-2 text-right">動能分數</th>
            <th className="py-1.5">區間</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.symbol} className="border-b border-slate-100">
              <td className="py-1.5 pr-2 font-semibold text-slate-900">{r.rank}</td>
              <td className="py-1.5 pr-2">
                <span className="font-medium text-slate-900">{r.symbol.replace(/\.TWO?$/, '')}</span>
                {r.name ? <span className="ml-1 text-xs text-slate-500">{r.name}</span> : null}
                <IgnitionBadge ign={r.ignition} />
              </td>
              <td className="py-1.5 pr-2 text-right">{r.close}</td>
              <td className={`py-1.5 pr-2 text-right ${tone(r.ret_6m_pct)}`}>{pct(r.ret_6m_pct, 0)}</td>
              <td className={`py-1.5 pr-2 text-right ${tone(r.ret_1m_pct)}`}>{pct(r.ret_1m_pct, 0)}</td>
              <td className="py-1.5 pr-2 text-right text-slate-600">{r.vol_ann_pct.toFixed(0)}%</td>
              <td className="py-1.5 pr-2 text-right text-slate-700">{r.score.toFixed(2)}</td>
              <td className="py-1.5">
                <span
                  className={`rounded px-1.5 py-0.5 text-xs ${
                    r.zone === 'buy' ? 'bg-emerald-100 text-emerald-800' : r.zone === 'hold' ? 'bg-sky-50 text-sky-800' : 'bg-rose-50 text-rose-700'
                  }`}
                >
                  {r.zone === 'buy' ? '買進區' : r.zone === 'hold' ? '續抱區' : '賣出區'}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ============================== 我的持股（輸入 + 策略提醒） ============================== */

export function MyHoldingsPanel() {
  const [holdings, setHoldings] = useState<Holding[]>(() => loadHoldings());
  const [result, setResult] = useState<EvaluateResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [paste, setPaste] = useState('');
  const [showPaste, setShowPaste] = useState(false);
  const [copied, setCopied] = useState(false);

  const update = (next: Holding[]) => {
    setHoldings(next);
    saveHoldings(next);
  };

  const evaluate = useCallback(async (list: Holding[]) => {
    const valid = list.filter((h) => h.ticker && h.shares > 0);
    if (!valid.length) return;
    setLoading(true);
    setError('');
    try {
      const r = await evaluateHoldings(valid);
      setResult(r);
      notifyIfNeeded(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  // 開頁即自動評估（主動提醒）
  useEffect(() => {
    void evaluate(holdings);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const actionable = (result?.holdings ?? []).filter((h) => ['賣出換股', '減碼', '低檔加碼', '加碼'].includes(h.action));
  const ignited = (result?.holdings ?? []).filter((h) => h.ignition && ['續抱', '加碼'].includes(h.action));
  const exportText = holdings
    .filter((h) => h.ticker && h.shares > 0)
    .map((h) => `${h.ticker} ${h.cost} ${h.shares}`)
    .join('\n');

  return (
    <div className="space-y-4">
      {/* 主動提醒橫幅 */}
      {result && (actionable.length || ignited.length) ? (
        <div className={`rounded-lg border p-4 shadow-sm ${result.in_rebalance_window ? 'border-rose-300 bg-rose-50' : 'border-amber-300 bg-amber-50'}`}>
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
            <BellRing className="h-4 w-4" />
            {result.in_rebalance_window
              ? `本月調整時間到：${actionable.length} 檔持股需要動作`
              : `下次月調（${result.next_rebalance}）預計：${actionable.length} 檔持股需要動作`}
          </div>
          <div className="mt-2 flex flex-wrap gap-1.5 text-xs">
            {actionable.map((h) => (
              <span key={h.symbol} className={`rounded border px-2 py-0.5 ${ACTION_STYLE[h.action]}`}>
                {h.action} {h.symbol.replace(/\.TWO?$/, '')}
              </span>
            ))}
            {ignited.map((h) => (
              <span key={`ign-${h.symbol}`} className="rounded border border-amber-300 bg-amber-100 px-2 py-0.5 text-amber-900">
                🔥 點火可加碼 {h.symbol.replace(/\.TWO?$/, '')}
              </span>
            ))}
          </div>
          {!result.in_rebalance_window ? (
            <div className="mt-2 text-xs text-slate-600">策略每月初才執行換股（與回測一致）；月中只有「點火」事件值得提前加碼。</div>
          ) : null}
        </div>
      ) : null}

      {/* 持股輸入 */}
      <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div>
            <div className="text-sm font-semibold text-slate-900">我的持股</div>
            <div className="text-xs text-slate-500">台股直接打代號（2330、00878）；美股打代號（NVDA）。資料只存在這台瀏覽器。</div>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" className="h-9" type="button" onClick={() => setShowPaste((v) => !v)}>
              貼上匯入
            </Button>
            <Button variant="outline" className="h-9" type="button" onClick={() => update([...holdings, { ticker: '', cost: 0, shares: 0 }])}>
              <Plus className="mr-1 h-4 w-4" />
              新增一列
            </Button>
            <Button className="h-9 bg-slate-950 text-white hover:bg-slate-800" type="button" disabled={loading} onClick={() => evaluate(holdings)}>
              <RefreshCw className={`mr-1 h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
              {loading ? '評估中' : '依策略評估'}
            </Button>
          </div>
        </div>

        {showPaste ? (
          <div className="mb-3 space-y-2">
            <textarea
              value={paste}
              onChange={(e) => setPaste(e.target.value)}
              rows={6}
              placeholder={'每行「代號 成本 股數」，例如：\nNVDA 175 53\n2330 600 1000'}
              className="w-full rounded-md border border-slate-200 p-2 font-mono text-sm"
            />
            <div className="flex gap-2">
              <Button
                className="h-8"
                type="button"
                onClick={() => {
                  const next = parseLines(paste);
                  update(next);
                  setShowPaste(false);
                  void evaluate(next);
                }}
              >
                取代目前持股
              </Button>
              <Button variant="outline" className="h-8" type="button" onClick={() => setShowPaste(false)}>
                取消
              </Button>
            </div>
          </div>
        ) : null}

        <div className="overflow-x-auto">
          <table className="w-full min-w-[420px] text-sm">
            <thead>
              <tr className="border-b border-slate-200 text-left text-xs text-slate-500">
                <th className="py-1.5 pr-2">代號</th>
                <th className="py-1.5 pr-2">平均成本</th>
                <th className="py-1.5 pr-2">股數</th>
                <th className="py-1.5" />
              </tr>
            </thead>
            <tbody>
              {holdings.map((h, i) => (
                <tr key={i} className="border-b border-slate-100">
                  <td className="py-1 pr-2">
                    <Input
                      value={h.ticker}
                      onChange={(e) => update(holdings.map((x, j) => (j === i ? { ...x, ticker: e.target.value.toUpperCase().trim() } : x)))}
                      className="h-8 w-28"
                    />
                  </td>
                  <td className="py-1 pr-2">
                    <Input
                      type="number"
                      value={h.cost || ''}
                      onChange={(e) => update(holdings.map((x, j) => (j === i ? { ...x, cost: Number(e.target.value) || 0 } : x)))}
                      className="h-8 w-28"
                    />
                  </td>
                  <td className="py-1 pr-2">
                    <Input
                      type="number"
                      value={h.shares || ''}
                      onChange={(e) => update(holdings.map((x, j) => (j === i ? { ...x, shares: Number(e.target.value) || 0 } : x)))}
                      className="h-8 w-28"
                    />
                  </td>
                  <td className="py-1">
                    <button type="button" className="text-slate-400 hover:text-rose-600" onClick={() => update(holdings.filter((_, j) => j !== i))} title="刪除">
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {!holdings.length ? <p className="py-3 text-sm text-slate-500">尚未輸入持股。按「新增一列」或「貼上匯入」。</p> : null}
        </div>
        {error ? <p className="mt-2 text-sm text-rose-600">{error}</p> : null}
      </div>

      {/* 評估結果 */}
      {result ? (
        <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
          <div className="mb-3 flex flex-wrap gap-x-6 gap-y-1 text-sm">
            <span>
              總市值 <b>{ntd(result.total_twd)}</b>
            </span>
            <span className="text-slate-600">
              美股個股 {ntd(result.sleeve_twd.us)}（每檔目標 {ntd(result.target_per_name_twd.us)}）
            </span>
            <span className="text-slate-600">
              台股個股 {ntd(result.sleeve_twd.tw)}（每檔目標 {ntd(result.target_per_name_twd.tw)}）
            </span>
            <span className="text-slate-500">匯率 {result.fx_usd_twd}</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[720px] text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-left text-xs text-slate-500">
                  <th className="py-1.5 pr-2">動作</th>
                  <th className="py-1.5 pr-2">標的</th>
                  <th className="py-1.5 pr-2 text-right">排名</th>
                  <th className="py-1.5 pr-2 text-right">現價</th>
                  <th className="py-1.5 pr-2 text-right">損益</th>
                  <th className="py-1.5 pr-2 text-right">佔比</th>
                  <th className="py-1.5">理由</th>
                </tr>
              </thead>
              <tbody>
                {result.holdings.map((h) => (
                  <tr key={h.symbol} className="border-b border-slate-100 align-top">
                    <td className="py-1.5 pr-2">
                      <span className={`whitespace-nowrap rounded border px-1.5 py-0.5 text-xs font-medium ${ACTION_STYLE[h.action]}`}>{h.action}</span>
                    </td>
                    <td className="py-1.5 pr-2 font-medium text-slate-900">
                      {h.symbol.replace(/\.TWO?$/, '')}
                      <IgnitionBadge ign={h.ignition} />
                    </td>
                    <td className="py-1.5 pr-2 text-right text-slate-700">{h.rank ?? '—'}</td>
                    <td className="py-1.5 pr-2 text-right">{h.price ?? '—'}</td>
                    <td className={`py-1.5 pr-2 text-right ${tone(h.pnl_pct)}`}>{pct(h.pnl_pct)}</td>
                    <td className="py-1.5 pr-2 text-right text-slate-600">{h.weight_pct.toFixed(1)}%</td>
                    <td className="py-1.5 text-xs leading-relaxed text-slate-600">{h.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {(['us', 'tw'] as Market[]).map((m) =>
            result.low_entry_buys?.[m]?.length ? (
              <div key={`low-${m}`} className="mt-4">
                <div className="mb-1 text-sm font-semibold text-slate-900">
                  {m === 'us' ? '美股' : '台股'}低檔布局可買（你還沒有的）
                  <span className="ml-2 text-xs font-normal text-slate-500">每檔目標約 {ntd(result.low_entry_target_twd?.[m])}，持有 12 個月</span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {result.low_entry_buys[m].map((b) => (
                    <span key={b.symbol} className="rounded border border-emerald-300 bg-emerald-50 px-2 py-1 text-xs text-emerald-900">
                      {b.symbol.replace(/\.TWO?$/, '')}{b.name ? ` ${b.name}` : ''} · {b.close} · 距高點 {pct(b.dd_52w_pct, 0)} · 3 年 {pct(b.ret_3y_pct, 0)}
                    </span>
                  ))}
                </div>
              </div>
            ) : null,
          )}
          {(['us', 'tw'] as Market[]).map((m) =>
            result.new_buys[m]?.length ? (
              <div key={m} className="mt-4">
                <div className="mb-1 text-sm font-semibold text-slate-900">
                  {m === 'us' ? '美股' : '台股'}動能新買進（買進區中你還沒有的）
                  <span className="ml-2 text-xs font-normal text-slate-500">每檔目標約 {ntd(result.target_per_name_twd[m])}</span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {result.new_buys[m].map((b) => (
                    <span key={b.symbol} className="rounded border border-emerald-200 bg-emerald-50 px-2 py-1 text-xs text-emerald-900">
                      #{b.rank} {b.symbol.replace(/\.TWO?$/, '')}
                      {b.name ? ` ${b.name}` : ''} · {b.close} · 6 月 {pct(b.ret_6m_pct, 0)}
                      <IgnitionBadge ign={b.ignition} />
                    </span>
                  ))}
                </div>
              </div>
            ) : null,
          )}
        </div>
      ) : null}

      {/* 自動推播設定 */}
      <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <div className="text-sm font-semibold text-slate-900">關掉網頁也收得到提醒（手機推播）</div>
            <div className="text-xs text-slate-500">
              開著這頁時會跳瀏覽器通知；要在月調日／點火時推到手機，請把持股存進 GitHub Secret「HOLDINGS」一次（持股變動時再更新）。
            </div>
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              className="h-9"
              type="button"
              onClick={() => {
                if ('Notification' in window) void Notification.requestPermission();
              }}
            >
              <BellRing className="mr-1 h-4 w-4" />
              開啟瀏覽器通知
            </Button>
            <Button
              className="h-9"
              type="button"
              disabled={!exportText}
              onClick={async () => {
                try {
                  await navigator.clipboard.writeText(exportText);
                  setCopied(true);
                  setTimeout(() => setCopied(false), 2500);
                } catch {
                  setShowPaste(true);
                  setPaste(exportText);
                }
              }}
            >
              <Copy className="mr-1 h-4 w-4" />
              {copied ? '已複製' : '複製 HOLDINGS 內容'}
            </Button>
          </div>
        </div>
        <ol className="mt-2 list-decimal space-y-0.5 pl-5 text-xs text-slate-600">
          <li>GitHub repo → Settings → Secrets and variables → Actions → New repository secret，名稱 HOLDINGS，貼上剛複製的內容。</li>
          <li>再新增一個推播管道 Secret：最簡單是手機裝 ntfy App、訂閱一個難猜的主題名，Secret 名稱 NTFY_TOPIC 填同名（也支援 Telegram／Discord／Email）。</li>
          <li>Actions → Trade Alerts → Run workflow 測試一次。之後每月初與盤中點火時自動推播。</li>
        </ol>
      </div>
    </div>
  );
}

function notifyIfNeeded(r: EvaluateResult): void {
  const items = r.holdings.filter((h) => ['賣出換股', '減碼', '低檔加碼', '加碼'].includes(h.action) || (h.ignition && h.action === '續抱'));
  if (!items.length || !('Notification' in window) || Notification.permission !== 'granted') return;
  // 只在「月調窗口」或「有點火」時跳通知，且同一組內容一天只跳一次。
  const ignited = items.filter((h) => h.ignition || h.action === '低檔加碼');
  if (!r.in_rebalance_window && !ignited.length) return;
  const sig = `${new Date().toISOString().slice(0, 10)}|${items.map((h) => h.symbol + h.action).join(',')}`;
  try {
    if (localStorage.getItem(NOTIFY_KEY) === sig) return;
    localStorage.setItem(NOTIFY_KEY, sig);
  } catch {
    /* ignore */
  }
  const list = (r.in_rebalance_window ? items : ignited).map((h) => `${h.ignition && !r.in_rebalance_window ? '點火加碼' : h.action} ${h.symbol.replace(/\.TWO?$/, '')}`);
  new Notification('策略提醒：持股需要動作', { body: list.slice(0, 8).join('、') + (list.length > 8 ? '…' : '') });
}

/* ============================== 策略回測（對標 VOO / 0050） ============================== */

export function StrategyBacktestPanel() {
  const [data, setData] = useState<{ generated_at: string; markets: Record<Market, StrategyBacktest> } | null>(null);
  const [market, setMarket] = useState<Market>('us');
  const [error, setError] = useState('');

  useEffect(() => {
    fetchStrategyBacktest().then(setData).catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  const bt = data?.markets[market];
  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="text-sm font-semibold text-slate-900">策略回測 · 低檔布局 70% ＋ 動能 30% vs {market === 'us' ? 'VOO / QQQ' : '0050'}</div>
            <div className="text-xs text-slate-500">
              {bt ? `${bt.start_date} ~ ${bt.end_date} ・ 股票池 ${bt.universe_size} 檔 ・ 每年約換股 ${bt.trades_per_year} 次 ・ 每次成本 ${(bt.rules.cost_per_trade * 100).toFixed(1)}%` : '讀取中…'}
            </div>
          </div>
          <MarketToggle value={market} onChange={setMarket} />
        </div>
        {error ? <p className="mt-2 text-sm text-rose-600">{error}</p> : null}
      </div>

      {bt ? (
        <>
          <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[560px] text-sm">
                <thead>
                  <tr className="border-b border-slate-200 text-left text-xs text-slate-500">
                    <th className="py-1.5 pr-2">期間</th>
                    <th className="py-1.5 pr-2 text-right">策略年化</th>
                    <th className="py-1.5 pr-2 text-right">{bt.benchmark_symbol} 年化</th>
                    <th className="py-1.5 pr-2 text-right">策略最大回撤</th>
                    <th className="py-1.5 pr-2 text-right">{bt.benchmark_symbol} 最大回撤</th>
                    <th className="py-1.5 pr-2 text-right">策略 Sharpe</th>
                    <th className="py-1.5 text-right">{bt.benchmark_symbol} Sharpe</th>
                    {Object.keys(Object.values(bt.periods)[0]?.compare ?? {}).map((name) => (
                      <th key={name} className="py-1.5 pl-2 text-right">{name} 年化</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(bt.periods).map(([label, p]) => (
                    <tr key={label} className="border-b border-slate-100">
                      <td className="py-1.5 pr-2 font-medium">{label}</td>
                      <td className="py-1.5 pr-2 text-right font-semibold text-emerald-700">{p.strategy.cagr_pct}%</td>
                      <td className="py-1.5 pr-2 text-right text-slate-600">{p.benchmark.cagr_pct}%</td>
                      <td className="py-1.5 pr-2 text-right text-rose-700">{p.strategy.max_drawdown_pct}%</td>
                      <td className="py-1.5 pr-2 text-right text-slate-600">{p.benchmark.max_drawdown_pct}%</td>
                      <td className="py-1.5 pr-2 text-right">{p.strategy.sharpe}</td>
                      <td className="py-1.5 text-right text-slate-600">{p.benchmark.sharpe}</td>
                      {Object.entries(p.compare ?? {}).map(([name, c]) => (
                        <td key={name} className="py-1.5 pl-2 text-right text-slate-600">{c.cagr_pct}%</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {bt.tracks ? (
              <div className="mt-3 overflow-x-auto">
                <table className="w-full min-w-[560px] text-sm">
                  <thead>
                    <tr className="border-b border-slate-200 text-left text-xs text-slate-500">
                      <th className="py-1.5 pr-2">策略</th>
                      {Object.keys(bt.periods).map((k) => (
                        <th key={k} className="py-1.5 pr-2 text-right">{k} 年化／回撤</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-slate-100">
                      <td className="py-1.5 pr-2">動能輪動</td>
                      {Object.values(bt.periods).map((p, i) => (
                        <td key={i} className="py-1.5 pr-2 text-right">{p.strategy.cagr_pct}%／{p.strategy.max_drawdown_pct}%</td>
                      ))}
                    </tr>
                    {Object.entries(bt.tracks).map(([k, t]) => (
                      <tr key={k} className={`border-b border-slate-100 ${k === 'combo' ? 'font-semibold text-emerald-800' : ''}`}>
                        <td className="py-1.5 pr-2">
                          {t.name}
                          {t.win_rate_pct != null ? <span className="ml-1 text-xs font-normal text-slate-500">（勝率 {t.win_rate_pct}%、{t.trades} 筆）</span> : null}
                        </td>
                        {Object.keys(bt.periods).map((lab) => (
                          <td key={lab} className="py-1.5 pr-2 text-right">
                            {t.periods[lab] ? `${t.periods[lab].strategy.cagr_pct}%／${t.periods[lab].strategy.max_drawdown_pct}%` : '—'}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
            <BacktestCurve points={bt.equity_curve} benchmark={bt.benchmark_symbol} />
            <p className="mt-2 text-xs text-slate-500">
              「同池等權持有」＝把整個 AI 科技股池平均買進不動。策略要贏過它，才代表排名選股真的有加分，而不只是吃到 AI 族群本身的漲幅。
            </p>
            <p className="mt-2 text-xs text-amber-700">⚠️ {bt.caveat}</p>
          </div>
          <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <div className="mb-1 text-sm font-semibold text-slate-900">近期換股紀錄（回測）</div>
            <ul className="space-y-1 text-xs text-slate-600">
              {bt.recent_rebalances
                .slice()
                .reverse()
                .map((r) => (
                  <li key={r.date}>
                    <b>{r.date}</b>　買：{r.buy.join('、') || '—'}　賣：{r.sell.join('、') || '—'}
                  </li>
                ))}
            </ul>
          </div>
        </>
      ) : null}
    </div>
  );
}

function BacktestCurve({ points, benchmark }: { points: StrategyBacktest['equity_curve']; benchmark: string }) {
  if (points.length < 2) return null;
  const W = 720;
  const H = 220;
  const pad = 8;
  const logs = points.flatMap((p) => [Math.log(p.strategy), Math.log(p.benchmark)]);
  const lo = Math.min(...logs);
  const hi = Math.max(...logs);
  const x = (i: number) => pad + (i / (points.length - 1)) * (W - 2 * pad);
  const y = (v: number) => H - pad - ((Math.log(v) - lo) / (hi - lo || 1)) * (H - 2 * pad);
  const path = (key: 'strategy' | 'benchmark') => points.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p[key]).toFixed(1)}`).join(' ');
  const last = points[points.length - 1];
  return (
    <div className="mt-3">
      <svg viewBox={`0 0 ${W} ${H}`} className="h-56 w-full" preserveAspectRatio="none" role="img" aria-label="策略與大盤淨值曲線（對數）">
        <path d={path('benchmark')} fill="none" stroke="#94a3b8" strokeWidth="1.5" />
        <path d={path('strategy')} fill="none" stroke="#047857" strokeWidth="2" />
      </svg>
      <div className="flex gap-4 text-xs text-slate-600">
        <span>
          <span className="mr-1 inline-block h-2 w-3 bg-emerald-700" />
          策略 ×{last.strategy.toFixed(1)}
        </span>
        <span>
          <span className="mr-1 inline-block h-2 w-3 bg-slate-400" />
          {benchmark} ×{last.benchmark.toFixed(1)}
        </span>
        <span className="text-slate-400">（對數刻度）</span>
      </div>
    </div>
  );
}

/* ============================== 個股分析（以策略排名判斷） ============================== */

const VERDICT_STYLE: Record<string, string> = {
  可買進: 'bg-emerald-100 text-emerald-800 border-emerald-200',
  '持有續抱，不新買': 'bg-sky-50 text-sky-800 border-sky-200',
  '不買／月初換股': 'bg-rose-100 text-rose-800 border-rose-200',
  低檔布局可買: 'bg-emerald-200 text-emerald-900 border-emerald-300',
  等低檔: 'bg-amber-50 text-amber-800 border-amber-200',
  '核心 ETF': 'bg-slate-100 text-slate-700 border-slate-200',
  資料不足: 'bg-slate-50 text-slate-500 border-slate-200',
};

export function StockLookupPanel() {
  const [input, setInput] = useState('');
  const [results, setResults] = useState<LookupResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const run = async () => {
    const symbols = input.split(/[\s,，、]+/).map((s) => s.trim().toUpperCase()).filter(Boolean);
    if (!symbols.length) {
      setError('請先輸入股票代號。');
      return;
    }
    setLoading(true);
    setError('');
    try {
      setResults(await lookupSymbols(symbols.slice(0, 10)));
    } catch (e) {
      setResults([]);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <form
        className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm"
        onSubmit={(e) => {
          e.preventDefault();
          void run();
        }}
      >
        <div className="flex flex-wrap items-end gap-2">
          <label className="block min-w-[240px] flex-1">
            <span className="mb-1 block text-xs font-semibold text-slate-600">查個股在策略中的排名（最多 10 檔，逗號或空白分隔）</span>
            <Input value={input} onChange={(e) => setInput(e.target.value)} placeholder="NVDA, CEG, 2330" className="h-10" />
          </label>
          <Button className="h-10 bg-slate-950 text-white hover:bg-slate-800" disabled={loading} type="submit">
            {loading ? '查詢中' : '查詢'}
          </Button>
        </div>
        <div className="mt-2 text-xs text-slate-500">
          兩條規則：①長線贏家回落 ≥30% → 低檔布局可買（主力）；②動能排名前段 → 動能可買。都不符合時顯示「等低檔」與低檔價。
        </div>
        {error ? <p className="mt-2 text-sm text-rose-600">{error}</p> : null}
      </form>

      {results.map((r) => (
        <div key={r.symbol} className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-base font-semibold text-slate-900">
              {r.symbol.replace(/\.TWO?$/, '')}
              {r.row?.name ? <span className="ml-1 text-sm font-normal text-slate-500">{r.row.name}</span> : null}
            </span>
            <span className={`rounded border px-2 py-0.5 text-xs font-medium ${VERDICT_STYLE[r.verdict] ?? VERDICT_STYLE['資料不足']}`}>{r.verdict}</span>
            <IgnitionBadge ign={r.row?.ignition} />
          </div>
          <p className="mt-1 text-sm text-slate-600">{r.detail}</p>
          {r.row && r.verdict !== '核心 ETF' ? (
            <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-sm">
              <span>
                排名 <b>{r.row.rank}</b> / {r.universe_size}
                {r.row.outside_universe ? <span className="text-xs text-slate-400">（不在股票池，等效排名）</span> : null}
              </span>
              <span>現價 {r.row.close}</span>
              <span className={tone(r.row.ret_6m_pct)}>近 6 月 {pct(r.row.ret_6m_pct, 0)}</span>
              <span className={tone(r.row.ret_1m_pct)}>近 1 月 {pct(r.row.ret_1m_pct, 0)}</span>
              <span className="text-slate-600">年化波動 {r.row.vol_ann_pct.toFixed(0)}%</span>
              <span className="text-slate-600">動能分數 {r.row.score.toFixed(2)}</span>
              {r.row.dd_52w_pct != null ? <span className="text-rose-700">距 52 週高 {pct(r.row.dd_52w_pct, 0)}</span> : null}
              {r.row.ret_3y_pct != null ? <span className={tone(r.row.ret_3y_pct)}>3 年 {pct(r.row.ret_3y_pct, 0)}</span> : null}
              {r.row.low_entry_price ? <span className="text-slate-600">低檔價 {r.row.low_entry_price}</span> : null}
            </div>
          ) : null}
          <PriceLine closes={r.closes} />
        </div>
      ))}
    </div>
  );
}

function PriceLine({ closes }: { closes: LookupResult['closes'] }) {
  if (closes.length < 2) return null;
  const W = 720;
  const H = 140;
  const vals = closes.map((c) => c.close);
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const x = (i: number) => (i / (closes.length - 1)) * W;
  const y = (v: number) => H - 4 - ((v - lo) / (hi - lo || 1)) * (H - 8);
  const d = closes.map((c, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(c.close).toFixed(1)}`).join(' ');
  const up = vals[vals.length - 1] >= vals[0];
  return (
    <div className="mt-3">
      <svg viewBox={`0 0 ${W} ${H}`} className="h-32 w-full" preserveAspectRatio="none" role="img" aria-label="近一年收盤價">
        <path d={d} fill="none" stroke={up ? '#047857' : '#be123c'} strokeWidth="1.8" />
      </svg>
      <div className="flex justify-between text-[11px] text-slate-400">
        <span>{closes[0].date}</span>
        <span>近一年收盤</span>
        <span>{closes[closes.length - 1].date}</span>
      </div>
    </div>
  );
}
