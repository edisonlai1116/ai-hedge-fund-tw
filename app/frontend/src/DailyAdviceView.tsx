import { useEffect, useMemo, useState } from 'react';
import { Activity, CalendarCheck, RefreshCw } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { fetchDailyAdvice, type AdviceItem, type DailyAdvice, type MarketAdvice, type SectorMood } from './services/strategy-api';
import { HoldingsTodayCard, ViratttBadge } from './StrategyViews';

const LEVEL_STYLE: Record<MarketAdvice['level'], string> = {
  action: 'bg-emerald-600 text-white',
  watch: 'bg-amber-100 text-amber-900',
  hold: 'bg-slate-100 text-slate-700',
};
const LEVEL_LABEL: Record<MarketAdvice['level'], string> = { action: '要操作', watch: '觀察', hold: '不用動' };
const MOOD_STYLE: Record<SectorMood['mood'], string> = {
  過熱: 'bg-rose-100 text-rose-800',
  強勢: 'bg-emerald-100 text-emerald-800',
  中性: 'bg-slate-100 text-slate-600',
  弱勢: 'bg-orange-100 text-orange-800',
  超賣: 'bg-sky-100 text-sky-800',
};
const MARKET_LABEL = { us: '美股', tw: '台股' } as const;

const pct = (v: number | null | undefined, d = 1) => (v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(d)}%`);
const tone = (v: number | null | undefined) => (v == null ? 'text-slate-500' : v >= 0 ? 'text-emerald-700' : 'text-rose-700');

function fgColor(v: number | null | undefined): string {
  if (v == null) return 'text-slate-500';
  if (v < 25) return 'text-rose-700';
  if (v < 45) return 'text-orange-600';
  if (v <= 55) return 'text-slate-700';
  if (v <= 75) return 'text-emerald-600';
  return 'text-emerald-800';
}

function Gauge({ value }: { value: number | null | undefined }) {
  const v = value == null ? null : Math.max(0, Math.min(100, value));
  return (
    <div className="relative mt-2 h-2.5 w-full rounded-full bg-gradient-to-r from-rose-500 via-slate-200 to-emerald-500">
      {v != null ? (
        <span
          className="absolute -top-1 h-4.5 w-1.5 -translate-x-1/2 rounded-sm border border-white bg-slate-900 shadow"
          style={{ left: `${v}%`, height: 18 }}
        />
      ) : null}
      <div className="mt-3 flex justify-between text-[10px] text-slate-400">
        <span>極度恐懼</span>
        <span>中性</span>
        <span>極度貪婪</span>
      </div>
    </div>
  );
}

function FgSpark({ points }: { points: { date: string; fg: number }[] }) {
  if (points.length < 10) return null;
  const w = 320;
  const h = 64;
  const xs = (i: number) => (i / (points.length - 1)) * w;
  const ys = (v: number) => h - (v / 100) * h;
  const d = points.map((p, i) => `${i ? 'L' : 'M'}${xs(i).toFixed(1)},${ys(p.fg).toFixed(1)}`).join(' ');
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="mt-3 h-16 w-full" preserveAspectRatio="none" role="img" aria-label="近一年恐懼貪婪走勢">
      <rect x="0" y={ys(100)} width={w} height={ys(75) - ys(100)} fill="#10b98114" />
      <rect x="0" y={ys(25)} width={w} height={ys(0) - ys(25)} fill="#f43f5e14" />
      <line x1="0" x2={w} y1={ys(50)} y2={ys(50)} stroke="#cbd5e1" strokeDasharray="3 3" />
      <path d={d} fill="none" stroke="#0f172a" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function ItemTable({ items }: { items: AdviceItem[] }) {
  if (!items.length) return null;
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[560px] text-sm">
        <thead>
          <tr className="border-b border-slate-200 text-left text-xs text-slate-500">
            <th className="py-1.5 pr-2">代號</th>
            <th className="py-1.5 pr-2">建議</th>
            <th className="py-1.5 pr-2 text-right">收盤</th>
            <th className="py-1.5 pr-2 text-right">距高點</th>
            <th className="py-1.5 pr-2">ai-hedge 技術</th>
            <th className="py-1.5 pr-2">理由</th>
          </tr>
        </thead>
        <tbody>
          {items.map((a, i) => (
            <tr key={`${a.symbol}-${a.type}-${i}`} className="border-b border-slate-100 align-top">
              <td className="py-1.5 pr-2 font-medium text-slate-900">
                {a.symbol}
                {a.name ? <span className="ml-1 text-xs text-slate-500">{a.name}</span> : null}
              </td>
              <td className="whitespace-nowrap py-1.5 pr-2 text-slate-800">{a.action}</td>
              <td className="py-1.5 pr-2 text-right tabular-nums">{a.close ?? '—'}</td>
              <td className={`py-1.5 pr-2 text-right tabular-nums ${tone(a.dd_52w_pct)}`}>{a.dd_52w_pct != null ? `${a.dd_52w_pct}%` : '—'}</td>
              <td className="py-1.5 pr-2">{a.symbol !== '—' ? <ViratttBadge v={a.virattt} /> : null}</td>
              <td className="py-1.5 pr-2 text-xs text-slate-600">{a.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ViratttSection({ data }: { data: NonNullable<MarketAdvice['virattt']> }) {
  const [open, setOpen] = useState(false);
  const list = (title: string, rows: NonNullable<MarketAdvice['virattt']>['bullish']) => (
    <div>
      <div className="mb-1 text-xs font-medium text-slate-700">{title}</div>
      <div className="flex flex-wrap gap-1.5">
        {rows.map((r) => (
          <span
            key={r.symbol}
            title={`趨勢 ${r.virattt.trend ?? '—'}・動能 ${r.virattt.momentum ?? '—'}・均值回歸 ${r.virattt.mean_reversion ?? '—'}・波動 ${r.virattt.volatility ?? '—'}・統計套利 ${r.virattt.stat_arb ?? '—'}`}
            className="inline-flex items-center gap-1 rounded border border-slate-200 px-1.5 py-0.5 text-xs"
          >
            {r.symbol.replace(/\.TWO?$/, '')}
            {r.name ? <span className="text-slate-400">{r.name}</span> : null}
            <ViratttBadge v={r.virattt} />
          </span>
        ))}
      </div>
    </div>
  );
  return (
    <div className="mt-3 rounded-md border border-slate-200 bg-slate-50 p-2">
      <button type="button" onClick={() => setOpen((o) => !o)} className="flex w-full flex-wrap items-center justify-between gap-2 text-left text-xs">
        <span className="font-semibold text-slate-800">
          ai-hedge-fund 技術分析師（virattt）
          <span className="ml-2 font-normal text-slate-500">
            看多 {data.counts.bullish ?? 0} ・ 中性 {data.counts.neutral ?? 0} ・ 看空 {data.counts.bearish ?? 0}
            {data.used_in_ranking ? ' ・ 已占台股排名 70%' : ' ・ 美股僅參考（回測混入排名較差）'}
          </span>
        </span>
        <span className="text-slate-500">{open ? '收合 ▲' : '展開最強／最弱 ▼'}</span>
      </button>
      {open ? (
        <div className="mt-2 space-y-2">
          {list('最看多 10 檔', data.bullish)}
          {list('最看空 10 檔', data.bearish)}
          <p className="text-[11px] text-slate-400">滑鼠移到代號上可看趨勢／動能／均值回歸／波動／統計套利子分數。</p>
        </div>
      ) : null}
    </div>
  );
}

function MarketCard({ market, data }: { market: 'us' | 'tw'; data: MarketAdvice }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-slate-900">{MARKET_LABEL[market]}</span>
          <span className={`rounded px-2 py-0.5 text-xs font-medium ${LEVEL_STYLE[data.level]}`}>{LEVEL_LABEL[data.level]}</span>
          <span className="text-sm text-slate-700">{data.summary}</span>
        </div>
        {data.momentum_share != null ? (
          <span className="text-xs text-slate-500">動能部位實際占比 {Math.round(data.momentum_share * 100)}%（低檔空槽資金也放動能）</span>
        ) : null}
      </div>
      {data.actions.length ? (
        <div className="mt-3">
          <div className="mb-1 text-xs font-semibold text-emerald-800">今日操作</div>
          <ItemTable items={data.actions} />
        </div>
      ) : null}
      {data.watch.length ? (
        <div className="mt-3">
          <div className="mb-1 text-xs font-semibold text-amber-800">觀察名單</div>
          <ItemTable items={data.watch} />
        </div>
      ) : null}
      {!data.actions.length && !data.watch.length ? (
        <p className="mt-2 text-sm text-slate-500">沒有新低檔訊號、沒有接近觸發價的股票、今天不是月調日——持股照抱即可。</p>
      ) : null}
      {data.virattt ? <ViratttSection data={data.virattt} /> : null}
      {data.hot_groups.length || data.oversold_groups.length ? (
        <div className="mt-3 flex flex-wrap gap-3 text-xs text-slate-600">
          {data.hot_groups.length ? <span>過熱族群：{data.hot_groups.join('、')}</span> : null}
          {data.oversold_groups.length ? <span>超賣族群：{data.oversold_groups.join('、')}</span> : null}
        </div>
      ) : null}
    </div>
  );
}

function SectorTable({ rows }: { rows: SectorMood[] }) {
  const [kind, setKind] = useState<'us' | 'tw' | 'etf'>('etf');
  const shown = useMemo(() => {
    const r = kind === 'etf' ? rows.filter((x) => x.kind === 'etf') : rows.filter((x) => x.kind === 'group' && x.key.startsWith(kind === 'us' ? '美股' : '台股'));
    return [...r].sort((a, b) => b.ret_3m_pct - a.ret_3m_pct);
  }, [rows, kind]);
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div className="text-sm font-semibold text-slate-900">
          類股情緒<span className="ml-2 text-xs font-normal text-slate-500">過熱＝RSI ≥ 70 且高於 50 日線 8%；超賣＝RSI ≤ 30 或低於 50 日線 10%</span>
        </div>
        <div className="flex gap-1">
          {(
            [
              ['etf', '美股類股 ETF'],
              ['us', '美股 AI 族群'],
              ['tw', '台股 AI 族群'],
            ] as const
          ).map(([k, lab]) => (
            <button
              key={k}
              type="button"
              onClick={() => setKind(k)}
              className={`rounded-md border px-2.5 py-1 text-xs ${kind === k ? 'border-slate-900 bg-slate-900 text-white' : 'border-slate-200 text-slate-600'}`}
            >
              {lab}
            </button>
          ))}
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[600px] text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-left text-xs text-slate-500">
              <th className="py-1.5 pr-2">類股</th>
              <th className="py-1.5 pr-2">情緒</th>
              <th className="py-1.5 pr-2 text-right">1 個月</th>
              <th className="py-1.5 pr-2 text-right">3 個月</th>
              <th className="py-1.5 pr-2 text-right">相對 SPY 3 個月</th>
              <th className="py-1.5 pr-2 text-right">RSI14</th>
              <th className="py-1.5 pr-2 text-right">距 50 日線</th>
              {kind !== 'etf' ? <th className="py-1.5 pr-2 text-right">站上 50 日線</th> : null}
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => (
              <tr key={r.key} className="border-b border-slate-100">
                <td className="py-1.5 pr-2 text-slate-900">
                  {r.name.replace(/^(美股|台股)·/, '')}
                  {r.kind === 'etf' ? <span className="ml-1 text-xs text-slate-400">{r.key}</span> : <span className="ml-1 text-xs text-slate-400">{r.members} 檔</span>}
                </td>
                <td className="py-1.5 pr-2">
                  <span className={`rounded px-1.5 py-0.5 text-xs ${MOOD_STYLE[r.mood]}`}>{r.mood}</span>
                </td>
                <td className={`py-1.5 pr-2 text-right tabular-nums ${tone(r.ret_1m_pct)}`}>{pct(r.ret_1m_pct)}</td>
                <td className={`py-1.5 pr-2 text-right tabular-nums ${tone(r.ret_3m_pct)}`}>{pct(r.ret_3m_pct)}</td>
                <td className={`py-1.5 pr-2 text-right tabular-nums ${tone(r.rel_spy_3m_pct)}`}>{pct(r.rel_spy_3m_pct)}</td>
                <td className="py-1.5 pr-2 text-right tabular-nums">{r.rsi14}</td>
                <td className={`py-1.5 pr-2 text-right tabular-nums ${tone(r.dist_ma50_pct)}`}>{pct(r.dist_ma50_pct)}</td>
                {kind !== 'etf' ? <td className="py-1.5 pr-2 text-right tabular-nums">{r.breadth_above_ma50_pct ?? '—'}%</td> : null}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Evidence({ advice }: { advice: DailyAdvice }) {
  const [m, setM] = useState<'us' | 'tw'>('us');
  const ev = advice.evidence?.[m];
  if (!ev) return null;
  const periods = ['全期', '2017-2021', '2022-今'];
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div className="text-sm font-semibold text-slate-900">情緒怎麼用？——回測證據</div>
        <div className="flex gap-1">
          {(['us', 'tw'] as const).map((k) => (
            <button
              key={k}
              type="button"
              onClick={() => setM(k)}
              className={`rounded-md border px-2.5 py-1 text-xs ${m === k ? 'border-slate-900 bg-slate-900 text-white' : 'border-slate-200 text-slate-600'}`}
            >
              {MARKET_LABEL[k]}
            </button>
          ))}
        </div>
      </div>
      {ev.sentiment_study ? (
        <>
          <div className="mb-1 text-xs text-slate-500">恐懼貪婪分區 → 之後的平均報酬（{ev.sentiment_study.since} 起）</div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[480px] text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-left text-xs text-slate-500">
                  <th className="py-1.5 pr-2">情緒</th>
                  <th className="py-1.5 pr-2 text-right">天數</th>
                  <th className="py-1.5 pr-2 text-right">股票池 1 個月</th>
                  <th className="py-1.5 pr-2 text-right">股票池 3 個月</th>
                  <th className="py-1.5 pr-2 text-right">股票池 6 個月</th>
                  <th className="py-1.5 pr-2 text-right">大盤 3 個月</th>
                </tr>
              </thead>
              <tbody>
                {ev.sentiment_study.rows.map((r) => (
                  <tr key={r.bucket} className="border-b border-slate-100">
                    <td className="py-1.5 pr-2">{r.bucket}</td>
                    <td className="py-1.5 pr-2 text-right tabular-nums">{r.days}</td>
                    <td className={`py-1.5 pr-2 text-right tabular-nums ${tone(r.pool_21)}`}>{pct(r.pool_21)}</td>
                    <td className={`py-1.5 pr-2 text-right tabular-nums ${tone(r.pool_63)}`}>{pct(r.pool_63)}</td>
                    <td className={`py-1.5 pr-2 text-right tabular-nums ${tone(r.pool_126)}`}>{pct(r.pool_126)}</td>
                    <td className={`py-1.5 pr-2 text-right tabular-nums ${tone(r.bench_63)}`}>{pct(r.bench_63)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-1 text-xs text-slate-500">{ev.sentiment_study.note}</p>
        </>
      ) : null}
      {ev.sentiment_ablation?.length ? (
        <>
          <div className="mb-1 mt-4 text-xs text-slate-500">策略消融（T-1 收盤訊號 → T 開盤成交、成本 0.2%）：年化報酬 / 最大回撤 / Sharpe</div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[560px] text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-left text-xs text-slate-500">
                  <th className="py-1.5 pr-2">版本</th>
                  {periods.map((p) => (
                    <th key={p} className="py-1.5 pr-2 text-right">
                      {p}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {ev.sentiment_ablation.map((row) => (
                  <tr key={row.name} className="border-b border-slate-100">
                    <td className="py-1.5 pr-2">{row.name}</td>
                    {periods.map((p) => {
                      const s = row[p] as { cagr_pct: number; max_drawdown_pct: number; sharpe: number } | undefined;
                      return (
                        <td key={p} className="py-1.5 pr-2 text-right tabular-nums">
                          {s ? `${s.cagr_pct}% / ${s.max_drawdown_pct}% / ${s.sharpe}` : '—'}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : null}
      <p className="mt-3 text-xs text-slate-500">
        測過但沒採用（報酬變差或兩市場不一致）：恐慌時放寬低檔門檻到 -20%/-25%、恐慌時加開低檔槽位、貪婪 &gt; 80 暫停買進、只在恐懼時才做低檔、新資金等恐懼才投入、每日檢查動能出場；
        virattt/ai-hedge-fund 的技術分析師（趨勢/動能/均值回歸加權）：台股排名混入 70% 後報酬較高（已採用），美股混入反而變差（不採用）。
      </p>
    </div>
  );
}

export function DailyAdvicePanel() {
  const [advice, setAdvice] = useState<DailyAdvice | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const load = () => {
    setLoading(true);
    setError('');
    fetchDailyAdvice()
      .then(setAdvice)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  };
  useEffect(load, []);

  const s = advice?.sentiment;
  const fg = s?.fear_greed;

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <CalendarCheck className="h-4 w-4" />
              {advice ? `${advice.date} 每日建議 ・ 更新 ${advice.generated_at.replace('T', ' ').slice(0, 16)} ・ 下次月調 ${advice.next_rebalance}` : '讀取中…'}
            </div>
            <div className="mt-1 text-lg font-semibold text-slate-900">{advice?.headline ?? '—'}</div>
          </div>
          <Button variant="outline" size="sm" onClick={load} disabled={loading}>
            <RefreshCw className={`mr-1 h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            重新整理
          </Button>
        </div>
        {error ? <p className="mt-2 text-sm text-rose-600">{error}</p> : null}
        {advice?.sentiment_guidance.length ? (
          <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-slate-700">
            {advice.sentiment_guidance.map((g) => (
              <li key={g}>{g}</li>
            ))}
          </ul>
        ) : null}
      </div>

      <HoldingsTodayCard />

      {s ? (
        <div className="grid gap-4 md:grid-cols-3">
          <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm md:col-span-1">
            <div className="flex items-center gap-1.5 text-xs text-slate-500">
              <Activity className="h-4 w-4" />
              恐懼貪婪指數
            </div>
            <div className="mt-1 flex items-baseline gap-2">
              <span className={`text-3xl font-semibold tabular-nums ${fgColor(fg?.score)}`}>{fg?.score != null ? Math.round(fg.score) : '—'}</span>
              <span className={`text-sm font-medium ${fgColor(fg?.score)}`}>{fg?.label}</span>
            </div>
            <Gauge value={fg?.score} />
            <div className="mt-2 grid grid-cols-3 gap-1 text-center text-xs text-slate-500">
              <div>昨日 {fg?.prev_1d != null ? Math.round(fg.prev_1d) : '—'}</div>
              <div>1 週前 {fg?.prev_1w != null ? Math.round(fg.prev_1w) : '—'}</div>
              <div>1 個月前 {fg?.prev_1m != null ? Math.round(fg.prev_1m) : '—'}</div>
            </div>
            {s.cnn ? (
              <div className="mt-2 rounded bg-slate-50 px-2 py-1.5 text-xs text-slate-600">
                CNN 官方：<span className={`font-semibold ${fgColor(s.cnn.score)}`}>{Math.round(s.cnn.score)}</span>（{s.cnn.rating}）
                {s.cnn_validation ? ` ・ 與自建版近一年相關 ${s.cnn_validation.corr}` : ''}
              </div>
            ) : null}
            <FgSpark points={fg?.history ?? []} />
            <p className="mt-1 text-[11px] text-slate-400">規則用自建版本（只用當日以前資料，可回測）。</p>
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <div className="text-xs text-slate-500">VIX 波動率指數</div>
            <div className="mt-1 flex items-baseline gap-2">
              <span className={`text-3xl font-semibold tabular-nums ${s.vix.panic ? 'text-rose-700' : 'text-slate-900'}`}>{s.vix.value ?? '—'}</span>
              {s.vix.value != null && s.vix.prev_1d != null ? (
                <span className={`text-sm ${s.vix.value > s.vix.prev_1d ? 'text-rose-600' : 'text-emerald-600'}`}>
                  {s.vix.value > s.vix.prev_1d ? '▲' : '▼'} {Math.abs(s.vix.value - s.vix.prev_1d).toFixed(2)}
                </span>
              ) : null}
            </div>
            <div className="mt-2 space-y-1 text-xs text-slate-600">
              <div>50 日均：{s.vix.ma50 ?? '—'}</div>
              <div>VIX 3 個月：{s.vix.vix3m ?? '—'}</div>
              <div>
                期限結構：
                {s.vix.backwardation ? <span className="font-medium text-rose-700">逆價差（短線恐慌）</span> : <span className="text-emerald-700">正價差（正常）</span>}
              </div>
              <div className="pt-1 text-slate-500">VIX ≥ 30 視為恐慌；歷史上 VIX 越高，之後 3~6 個月報酬越高。</div>
            </div>
            {s.taiex ? (
              <div className="mt-3 border-t border-slate-100 pt-2 text-xs text-slate-600">
                <div className="font-medium text-slate-800">台股加權指數 {s.taiex.close.toLocaleString()}</div>
                <div>
                  距 200 日線 <span className={tone(s.taiex.vs_ma200_pct)}>{pct(s.taiex.vs_ma200_pct)}</span> ・ RSI {s.taiex.rsi14} ・ 1 個月{' '}
                  <span className={tone(s.taiex.ret_1m_pct)}>{pct(s.taiex.ret_1m_pct)}</span>
                </div>
              </div>
            ) : null}
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <div className="text-xs text-slate-500">恐懼貪婪成分（0 = 最恐懼，100 = 最貪婪；近一年百分位）</div>
            <div className="mt-2 space-y-1.5">
              {(fg?.components ?? []).map((c) => (
                <div key={c.key} className="text-xs">
                  <div className="flex justify-between text-slate-600">
                    <span>{c.name}</span>
                    <span className={`tabular-nums ${fgColor(c.score)}`}>{c.score}</span>
                  </div>
                  <div className="mt-0.5 h-1.5 rounded bg-slate-100">
                    <div className="h-1.5 rounded bg-slate-700" style={{ width: `${Math.max(2, c.score)}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : null}

      {advice
        ? (['us', 'tw'] as const).map((m) => (advice.markets[m] ? <MarketCard key={m} market={m} data={advice.markets[m]!} /> : null))
        : null}

      {s?.sectors?.length ? <SectorTable rows={s.sectors} /> : null}

      {advice ? <Evidence advice={advice} /> : null}

      {advice?.history?.length ? (
        <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
          <div className="mb-2 text-sm font-semibold text-slate-900">近 30 天每日建議</div>
          <div className="space-y-1 text-sm">
            {advice.history.map((h) => (
              <div key={h.date} className="flex flex-wrap gap-2 border-b border-slate-100 py-1">
                <span className="w-24 shrink-0 tabular-nums text-slate-500">{h.date}</span>
                <span className="text-slate-800">{h.headline}</span>
                {Object.values(h.markets).some((x) => x.buys?.length) ? (
                  <span className="text-xs text-emerald-700">
                    買：{Object.values(h.markets).flatMap((x) => x.buys ?? []).join('、')}
                  </span>
                ) : null}
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {advice ? (
        <div className="rounded-lg border border-slate-200 bg-white p-4 text-xs text-slate-500 shadow-sm">
          <div className="mb-1 font-semibold text-slate-700">每日檢查的規則</div>
          <ul className="list-disc space-y-0.5 pl-5">
            {advice.rules.map((r) => (
              <li key={r.rule}>
                {r.rule} <span className="text-slate-400">（{r.check}）</span>
              </li>
            ))}
          </ul>
          <p className="mt-2">{advice.disclaimer}</p>
        </div>
      ) : null}
    </div>
  );
}
