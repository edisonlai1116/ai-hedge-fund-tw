import { Fragment, useEffect, useState } from 'react';
import { Gauge, RefreshCw } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  fetchOpportunity,
  fetchRankingValidation,
  opportunityLive,
  type OpportunityReport,
  type OppRow,
} from './services/strategy-api';

const STATUS_STYLE: Record<string, string> = {
  BUY_NOW: 'bg-emerald-100 text-emerald-800 border-emerald-200',
  BUY_ON_PULLBACK: 'bg-teal-50 text-teal-800 border-teal-200',
  HOLD_CORE: 'bg-sky-100 text-sky-800 border-sky-200',
  HOLD: 'bg-sky-50 text-sky-800 border-sky-200',
  PARTIAL_PROFIT: 'bg-amber-100 text-amber-800 border-amber-200',
  WAIT: 'bg-slate-100 text-slate-600 border-slate-200',
  SELL: 'bg-rose-100 text-rose-800 border-rose-200',
};
const REGIME_TEXT: Record<string, string> = {
  BULL: '多頭：成長股領漲、波動低',
  NEUTRAL: '中性',
  RISK_OFF: '風險趨避：大盤跌破年線或恐慌指數高',
  RISK_ON_ROTATION: '資金輪動：電力/能源相對科技走強',
};
const COLS: [string, string][] = [
  ['low_entry', '低檔★'],
  ['momentum_rank', '動能排名★'],
  ['quality', '品質'],
  ['growth', '成長'],
  ['earnings_acceleration', '財報加速'],
  ['ai_exposure', 'AI'],
  ['industry_momentum', '產業'],
  ['valuation', '估值'],
  ['technical', '技術'],
  ['catalyst', '催化'],
  ['relative_strength', '相對強度'],
  ['risk', '風險'],
  ['overextension', '過熱'],
];

const n0 = (v: number | null | undefined) => (v == null ? '—' : Math.round(v).toString());
const cell = (k: string, v: number | null | undefined) => {
  if (v == null) return 'text-slate-300';
  const bad = k === 'risk' || k === 'overextension';
  const good = bad ? v <= 40 : v >= 65;
  const poor = bad ? v >= 70 : v <= 35;
  return good ? 'text-emerald-700 font-medium' : poor ? 'text-rose-700' : 'text-slate-700';
};

function loadHoldings(): { ticker: string; cost: number; shares: number }[] {
  try {
    const d = JSON.parse(localStorage.getItem('real_holdings_v2') || '{}');
    return Array.isArray(d.holdings) ? d.holdings.filter((h: { ticker: string; shares: number }) => h.ticker && h.shares > 0) : [];
  } catch {
    return [];
  }
}

export function OpportunityPanel() {
  const [data, setData] = useState<OpportunityReport | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [validation, setValidation] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    fetchOpportunity().then(setData).catch((e) => setError(e instanceof Error ? e.message : String(e)));
    fetchRankingValidation('us').then(setValidation).catch(() => setValidation(null));
  }, []);

  const runLive = async (withHoldings: boolean) => {
    const syms = input.split(/[\s,，、]+/).map((s) => s.trim().toUpperCase()).filter(Boolean);
    const holdings = withHoldings ? loadHoldings() : [];
    if (!syms.length && !holdings.length) {
      setError('請輸入代號，或先到「我的持股」輸入持股。');
      return;
    }
    setLoading(true);
    setError('');
    try {
      setData(await opportunityLive(syms.length ? syms : (data?.ranking ?? []).slice(0, 12).map((r) => r.ticker), holdings.slice(0, 13)));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
              <Gauge className="h-4 w-4" />
              機會評分（同一批股票互相比較：現在誰的風險報酬最好）
            </div>
            <div className="text-xs text-slate-500">
              {data ? `更新：${data.generated_at.replace('T', ' ').slice(0, 16)} UTC ・ 模式 ${data.mode}` : '讀取中…'}
            </div>
          </div>
          {data ? (
            <div className="rounded-md border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs">
              市場狀態 <b>{data.regime.regime}</b>：{REGIME_TEXT[data.regime.regime] ?? ''}
            </div>
          ) : null}
        </div>
        <div className="mt-3 flex flex-wrap items-end gap-2">
          <Input value={input} onChange={(e) => setInput(e.target.value)} placeholder="自訂代號（最多 25 檔）：AVGO VST CEG …" className="h-9 min-w-[240px] flex-1" />
          <Button className="h-9" type="button" disabled={loading} onClick={() => runLive(false)}>
            <RefreshCw className={`mr-1 h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            即時評分
          </Button>
          <Button variant="outline" className="h-9" type="button" disabled={loading} onClick={() => runLive(true)}>
            用我的持股評分（含輪動建議）
          </Button>
        </div>
        {loading ? <p className="mt-2 text-xs text-slate-500">抓財報、預估與新聞中，約需 20~60 秒…</p> : null}
        {error ? <p className="mt-2 text-sm text-rose-600">{error}</p> : null}
      </div>

      {data?.rotations?.length ? (
        <div className="rounded-lg border border-amber-300 bg-amber-50 p-4 text-sm shadow-sm">
          <div className="font-semibold text-slate-900">輪動建議（SELL A → BUY B）</div>
          <ul className="mt-1 space-y-0.5 text-xs text-slate-700">
            {data.rotations.map((r) => (
              <li key={`${r.sell}-${r.buy}`}>
                <b>{r.type === 'FULL_ROTATION' ? '全部' : `部分（約 ${Math.round(r.fraction_of_A * 100)}%）`}</b>：賣 {r.sell} → 買 {r.buy}
                　rotation_score {r.rotation_score >= 0 ? '+' : ''}{(r.rotation_score * 100).toFixed(1)}%　{r.why}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {data ? (
        <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[1100px] text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-left text-xs text-slate-500">
                  <th className="py-1.5 pr-2">#</th>
                  <th className="py-1.5 pr-2">標的</th>
                  <th className="py-1.5 pr-2 text-right">現價</th>
                  <th className="py-1.5 pr-2 text-right">機會</th>
                  <th className="py-1.5 pr-2">動作</th>
                  {COLS.map(([k, label]) => (
                    <th key={k} className="py-1.5 pr-2 text-right">{label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.ranking.map((r) => (
                  <Fragment key={r.ticker}>
                    <tr className="cursor-pointer border-b border-slate-100 hover:bg-slate-50" onClick={() => setOpen(open === r.ticker ? null : r.ticker)}>
                      <td className="py-1.5 pr-2 text-slate-500">{r.batch_rank ?? '—'}</td>
                      <td className="py-1.5 pr-2">
                        <span className="font-medium text-slate-900">{r.ticker.replace(/\.TWO?$/, '')}</span>
                        {r.held ? <span className="ml-1 rounded bg-slate-200 px-1 text-[10px]">持有</span> : null}
                        <div className="text-[11px] text-slate-500">{r.theme ?? ''}{r.ai?.cycle_position ? ` · ${r.ai.cycle_position}` : ''}</div>
                      </td>
                      <td className="py-1.5 pr-2 text-right">{r.price ?? '—'}</td>
                      <td className="py-1.5 pr-2 text-right font-semibold">{n0(r.opportunity)}</td>
                      <td className="py-1.5 pr-2">
                        <span className={`whitespace-nowrap rounded border px-1.5 py-0.5 text-xs ${STATUS_STYLE[r.status] ?? STATUS_STYLE.WAIT}`}>{r.status}</span>
                        {r.flags.includes('POTENTIAL_OVERSOLD') ? <div className="text-[10px] text-amber-700">潛在超跌</div> : null}
                      </td>
                      {COLS.map(([k]) => (
                        <td key={k} className={`py-1.5 pr-2 text-right ${cell(k, r.scores?.[k])}`}>{n0(r.scores?.[k])}</td>
                      ))}
                    </tr>
                    {open === r.ticker ? (
                      <tr className="border-b border-slate-100 bg-slate-50">
                        <td colSpan={5 + COLS.length} className="px-3 py-3">
                          <OppCard r={r} />
                        </td>
                      </tr>
                    ) : null}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-slate-500">★ 已驗證：低檔分數（長線贏家距 52 週高點深度，占 70%）與動能排名（占 30% 中的 61%）。其餘模組無歷史時點資料或驗證未改善，僅供參考。點任一列看完整評分卡。</p>
          <ul className="mt-1 list-disc pl-5 text-[11px] text-slate-500">
            {data.notes.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </div>
      ) : null}

      {validation ? <ValidationBox v={validation} /> : null}
    </div>
  );
}

function OppCard({ r }: { r: OppRow }) {
  const z = r.entry || {};
  const zone = (k: 'zone_1' | 'zone_2' | 'zone_3') =>
    z[k] ? `${z[k]!.low}–${z[k]!.high}（${z[k]!.basis}，${z[k]!.pct_from_price}%）` : '—';
  const n = r.narrative || {};
  return (
    <div className="grid gap-3 text-xs text-slate-700 md:grid-cols-2">
      <div className="space-y-1">
        <div className="text-sm font-semibold text-slate-900">
          {r.ticker} {r.company ? <span className="font-normal text-slate-500">{r.company}</span> : null}
        </div>
        <div>
          {r.sub_theme ?? '-'} ・ 估值類別 {r.valuation_class ?? '-'} ・ 資料覆蓋 {Math.round((r.opportunity_detail?.coverage ?? 0) * 100)}% ・
          已驗證占比 {Math.round((r.opportunity_detail?.validated_share ?? 0) * 100)}%
        </div>
        <div>
          <b>動作 {r.status}</b>：{r.status_why}
        </div>
        <div>Zone 1：{zone('zone_1')}</div>
        <div>Zone 2：{zone('zone_2')}</div>
        <div>Zone 3：{zone('zone_3')}</div>
        {r.sec_valuation?.relative_to_own_3y != null ? (
          <div>SEC 估值：P/S {r.sec_valuation.ps}（自身 3 年中位數 {r.sec_valuation.ps_3y_median}，{r.sec_valuation.relative_to_own_3y} 倍）{r.sec_valuation.pe ? `・P/E ${r.sec_valuation.pe}` : ''}</div>
        ) : null}
        <div>Avoid Above：{z.avoid_above ?? '—'}{z.notes?.length ? `（${z.notes.join('；')}）` : ''}</div>
        <div>
          Better than：{r.rotation_view?.better_than?.join('、') || '—'}　Worse than：{r.rotation_view?.worse_than?.join('、') || '—'}
        </div>
        {r.shock ? (
          <div className="rounded border border-amber-200 bg-amber-50 p-1.5">
            大跌診斷：{r.shock.shock.date} {(r.shock.shock.return * 100).toFixed(1)}%，原因 {r.shock.shock.causes.join('、')}；
            基本面受損 {r.shock.fundamental_damage_score}/100 → {r.shock.verdict ?? '無'}
          </div>
        ) : null}
      </div>
      <div className="space-y-1">
        <div><b>為什麼現在值得買？</b>{n.why_now}</div>
        <div><b>為什麼現在不能追？</b>{n.why_not_chase}</div>
        <div><b>最重要的風險？</b>{n.key_risk}</div>
        <div><b>什麼價格開始有吸引力？</b>{n.attractive_price}</div>
        <div><b>什麼事件會讓 thesis 失效？</b>{n.thesis_invalidation}</div>
        <div><b>Key Catalyst：</b>{n.key_catalyst}</div>
        {r.catalyst?.events?.length ? (
          <ul className="list-disc pl-4 text-[11px] text-slate-500">
            {r.catalyst.events.slice(0, 4).map((e) => (
              <li key={e.title}>
                [{e.category} {e.direction > 0 ? '+' : e.direction < 0 ? '−' : '0'}，已反映 {e.priced_in ?? '?'}%] {e.title}
              </li>
            ))}
          </ul>
        ) : null}
      </div>
    </div>
  );
}

type Metrics = { CAGR?: number; Sharpe?: number; MaxDD?: number };

function ValidationBox({ v }: { v: Record<string, unknown> }) {
  const results = (v.results ?? {}) as Record<string, Record<string, Metrics>>;
  const pick = ['最終策略：動能排名 + 過熱≥60 等降溫才進', 'Opportunity v1（多模組價格綜合）', '同池等權持有', 'QQQ 買進持有', 'SPY 買進持有'];
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="text-sm font-semibold text-slate-900">驗證（美股 AI 科技池，point-in-time：T-1 收盤訊號 → T 開盤成交，成本 0.2%/次）</div>
      <div className="overflow-x-auto">
        <table className="mt-2 w-full min-w-[560px] text-xs">
          <thead>
            <tr className="border-b border-slate-200 text-left text-slate-500">
              <th className="py-1 pr-2">策略</th>
              <th className="py-1 pr-2 text-right">樣本內 2017-21</th>
              <th className="py-1 pr-2 text-right">樣本外 2022-</th>
              <th className="py-1 text-right">全期 Sharpe / 回撤</th>
            </tr>
          </thead>
          <tbody>
            {pick.filter((k) => results[k]).map((k) => {
              const r = results[k];
              const is_ = r['In-sample 2017-2021'] ?? {};
              const oos = r['Out-of-sample 2022-'] ?? {};
              const all = r['全期'] ?? {};
              return (
                <tr key={k} className="border-b border-slate-100">
                  <td className="py-1 pr-2">{k}</td>
                  <td className="py-1 pr-2 text-right">{is_.CAGR ?? '—'}%</td>
                  <td className="py-1 pr-2 text-right">{oos.CAGR ?? '—'}%</td>
                  <td className="py-1 text-right">{all.Sharpe ?? '—'} / {all.MaxDD ?? '—'}%</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-[11px] text-amber-700">⚠️ 股票池是今天的 AI 科技股（後見之明偏差），所有數字都被高估；相對比較較有意義。多模組綜合分數輸給單純動能排名，因此最終只保留動能排名為核心。</p>
    </div>
  );
}
