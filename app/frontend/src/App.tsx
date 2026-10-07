import { useEffect, useState } from 'react';
import {
  Bot,
  BriefcaseBusiness,
  CalendarCheck,
  ChevronDown,
  Gauge,
  LineChart,
  RadioTower,
  Search,
  Sparkles,
  TrendingUp,
} from 'lucide-react';

import { Toaster } from './components/ui/sonner';
import { fetchSystemStatus, type SystemStatus } from './services/simple-signal-api';
import { OpportunityPanel } from './OpportunityView';
import { DailyAdvicePanel } from './DailyAdviceView';
import { AskPanel } from './AskView';
import { MyHoldingsPanel, StockLookupPanel, StrategyBacktestPanel, StrategyPicksPanel } from './StrategyViews';

type TabKey = 'advice' | 'ask' | 'analyze' | 'daily' | 'opportunity' | 'holdings' | 'backtest';

const TABS: { key: TabKey; label: string; icon: typeof Search; hint: string }[] = [
  { key: 'advice', label: '今日建議', icon: CalendarCheck, hint: '市場情緒・今天要不要動' },
  { key: 'ask', label: 'AI 問答', icon: Bot, hint: '結合持股問免費 LLM' },
  { key: 'analyze', label: '個股查詢', icon: Search, hint: '查任一檔的策略排名' },
  { key: 'daily', label: '策略精選', icon: Sparkles, hint: '本月買進名單（單一策略）' },
  { key: 'opportunity', label: '機會評分', icon: Gauge, hint: '跨股比較・進場區・輪動' },
  { key: 'holdings', label: '我的持股', icon: BriefcaseBusiness, hint: '加碼 / 減碼 / 換股提醒' },
  { key: 'backtest', label: '策略回測', icon: LineChart, hint: '對標 VOO / 0050' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<TabKey>('advice');

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
                <div className="text-xs text-slate-500">每日建議 ・ 市場情緒 ・ 我的持股提醒 ・ 對標 VOO / 0050</div>
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

          {activeTab === 'advice' ? <DailyAdvicePanel /> : null}
          {activeTab === 'ask' ? <AskPanel /> : null}
          {activeTab === 'analyze' ? <StockLookupPanel /> : null}
          {activeTab === 'daily' ? <StrategyPicksPanel /> : null}
          {activeTab === 'opportunity' ? <OpportunityPanel /> : null}
          {activeTab === 'holdings' ? <MyHoldingsPanel /> : null}
          {activeTab === 'backtest' ? <StrategyBacktestPanel /> : null}
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
