import { useEffect, useMemo, useState } from 'react';
import { Bot, ClipboardCopy, Loader2, Send } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { askQuestion, fetchAskPresets, type AskResult } from './services/strategy-api';
import { loadHoldings } from './StrategyViews';

const FALLBACK_PRESETS = [
  '我現在滿倉，如果想賣出一些換現金，應該先賣哪幾檔？各賣多少？',
  '今天我的持股有哪些需要動作？為什麼？',
  '我的持股是不是太集中在某個主題？要怎麼調整？',
];

async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand('copy');
      document.body.removeChild(ta);
      return ok;
    } catch {
      return false;
    }
  }
}

export function AskPanel() {
  const holdings = useMemo(() => loadHoldings(), []);
  const [presets, setPresets] = useState<string[]>(FALLBACK_PRESETS);
  const [llmAvailable, setLlmAvailable] = useState<boolean | null>(null);
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState<AskResult | null>(null);
  const [loading, setLoading] = useState<'llm' | 'copy' | null>(null);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState('');
  const [showPrompt, setShowPrompt] = useState(false);

  useEffect(() => {
    fetchAskPresets()
      .then((p) => {
        if (p.questions?.length) setPresets(p.questions);
        setLlmAvailable(p.llm_available);
      })
      .catch(() => setLlmAvailable(false));
  }, []);

  const run = async (callLlm: boolean) => {
    if (!question.trim()) {
      setError('請先輸入問題，或點一個常用問題。');
      return;
    }
    setError('');
    setCopied('');
    setLoading(callLlm ? 'llm' : 'copy');
    try {
      const r = await askQuestion(question, holdings, callLlm);
      setResult(r);
      if (!callLlm) {
        const ok = await copyText(r.prompt);
        setCopied(ok ? '已複製提問（問題＋持股＋系統建議），可以貼到 ChatGPT / Claude / Gemini。' : '瀏覽器不允許自動複製，請展開下方提問全文手動複製。');
        setShowPrompt(true);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(null);
    }
  };

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
        <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
          <Bot className="h-4 w-4" />
          AI 問答（結合我的持股與系統今日建議）
        </div>
        <p className="mt-1 text-xs text-slate-500">
          系統先算好持股評估、今日建議、換現金先後順序，再交給免費 LLM（Google Gemini）解讀；也可以只複製整份提問到外面的 LLM 問。
          {holdings.length ? ` 目前帶入 ${holdings.length} 檔持股（來自「我的持股」）。` : ' 尚未輸入持股：到「我的持股」分頁輸入後，回答會針對你的部位。'}
        </p>

        <div className="mt-3 flex flex-wrap gap-1.5">
          {presets.map((q) => (
            <button
              key={q}
              type="button"
              onClick={() => setQuestion(q)}
              className={`rounded-full border px-2.5 py-1 text-xs ${question === q ? 'border-slate-900 bg-slate-900 text-white' : 'border-slate-200 text-slate-600 hover:border-slate-400'}`}
            >
              {q}
            </button>
          ))}
        </div>

        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          maxLength={1000}
          rows={3}
          placeholder="例如：我現在滿倉，想賣出約 50 萬換現金，要賣什麼？"
          className="mt-3 w-full rounded-md border border-slate-200 p-2 text-sm outline-none focus:border-slate-400"
        />

        <div className="mt-2 flex flex-wrap items-center gap-2">
          <Button onClick={() => run(true)} disabled={loading !== null || llmAvailable === false} size="sm">
            {loading === 'llm' ? <Loader2 className="mr-1 h-4 w-4 animate-spin" /> : <Send className="mr-1 h-4 w-4" />}
            問 AI（免費 Gemini）
          </Button>
          <Button onClick={() => run(false)} disabled={loading !== null} size="sm" variant="outline">
            {loading === 'copy' ? <Loader2 className="mr-1 h-4 w-4 animate-spin" /> : <ClipboardCopy className="mr-1 h-4 w-4" />}
            複製提問到外部 LLM
          </Button>
          {llmAvailable === false ? (
            <span className="text-xs text-amber-700">伺服器未設定免費 LLM 金鑰，請先用「複製提問」。</span>
          ) : null}
          {loading === 'llm' ? <span className="text-xs text-slate-500">免費模型可能需要 10~60 秒…</span> : null}
        </div>
        {error ? <p className="mt-2 text-sm text-rose-600">{error}</p> : null}
        {copied ? <p className="mt-2 text-sm text-emerald-700">{copied}</p> : null}
        <p className="mt-2 text-[11px] text-slate-400">
          隱私：持股只用於這次提問、不會儲存；按「問 AI」時，整份提問（含持股）會送到 Google Gemini。回答僅供參考，非投資建議。
        </p>
      </div>

      {result?.answer || result?.error ? (
        <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
          <div className="mb-2 flex items-center justify-between text-xs text-slate-500">
            <span>{result.model ? `回答模型：${result.model}` : '回答'}</span>
            {result.answer ? (
              <button type="button" className="hover:text-slate-900" onClick={() => copyText(result.answer || '')}>
                複製回答
              </button>
            ) : null}
          </div>
          {result.answer ? (
            <div className="whitespace-pre-wrap text-sm leading-relaxed text-slate-800">{result.answer.replace(/\*\*/g, '')}</div>
          ) : (
            <p className="text-sm text-amber-700">{result.error}</p>
          )}
        </div>
      ) : null}

      {result?.cash_plan?.length ? (
        <div className="rounded-lg border border-slate-200 bg-white p-4 text-sm shadow-sm">
          <div className="mb-1 font-semibold text-slate-900">系統試算：要換現金時的賣出順序</div>
          <div className="mb-2 text-xs text-slate-500">確定性規則（未經回測）：系統已建議賣出 → 集中度過高 → 排名偏低／高風險低檔 → 排名偏後 → 一般持股 → 核心 ETF → 最後才動低檔可買與動能前段。</div>
          <ol className="list-decimal space-y-0.5 pl-5">
            {result.cash_plan.map((x) => (
              <li key={x.symbol}>
                <span className="font-medium">{x.symbol}</span>
                <span className="text-slate-500">（NT${Math.round(x.value_twd).toLocaleString()}）</span>：{x.why}
                {x.suggest_trim_twd ? <span className="text-rose-700">，建議減碼約 NT${x.suggest_trim_twd.toLocaleString()}</span> : null}
              </li>
            ))}
          </ol>
        </div>
      ) : null}

      {result?.prompt ? (
        <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <button type="button" onClick={() => setShowPrompt((s) => !s)} className="text-sm font-semibold text-slate-900">
              {showPrompt ? '▼' : '▶'} 提問全文（可複製到外部 LLM）
            </button>
            <button
              type="button"
              className="text-xs text-slate-500 hover:text-slate-900"
              onClick={async () => setCopied((await copyText(result.prompt)) ? '已複製提問全文。' : '複製失敗，請手動選取。')}
            >
              複製提問全文
            </button>
          </div>
          {showPrompt ? (
            <textarea readOnly value={result.prompt} rows={18} className="mt-2 w-full rounded-md border border-slate-200 bg-slate-50 p-2 font-mono text-xs" />
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
