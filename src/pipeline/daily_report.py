"""免 API Key 的每日報告 pipeline（給 GitHub Actions 排程用）。

每天自動：
1) 掃描股癌 RSS，偵測是否有新集數；有就（mock）抽取點名個股、merge 進 gooaye_opinions.json。
2) 標的 = 你的持股（股票成本.txt）∪ 股癌點名個股。
3) 每檔算：技術面（Yahoo, 免 Key, simple_signal.build_report）+ 股癌共識 + 個股新聞情緒，
   再疊加市場級總經/地緣輿情，綜合成「買進分數」並排序出「每日最該買」。
4) 輸出 docs/data/daily_report.json（+ 每日 history），供 GitHub Pages 靜態網頁讀取。

隱私：輸出**不含成本與股數**，只含代號與分析結果（Pages 站台在 Free 方案下為公開）。
穩健：任何個股/新聞/RSS 失敗都跳過或回中性，報告一定會產出，網頁不會開天窗。
無 langchain 相依；只用 yfinance/feedparser/requests（皆專案既有相依）。
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

from src.sentiment.consensus_engine import consensus_from_json, score_opinions

# --- 路徑 -------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "docs", "data")
OPINIONS_JSON = os.path.join(DATA_DIR, "gooaye_opinions.json")
REPORT_JSON = os.path.join(DATA_DIR, "daily_report.json")
HISTORY_DIR = os.path.join(DATA_DIR, "history")
HISTORY_INDEX = os.path.join(HISTORY_DIR, "index.json")
HOLDINGS_TXT = os.path.join(REPO_ROOT, "股票成本.txt")
GOOAYE_FEED = "https://feeds.soundon.fm/podcasts/954689a5-3096-43a4-a80b-7810b219cef3.xml"

# --- 綜合買進分數權重 -------------------------------------------------------
WEIGHTS = {"technical": 0.50, "gooaye": 0.25, "ticker_news": 0.10, "macro": 0.15}
_NUM = re.compile(r"^-?\d[\d,]*(?:\.\d+)?$")


# ===== 純函式（可離線單元測試） ============================================
def compute_buy_score(technical: Optional[float], gooaye: Optional[float],
                      ticker_news: Optional[float], macro: Optional[float]) -> Dict:
    """把各面向分數（0–100，缺項以 50 中性代入）加權成買進分數。"""
    parts = {
        "technical": 50 if technical is None else float(technical),
        "gooaye": 50 if gooaye is None else float(gooaye),
        "ticker_news": 50 if ticker_news is None else float(ticker_news),
        "macro": 50 if macro is None else float(macro),
    }
    score = sum(parts[k] * WEIGHTS[k] for k in WEIGHTS)
    return {"buy_score": int(round(score)), "components": {k: int(round(v)) for k, v in parts.items()}}


def recommendation(buy_score: int) -> str:
    if buy_score >= 70:
        return "強力買進"
    if buy_score >= 58:
        return "偏多分批"
    if buy_score >= 45:
        return "中性觀望"
    if buy_score >= 35:
        return "偏空減碼"
    return "避險／賣出"


def rank_rows(rows: List[Dict]) -> List[Dict]:
    """依買進分數由高到低排序。"""
    return sorted(rows, key=lambda r: r.get("buy_score", 0), reverse=True)


def parse_holding_tickers(text: str) -> List[str]:
    """從持股檔文字解析代號（每行 `代號 成本 股數`），只回代號、不回成本。"""
    out = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) >= 3 and _NUM.match(parts[1]) and _NUM.match(parts[2]):
            out.append(parts[0])
    return out


# ===== I/O 與網路（包 try/except，失敗降級） ===============================
def _read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _read_text(path) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def scan_gooaye(feed_url: str = GOOAYE_FEED, opinions_path: str = OPINIONS_JSON) -> Dict:
    """掃描股癌 RSS；偵測到新集數就（mock fallback）抽取點名個股並 merge 進 opinions JSON。

    回傳 {updated, episode_title, episode_id, total_opinions}。任何失敗都安全降級。
    """
    store = _read_json(opinions_path, {"last_episode_id": None, "opinions": []})
    if isinstance(store, list):  # 容錯：舊格式（純 list）
        store = {"last_episode_id": None, "opinions": store}

    result = {"updated": False, "episode_title": None, "episode_id": store.get("last_episode_id"),
              "total_opinions": len(store.get("opinions", []))}
    try:
        from src.sentiment.audio_feed_adapter import AudioFeedAdapter
        adapter = AudioFeedAdapter(feed_url=feed_url, source_name="Gooaye")
        episode = adapter.fetch_recent_data()
        if not episode or "id" not in episode:
            return result
        result["episode_title"] = episode.get("title")
        result["episode_id"] = episode.get("id")
        if episode["id"] == store.get("last_episode_id"):
            return result  # 沒有新集數

        # 新集數 → 抽取點名個股（mock fallback 預設啟用，無 ffmpeg 也能跑）
        opinions = adapter.extract_opinions(episode) or []
        store["opinions"].extend(opinions)
        store["last_episode_id"] = episode["id"]
        _write_json(store, opinions_path)
        result.update({"updated": True, "total_opinions": len(store["opinions"])})
    except Exception as e:
        print(f"[scan_gooaye] 降級：{e}")
    return result


def _normalize(ticker: str) -> str:
    """轉成 Yahoo 代號。純數字 / 數字+字母（台股）→ 加 .TW；其餘原樣。"""
    t = ticker.strip().upper()
    if "." in t:
        return t
    if t and (t[0].isdigit()):
        return f"{t}.TW"
    return t


def _technical(symbol: str) -> Optional[Dict]:
    """免 Key 技術面分析（Yahoo + simple_signal）。失敗回 None。"""
    try:
        from src.simple_signal import download_prices, build_report
        df = download_prices(symbol, "1y")
        if df is None or df.empty:
            return None
        r = build_report(symbol, df, fetch_fundamentals=False, lightweight=True)
        return {
            "composite_score": getattr(r, "composite_score", 50),
            "bias": getattr(r, "bias", ""),
            "today_action": getattr(r, "today_action", ""),
            "buy_zone": getattr(r, "buy_zone", ""),
            "sell_zone": getattr(r, "sell_zone", ""),
            "stop_loss": getattr(r, "stop_loss", ""),
            "latest_close": getattr(r, "latest_close", None),
        }
    except Exception as e:
        print(f"[_technical] {symbol} 降級：{e}")
        return None


def _ticker_news(symbol: str) -> Optional[Dict]:
    try:
        from src.sentiment.news_feed import ticker_news_sentiment
        return ticker_news_sentiment(symbol)
    except Exception:
        return None


def _gooaye_consensus(ticker: str) -> Dict:
    """股癌/輿情共識：優先用 JSON（CI 持久化），再 fallback DB。"""
    if os.path.exists(OPINIONS_JSON):
        store = _read_json(OPINIONS_JSON, {})
        ops = store.get("opinions", []) if isinstance(store, dict) else store
        simple = ticker.split(".")[0].upper()
        matched = [o for o in ops if str(o.get("target_ticker", "")).split(".")[0].upper() == simple]
        if matched:
            return score_opinions(matched)
    try:
        from src.sentiment.consensus_engine import WeightedConsensusEngine
        return WeightedConsensusEngine().get_stock_consensus(ticker)
    except Exception:
        return {"consensus_score": 50, "consensus_label": "中性觀察", "opinions": []}


def _macro() -> Dict:
    try:
        from src.sentiment.news_feed import macro_sentiment
        return macro_sentiment()
    except Exception:
        return {"score": 50, "label": "中性", "n": 0, "sample_headlines": []}


def analyze_ticker(raw_ticker: str, macro_score: int, held: bool, named: bool) -> Dict:
    """組裝單檔的買進分數列。"""
    symbol = _normalize(raw_ticker)
    tech = _technical(symbol)
    news = _ticker_news(symbol)
    cons = _gooaye_consensus(raw_ticker)

    technical_score = tech["composite_score"] if tech else None
    gooaye_score = cons["consensus_score"] if cons.get("opinions") else None
    news_score = news["score"] if news else None

    scored = compute_buy_score(technical_score, gooaye_score, news_score, macro_score)
    top_op = cons["opinions"][0] if cons.get("opinions") else None
    return {
        "ticker": raw_ticker,
        "symbol": symbol,
        "held": held,
        "gooaye_named": named,
        "buy_score": scored["buy_score"],
        "recommendation": recommendation(scored["buy_score"]),
        "components": scored["components"],
        "technical": tech,
        "gooaye": {
            "score": cons.get("consensus_score"),
            "label": cons.get("consensus_label"),
            "opinion_count": len(cons.get("opinions", [])),
            "top_logic": (top_op or {}).get("core_logic", ""),
            "top_quote": (top_op or {}).get("original_quote", ""),
        },
        "news": {"score": (news or {}).get("score"), "label": (news or {}).get("label")},
    }


def build_report(holdings: List[str], opinions_store: Dict, macro: Dict, gooaye_status: Dict) -> Dict:
    """組裝完整每日報告 dict。"""
    ops = opinions_store.get("opinions", []) if isinstance(opinions_store, dict) else (opinions_store or [])
    named_tickers = sorted({str(o.get("target_ticker", "")).strip() for o in ops if o.get("target_ticker")})

    held_set = {h.upper() for h in holdings}
    named_simple = {n.split(".")[0].upper() for n in named_tickers}
    # 標的池：持股 ∪ 股癌點名（用 simple 代號去重，保留原字串）
    universe = []
    seen = set()
    for t in holdings + named_tickers:
        key = t.split(".")[0].upper()
        if key not in seen:
            seen.add(key)
            universe.append(t)

    macro_score = int(macro.get("score", 50))
    rows = []
    for t in universe:
        key = t.split(".")[0].upper()
        rows.append(analyze_ticker(
            t, macro_score,
            held=(key in held_set or t.upper() in held_set),
            named=(key in named_simple),
        ))
    rows = rank_rows(rows)

    tz = timezone(timedelta(hours=8))  # 台北時間
    now = datetime.now(tz)
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "generated_date": now.strftime("%Y-%m-%d"),
        "mode": "免 Key（Yahoo 技術面 + 股癌共識 + 關鍵字新聞輿情）",
        "macro": {
            "score": macro_score,
            "label": macro.get("label", ""),
            "headline_count": macro.get("n", 0),
            "sample_headlines": macro.get("sample_headlines", []),
        },
        "gooaye_status": gooaye_status,
        "weights": WEIGHTS,
        "top_picks": rows,
        "disclaimer": "本報告為自動產生之研究輔助，非投資建議；資料來源含 Yahoo Finance、股癌 Podcast 與公開新聞，僅供參考。",
    }


def update_history_index(report: Dict) -> List[Dict]:
    """維護 history/index.json：每日一筆精簡摘要（給歷史走勢頁畫趨勢與選日期）。"""
    idx = _read_json(HISTORY_INDEX, [])
    if not isinstance(idx, list):
        idx = []
    picks = report.get("top_picks", [])
    top = picks[0] if picks else {}
    summary = {
        "date": report["generated_date"],
        "generated_at": report["generated_at"],
        "macro_score": report.get("macro", {}).get("score", 50),
        "top_ticker": top.get("ticker"),
        "top_score": top.get("buy_score"),
        "strong_buy": sum(1 for p in picks if p.get("recommendation") == "強力買進"),
        "pick_count": len(picks),
    }
    idx = [e for e in idx if e.get("date") != summary["date"]]  # 同日覆蓋
    idx.append(summary)
    idx.sort(key=lambda e: e.get("date", ""))
    _write_json(idx, HISTORY_INDEX)
    return idx


def main():
    print(f"[daily_report] start  data_dir={DATA_DIR}")
    os.makedirs(HISTORY_DIR, exist_ok=True)

    gooaye_status = scan_gooaye()
    print(f"[daily_report] gooaye: updated={gooaye_status['updated']} title={gooaye_status['episode_title']}")

    holdings = parse_holding_tickers(_read_text(HOLDINGS_TXT))
    opinions_store = _read_json(OPINIONS_JSON, {"opinions": []})
    macro = _macro()
    print(f"[daily_report] macro score={macro.get('score')} ({macro.get('n')} headlines); holdings={len(holdings)}")

    report = build_report(holdings, opinions_store, macro, gooaye_status)
    _write_json(report, REPORT_JSON)
    _write_json(report, os.path.join(HISTORY_DIR, f"{report['generated_date']}.json"))
    idx = update_history_index(report)
    print(f"[daily_report] wrote {REPORT_JSON} with {len(report['top_picks'])} picks; history days={len(idx)}")


if __name__ == "__main__":
    main()
