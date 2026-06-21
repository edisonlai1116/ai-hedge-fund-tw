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
NICOLAS_OPINIONS_JSON = os.path.join(DATA_DIR, "nicolas_opinions.json")
# 尼可拉斯楊沒有公開 podcast RSS → 改用其 YouTube 頻道(@nicolasyounglive)的官方 XML feed
# 自動追蹤最新影片/直播。channel_id 可用環境變數覆寫；NICOLAS_FEED 預設即指向該 YouTube feed。
NICOLAS_YT_CHANNEL = os.environ.get("NICOLAS_YT_CHANNEL", "UCXUP_aBLQBNFgLjvnrMTHtw").strip()
NICOLAS_FEED = os.environ.get(
    "NICOLAS_FEED", f"https://www.youtube.com/feeds/videos.xml?channel_id={NICOLAS_YT_CHANNEL}"
).strip()
REPORT_JSON = os.path.join(DATA_DIR, "daily_report.json")
HISTORY_DIR = os.path.join(DATA_DIR, "history")
HISTORY_INDEX = os.path.join(HISTORY_DIR, "index.json")
HOLDINGS_TXT = os.path.join(REPO_ROOT, "股票成本.txt")
GOOAYE_FEED = "https://feeds.soundon.fm/podcasts/954689a5-3096-43a4-a80b-7810b219cef3.xml"

# --- 綜合買進分數權重 -------------------------------------------------------
# 2026-06-19 調整：原本只有 technical(50%) 主導，但實證發現 technical 偏「價值／均值回歸」
# （懲罰高 RSI、加分低本益比與超跌），導致排名最前的多是漲不動的大型權值半導體與防禦股
# （如 JNJ、LIN、2330、TSM），真正會漲的中小型強勢股（2379、3008、WDC、CIFR）反被排到 40 名後。
# 用 06-13~06-15 三天 Top 50 實測：buy_score 與未來報酬的 Spearman 僅 +0.29/+0.17/-0.03（近乎無預測力），
# 但「近一個月相對強度(動能)」達 +0.31/+0.38/+0.41。故新增 relative_strength 因子並降低 technical 權重，
# 新公式回測 Spearman 提升為 +0.43/+0.45/+0.33。relative_strength 缺值時以中性 50 代入（與其他因子一致）。
WEIGHTS = {"technical": 0.35, "relative_strength": 0.30, "gooaye": 0.20, "ticker_news": 0.05, "macro": 0.10}
_NUM = re.compile(r"^-?\d[\d,]*(?:\.\d+)?$")

# 最終排序輸出的檔數（台美股合併取前 N）
TOP_N = 50

# 台股 universe：台灣 50（0050）成分股為主的大型權值/熱門股（免 Key，固定清單）。
TW_UNIVERSE = [
    "2330", "2317", "2454", "2308", "2382", "2412", "2881", "2882", "2891", "2886",
    "3711", "2303", "2002", "1303", "1301", "1216", "2207", "2884", "2885", "2892",
    "2880", "2883", "5880", "2890", "2887", "3045", "4904", "2912", "1101", "2357",
    "2395", "2603", "2609", "2615", "3008", "3034", "3037", "3231", "2376", "2377",
    "6505", "9910", "2474", "2345", "3661", "4938", "2379", "3017", "5871", "2327",
]

# 美股 universe：大型權值/熱門股後盾清單。即使 Yahoo 當日熱門榜抓不到，美股也一定會進榜。
US_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "BRK-B", "JPM",
    "LLY", "V", "UNH", "XOM", "MA", "COST", "HD", "PG", "JNJ", "ABBV",
    "NFLX", "BAC", "CRM", "ORCL", "AMD", "KO", "PEP", "WMT", "MRK", "CVX",
    "ADBE", "QCOM", "TXN", "INTC", "MU", "PLTR", "SMCI", "ARM", "TSM", "ASML",
]

# 台股代號 → 中文名（給網頁顯示「2330 台積電」）。
TW_NAMES = {
    "2330": "台積電", "2317": "鴻海", "2454": "聯發科", "2308": "台達電", "2382": "廣達",
    "2412": "中華電", "2881": "富邦金", "2882": "國泰金", "2891": "中信金", "2886": "兆豐金",
    "3711": "日月光投控", "2303": "聯電", "2002": "中鋼", "1303": "南亞", "1301": "台塑",
    "1216": "統一", "2207": "和泰車", "2884": "玉山金", "2885": "元大金", "2892": "第一金",
    "2880": "華南金", "2883": "開發金", "5880": "合庫金", "2890": "永豐金", "2887": "台新金",
    "3045": "台灣大", "4904": "遠傳", "2912": "統一超", "1101": "台泥", "2357": "華碩",
    "2395": "研華", "2603": "長榮", "2609": "陽明", "2615": "萬海", "3008": "大立光",
    "3034": "聯詠", "3037": "欣興", "3231": "緯創", "2376": "技嘉", "2377": "微星",
    "6505": "台塑化", "9910": "豐泰", "2474": "可成", "2345": "智邦", "3661": "世芯-KY",
    "4938": "和碩", "2379": "瑞昱", "3017": "奇鋐", "5871": "中租-KY", "2327": "國巨",
    "5876": "上海商銀", "2301": "光寶科", "2421": "建準", "6409": "旭隼", "2890.TW": "永豐金",
}


# ===== 純函式（可離線單元測試） ============================================
def relative_strength_score(mom_20d: Optional[float]) -> Optional[float]:
    """把「近一個月（約 20 交易日）報酬率 %」映射成 0–100 相對強度分數。
    +20% → 100、0% → 50、-20% → 0（線性夾擠）。缺值回 None（後續以中性 50 代入）。"""
    if mom_20d is None:
        return None
    return max(0.0, min(100.0, 50.0 + 2.5 * float(mom_20d)))


def compute_buy_score(technical: Optional[float], gooaye: Optional[float],
                      ticker_news: Optional[float], macro: Optional[float],
                      relative_strength: Optional[float] = None) -> Dict:
    """把各面向分數（0–100，缺項以 50 中性代入）加權成買進分數。"""
    parts = {
        "technical": 50 if technical is None else float(technical),
        "relative_strength": 50 if relative_strength is None else float(relative_strength),
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


def _fetch_latest_episode_stdlib(feed_url: str) -> Dict:
    """只用 Python 標準庫抓 RSS 最新一集（不需 feedparser）。失敗回 {}。

    用於偵測/顯示最新集數（標題、日期、id）；不下載音檔、不轉錄。
    """
    import re
    import urllib.request
    try:
        req = urllib.request.Request(feed_url, headers={"User-Agent": "Mozilla/5.0"})
        xml = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "ignore")
    except Exception:
        return {}
    m = re.search(r"<item>(.*?)</item>", xml, re.S)
    if not m:
        return {}
    item = m.group(1)
    def _tag(name):
        mm = re.search(rf"<{name}>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</{name}>", item, re.S)
        return mm.group(1).strip() if mm else ""
    title = _tag("title")
    guid = _tag("guid") or title
    pub = _tag("pubDate")
    return {"id": guid, "title": title, "published": pub} if title else {}


def scan_gooaye(feed_url: str = GOOAYE_FEED, opinions_path: str = OPINIONS_JSON) -> Dict:
    """掃描股癌 RSS；偵測到新集數就（mock fallback）抽取點名個股並 merge 進 opinions JSON。

    回傳 {updated, episode_title, episode_id, total_opinions}。任何失敗都安全降級。
    免 feedparser 環境也能至少顯示最新集數（用標準庫抓 RSS）。
    """
    store = _read_json(opinions_path, {"last_episode_id": None, "opinions": []})
    if isinstance(store, list):  # 容錯：舊格式（純 list）
        store = {"last_episode_id": None, "opinions": store}

    result = {"updated": False, "episode_title": None, "episode_id": store.get("last_episode_id"),
              "total_opinions": len(store.get("opinions", []))}

    # 1) 先用 AudioFeedAdapter（feedparser）；可抽取點名個股（mock fallback）。
    episode = None
    try:
        from src.sentiment.audio_feed_adapter import AudioFeedAdapter
        adapter = AudioFeedAdapter(feed_url=feed_url, source_name="Gooaye")
        episode = adapter.fetch_recent_data()
        if episode and "id" in episode:
            result["episode_title"] = episode.get("title")
            result["episode_id"] = episode.get("id")
            if episode["id"] != store.get("last_episode_id"):
                opinions = adapter.extract_opinions(episode) or []
                store["opinions"].extend(opinions)
                store["last_episode_id"] = episode["id"]
                _write_json(store, opinions_path)
                result.update({"updated": True, "total_opinions": len(store["opinions"])})
            return result
    except Exception as e:
        print(f"[scan_gooaye] AudioFeedAdapter 降級：{e}")

    # 2) 退回標準庫：至少顯示最新集數（標題/日期），不抽取點名。
    ep = _fetch_latest_episode_stdlib(feed_url)
    if ep:
        result["episode_title"] = ep.get("title")
        result["episode_id"] = ep.get("id")
        result["episode_published"] = ep.get("published")
    return result


def _fetch_latest_youtube(feed_url: str) -> Dict:
    """只用標準庫解析 YouTube 頻道 Atom feed 最新一支影片/直播（不需 feedparser）。
    YouTube feed 為 Atom 格式（<entry> / <yt:videoId> / <title> / <published>），與 RSS 不同。
    回傳 {id, title, published, url}；失敗回 {}。不下載影片、不轉錄。"""
    import re
    import urllib.request
    try:
        req = urllib.request.Request(feed_url, headers={"User-Agent": "Mozilla/5.0"})
        xml = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "ignore")
    except Exception:
        return {}
    m = re.search(r"<entry>(.*?)</entry>", xml, re.S)
    if not m:
        return {}
    entry = m.group(1)
    def _tag(name):
        mm = re.search(rf"<{name}>(.*?)</{name}>", entry, re.S)
        return mm.group(1).strip() if mm else ""
    vid = _tag("yt:videoId")
    title = _tag("title")
    if not vid or not title:
        return {}
    return {
        "id": f"yt:{vid}",
        "title": title,
        "published": _tag("published"),
        "url": f"https://www.youtube.com/watch?v={vid}",
    }


def scan_nicolas(feed_url: str = NICOLAS_FEED, opinions_path: str = NICOLAS_OPINIONS_JSON) -> Dict:
    """自動追蹤『尼可拉斯楊Live』最新一集——**改用其 YouTube 頻道官方 XML feed**（他沒有公開 podcast RSS）。

    偵測到最新影片/直播即記錄標題、日期、連結與 id 進 nicolas_opinions.json 的 `latest_episode`，
    並更新 last_episode_id（有新片 → updated=True）。個股觀點本身仍為種子/人工維護——逐集自動抽取
    需語音轉錄（本機 Whisper），雲端做不到，故只自動「追蹤集數」、不自動改寫觀點清單。
    任何失敗安全降級，不影響其餘流程。"""
    store = _read_json(opinions_path, {"last_episode_id": None, "opinions": []})
    if isinstance(store, list):
        store = {"last_episode_id": None, "opinions": store}
    result = {"updated": False, "episode_title": None, "episode_id": store.get("last_episode_id"),
              "total_opinions": len(store.get("opinions", [])), "source": "尼可拉斯楊Live"}
    if not feed_url:
        result["note"] = "未設定追蹤來源。"
        return result

    ep = _fetch_latest_youtube(feed_url)
    if not ep:
        result["note"] = "YouTube feed 暫時取不到（安全降級，沿用既有觀點）。"
        return result

    result["episode_title"] = ep.get("title")
    result["episode_id"] = ep.get("id")
    result["episode_published"] = ep.get("published")
    result["episode_url"] = ep.get("url")
    # 記錄最新一集（供前端/狀態顯示「尼可拉斯楊最新一集」）。
    store["latest_episode"] = {
        "title": ep.get("title"), "published": ep.get("published"),
        "url": ep.get("url"), "id": ep.get("id"), "checked_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
    }
    if ep["id"] != store.get("last_episode_id"):
        store["last_episode_id"] = ep["id"]
        result["updated"] = True
    _write_json(store, opinions_path)
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
        # 近一個月（約 20 交易日）報酬率 → 相對強度／動能因子。免額外下載，沿用同一份報價。
        mom_20d = None
        try:
            closes = df["Close"].dropna()
            if len(closes) > 21:
                mom_20d = (float(closes.iloc[-1]) / float(closes.iloc[-21]) - 1.0) * 100.0
        except Exception:
            mom_20d = None
        return {
            "composite_score": getattr(r, "composite_score", 50),
            "mom_20d": mom_20d,
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


def _full_analysis(symbol: str) -> Optional[Dict]:
    """Top 50 入選後的完整分析（lightweight=False）：取出各 agent 看法、
    3/6/9/12 個月預測股價、各天期建議買賣價。失敗回 None。"""
    try:
        from src.simple_signal import download_prices, build_report
        df = download_prices(symbol, "2y")
        if df is None or df.empty:
            return None
        r = build_report(symbol, df, fetch_fundamentals=False, lightweight=False)
        agents = [
            {
                "name": a.get("name"),
                "signal": a.get("signal"),
                "confidence": a.get("confidence"),
                "summary": a.get("summary"),
            }
            for a in (getattr(r, "agents", None) or [])
        ]
        return {
            "latest_close": getattr(r, "latest_close", None),
            "bias": getattr(r, "bias", ""),
            "today_action": getattr(r, "today_action", ""),
            "buy_zone": getattr(r, "buy_zone", ""),
            "sell_zone": getattr(r, "sell_zone", ""),
            "stop_loss": getattr(r, "stop_loss", ""),
            "expected_return_pct": getattr(r, "expected_return_pct", None),
            "risk_reward_ratio": getattr(r, "risk_reward_ratio", None),
            "agents": agents,
            "horizons": getattr(r, "horizons", None) or [],
            "price_forecast": getattr(r, "price_forecast", None),
        }
    except Exception as e:
        print(f"[_full_analysis] {symbol} 降級：{e}")
        return None


def _ticker_news(symbol: str) -> Optional[Dict]:
    try:
        from src.sentiment.news_feed import ticker_news_sentiment
        return ticker_news_sentiment(symbol)
    except Exception:
        return None


def _load_opinions(path: str) -> List[Dict]:
    store = _read_json(path, {})
    return store.get("opinions", []) if isinstance(store, dict) else (store or [])


def _gooaye_consensus(ticker: str) -> Dict:
    """專家輿情共識：合併股癌 + 尼可拉斯楊Live（多來源加權），再 fallback DB。
    兩個來源的觀點一起餵進 score_opinions（依 source 權重加權），讓買進分數的『輿情』分項
    同時反映股癌與尼可拉斯楊的看法。"""
    simple = ticker.split(".")[0].upper()
    matched: List[Dict] = []
    for path in (OPINIONS_JSON, NICOLAS_OPINIONS_JSON):
        if os.path.exists(path):
            matched += [o for o in _load_opinions(path)
                        if str(o.get("target_ticker", "")).split(".")[0].upper() == simple]
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


def _movers(limit: int = 40) -> List[str]:
    try:
        from src.sentiment.market_movers import fetch_market_movers
        return fetch_market_movers(limit=limit)
    except Exception:
        return []


def analyze_ticker(raw_ticker: str, macro_score: int, held: bool, named: bool, mover: bool = False) -> Dict:
    """組裝單檔的買進分數列。"""
    symbol = _normalize(raw_ticker)
    tech = _technical(symbol)
    news = _ticker_news(symbol)
    cons = _gooaye_consensus(raw_ticker)

    technical_score = tech["composite_score"] if tech else None
    rs_score = relative_strength_score(tech.get("mom_20d")) if tech else None
    gooaye_score = cons["consensus_score"] if cons.get("opinions") else None
    news_score = news["score"] if news else None

    scored = compute_buy_score(technical_score, gooaye_score, news_score, macro_score,
                               relative_strength=rs_score)
    top_op = cons["opinions"][0] if cons.get("opinions") else None
    market = "tw" if symbol.endswith(".TW") or symbol.endswith(".TWO") else "us"
    base = raw_ticker.split(".")[0].upper()
    name = TW_NAMES.get(base, "") if market == "tw" else base
    return {
        "ticker": raw_ticker,
        "symbol": symbol,
        "name": name,
        "market": market,
        "held": held,
        "gooaye_named": named,
        "market_mover": mover,
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


def build_report(movers: List[str], holdings: List[str], opinions_store: Dict,
                 macro: Dict, gooaye_status: Dict) -> Dict:
    """組裝完整每日報告 dict。

    標的池（反映每日市場變化）＝ 美股當日熱門榜 ∪ 股癌點名；
    持股只用來標記（不主導排序）。
    """
    ops = opinions_store.get("opinions", []) if isinstance(opinions_store, dict) else (opinions_store or [])
    named_tickers = sorted({str(o.get("target_ticker", "")).strip() for o in ops if o.get("target_ticker")})

    held_set = {h.upper() for h in holdings}
    named_simple = {n.split(".")[0].upper() for n in named_tickers}
    mover_simple = {m.split(".")[0].upper() for m in movers}

    # 標的池（台美股）：美股當日熱門榜 ∪ 美股大型股後盾 ∪ 台股台灣50 ∪ 股癌點名（simple 代號去重）
    universe = []
    seen = set()
    for t in list(movers) + list(US_UNIVERSE) + list(TW_UNIVERSE) + named_tickers:
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
            mover=(key in mover_simple),
        ))
    rows = rank_rows(rows)[:TOP_N]   # 台美股合併取前 50

    # 入選 Top 50 才跑完整分析（agents / 3-6-9-12 月預測 / 各天期買賣價），控制運算量。
    for r in rows:
        r["detail"] = _full_analysis(r["symbol"])

    tz = timezone(timedelta(hours=8))  # 台北時間
    now = datetime.now(tz)
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "generated_date": now.strftime("%Y-%m-%d"),
        "mode": "免 Key（台美股 Top 50：美股熱門榜 + 台灣50 + 股癌共識 + Yahoo 技術面 + 新聞輿情）",
        "universe": {
            "us_movers": len(movers), "tw_universe": len(TW_UNIVERSE),
            "gooaye_named": len(named_tickers), "scanned": len(universe), "top_n": TOP_N,
            "tw_in_top": sum(1 for r in rows if r.get("market") == "tw"),
            "us_in_top": sum(1 for r in rows if r.get("market") == "us"),
        },
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
    nicolas_status = scan_nicolas()
    print(f"[daily_report] nicolas: updated={nicolas_status['updated']} title={nicolas_status.get('episode_title')} "
          f"opinions={nicolas_status['total_opinions']}")

    holdings = parse_holding_tickers(_read_text(HOLDINGS_TXT))
    opinions_store = _read_json(OPINIONS_JSON, {"opinions": []})
    macro = _macro()
    movers = _movers()
    print(f"[daily_report] macro score={macro.get('score')} ({macro.get('n')} headlines); "
          f"movers={len(movers)}; holdings(tag only)={len(holdings)}")

    report = build_report(movers, holdings, opinions_store, macro, gooaye_status)
    _write_json(report, REPORT_JSON)
    _write_json(report, os.path.join(HISTORY_DIR, f"{report['generated_date']}.json"))
    idx = update_history_index(report)
    print(f"[daily_report] wrote {REPORT_JSON} with {len(report['top_picks'])} picks; history days={len(idx)}")


if __name__ == "__main__":
    main()
