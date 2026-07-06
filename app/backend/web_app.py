"""免 API Key 的精簡 Web 服務（給雲端部署用，例如 Render 免費方案）。

只掛載「查個股 / 持股檢視 / 美股當日掃描 / 股癌輿情」需要的免-Key 路由，**不含 langchain 大師委員會**，
所以映像小、啟動快、免費方案也跑得動。同時把前端 build(dist) 以靜態檔服務在同一網域（免 CORS 問題）。

提供：
- /simple-signals/*  ：輸入台股/美股個股 → 進出點、買賣建議、規則型 agent 意見（= 你原本 run-simple-web 的引擎）
- /sentiment/*       ：股癌 / 自定義輿情共識
- 背景任務          ：每 2 小時自動掃股癌 RSS 最新一集並寫入 DB（自動追蹤）
- /                 ：前端網頁（app/frontend/dist）

完整大師 LLM 委員會（warren_buffett 等）需要 API key，不在此精簡服務內；要用請跑本機 run-analysis.ps1。
"""
import asyncio
import logging
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web_app")

GOOAYE_FEED = "https://feeds.soundon.fm/podcasts/954689a5-3096-43a4-a80b-7810b219cef3.xml"

app = FastAPI(title="AI Hedge Fund (免 Key Web)", version="1.0.0")

# 同源服務前端 → 理論上免 CORS；仍開放以防前端被單獨部署。
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

# 韌性啟動：任一段失敗都記錄完整錯誤但不讓整個服務崩潰（避免 Render 啟動即 exit 1）。
_status = {"db": "skipped", "simple_signals": "not-loaded", "sentiment": "not-loaded", "errors": []}


def _safe(label, fn):
    try:
        fn()
        return True
    except Exception:
        tb = traceback.format_exc()
        logger.error(f"[startup] {label} 失敗：\n{tb}")
        _status["errors"].append({"where": label, "error": tb.strip().splitlines()[-1] if tb.strip() else "?"})
        return False


def _init_db():
    from app.backend.database.connection import engine
    from app.backend.database.models import Base
    Base.metadata.create_all(bind=engine)
    _status["db"] = "ok"


def _load_simple_signals():
    from app.backend.routes.simple_signals import router as r
    app.include_router(r)
    _status["simple_signals"] = "ok"


def _load_sentiment():
    from app.backend.routes.sentiment import router as r
    app.include_router(r)
    _status["sentiment"] = "ok"


_safe("db", _init_db)
_safe("simple_signals", _load_simple_signals)
_safe("sentiment", _load_sentiment)


@app.get("/healthz")
def healthz():
    return {"ok": True, "status": _status}


@app.get("/status")
def system_status():
    """系統更新狀態（給前端右上角顯示，確認真的有在更新）：
    - 股癌：目前更新到第幾集（集名/集數 + 發布時間）＋ 背景掃描最後檢查時間（證明 2 小時更新有在跑）
    - 每日 Top 50 報告：上次產生時間/日期
    """
    from datetime import datetime, timedelta, timezone

    tz = timezone(timedelta(hours=8))
    out = {
        "server_time": datetime.now(tz).isoformat(timespec="seconds"),
        "gooaye": {"episode_title": None, "episode_id": None, "published_date": None,
                   "opinion_count": None, "last_checked": None, "source": "none"},
        "daily_report": {"generated_at": None, "generated_date": None, "top_n": None},
    }

    # 1) 每日報告（docs/data/daily_report.json）：報告更新時間 + 報告所用股癌集數
    try:
        import json
        report_path = os.path.join(_DOCS, "data", "daily_report.json")
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        out["daily_report"]["generated_at"] = report.get("generated_at")
        out["daily_report"]["generated_date"] = report.get("generated_date")
        out["daily_report"]["top_n"] = (report.get("universe") or {}).get("top_n")
        gs = report.get("gooaye_status") or {}
        out["gooaye"].update({
            "episode_title": gs.get("episode_title"),
            "episode_id": gs.get("episode_id"),
            "published_date": gs.get("episode_published"),
            "opinion_count": gs.get("total_opinions"),
            "source": "report" if gs.get("episode_title") else out["gooaye"]["source"],
        })
    except Exception as e:
        logger.info(f"/status 讀每日報告失敗：{e}")

    # 2) DB 最新集數（背景每 2h 掃描寫入）：證明系統「現在」仍在更新
    try:
        from app.backend.database.connection import SessionLocal
        from app.backend.database.models import PodcastEpisode, PodcastTicker

        db = SessionLocal()
        try:
            ep = db.query(PodcastEpisode).order_by(PodcastEpisode.created_at.desc()).first()
            if ep is not None:
                cnt = db.query(PodcastTicker).filter(PodcastTicker.episode_id == ep.id).count()
                out["gooaye"].update({
                    "episode_title": ep.title or out["gooaye"]["episode_title"],
                    "episode_id": ep.id or out["gooaye"]["episode_id"],
                    "published_date": ep.published_date or out["gooaye"]["published_date"],
                    "opinion_count": cnt if cnt else out["gooaye"]["opinion_count"],
                    "last_checked": ep.created_at.isoformat(timespec="seconds") if ep.created_at else None,
                    "source": "db",
                })
        finally:
            db.close()
    except Exception as e:
        logger.info(f"/status 讀 DB 集數失敗：{e}")

    # 3) 尼可拉斯楊Live 最新一集（YouTube 自動追蹤；docs/data/nicolas_opinions.json 的 latest_episode）
    out["nicolas"] = {"episode_title": None, "published_date": None, "url": None,
                      "last_checked": None, "opinion_count": None}
    try:
        import json
        npath = os.path.join(_DOCS, "data", "nicolas_opinions.json")
        with open(npath, encoding="utf-8") as f:
            nstore = json.load(f)
        le = nstore.get("latest_episode") or {}
        out["nicolas"].update({
            "episode_title": le.get("title"),
            "published_date": le.get("published"),
            "url": le.get("url"),
            "last_checked": le.get("checked_at"),
            "opinion_count": len(nstore.get("opinions", [])),
        })
    except Exception as e:
        logger.info(f"/status 讀尼可拉斯楊集數失敗：{e}")

    # 4) 程式更新時間（右上角顯示）：Docker 建置時寫入 build_info.json（建置＝部署時間）；
    #    本機開發 fallback 用核心原始碼的最後修改時間。另附 Render 的 git commit 短碼。
    out["code"] = {"updated_at": None, "commit": (os.environ.get("RENDER_GIT_COMMIT") or "")[:7] or None}
    try:
        import json
        root = os.path.abspath(os.path.join(_DOCS, ".."))
        info_path = os.path.join(root, "build_info.json")
        if os.path.exists(info_path):
            with open(info_path, encoding="utf-8") as f:
                out["code"]["updated_at"] = (json.load(f) or {}).get("built_at")
        else:
            src_dir = os.path.join(root, "src")
            mtimes = []
            for base, _dirs, files in os.walk(src_dir):
                mtimes.extend(os.path.getmtime(os.path.join(base, fn)) for fn in files if fn.endswith(".py"))
            if mtimes:
                out["code"]["updated_at"] = datetime.fromtimestamp(max(mtimes), tz).isoformat(timespec="seconds")
    except Exception as e:
        logger.info(f"/status 讀程式更新時間失敗：{e}")

    return out


# 市場情緒 → 現金/股票配置建議（給模擬操作參考）。下載 SPY+VIX 較重，做 30 分鐘快取。
_REGIME_CACHE: dict[str, dict] = {}  # market -> {"data":..., "ts":...}
_REGIME_TTL = 1800.0


@app.get("/market-regime")
def market_regime(market: str = "us"):
    """依市場情緒（VIX + 貪婪指數 + 回撤）建議「股票 / 現金」配置比例，供模擬操作參考。

    回傳含 vix、貪婪指數、regime_score(0–100，越高越偏多/risk-on)、house 動作建議，
    以及由 regime_score 推導的 suggested_stock_pct / suggested_cash_pct。
    """
    import time

    mkt = "tw" if str(market).lower() == "tw" else "us"
    cached = _REGIME_CACHE.get(mkt)
    if cached and (time.time() - cached["ts"] < _REGIME_TTL):
        return cached["data"]

    try:
        from src.sp500_daily import compute_market_regime
        r = compute_market_regime(mkt)
    except Exception as e:
        logger.info(f"/market-regime 計算失敗：{e}")
        return {"ok": False, "error": "regime-unavailable", "market": mkt}

    # regime_score(0–100，越高越該偏多)→ 建議股票% (夾在 30–85，保留銀彈與避免梭哈)，現金% = 餘額。
    score = int(getattr(r, "regime_score", 50) or 50)
    stock_pct = max(30, min(85, round(score / 5) * 5))
    cash_pct = 100 - stock_pct
    data = {
        "ok": True,
        "market": mkt,
        "vix_close": round(float(getattr(r, "vix_close", 0.0) or 0.0), 2),
        "vix_regime": getattr(r, "vix_regime", ""),
        "fear_greed_score": int(getattr(r, "fear_greed_score", 50) or 50),
        "fear_greed_label": getattr(r, "fear_greed_label", ""),
        "spy_drawdown_pct": round(float(getattr(r, "spy_drawdown_pct", 0.0) or 0.0), 2),
        "regime_score": score,
        "action": getattr(r, "action", ""),
        "risk_budget": getattr(r, "risk_budget", ""),
        "summary": getattr(r, "summary", ""),
        "suggested_stock_pct": stock_pct,
        "suggested_cash_pct": cash_pct,
    }
    _REGIME_CACHE[mkt] = {"data": data, "ts": time.time()}
    return data


# 即時報價（給「跟單對帳本」算未實現損益與 SPY 對照用）。
# 走 yfinance 最近收盤，免 API key；做 60 秒記憶體快取避免被頁面反覆打爆。
_QUOTE_CACHE: dict[str, dict] = {}  # symbol -> {"price","name","currency","ts"}
_QUOTE_TTL = 60.0
_TW_NAMES_CACHE: dict | None = None


def _tw_names() -> dict:
    """惰性載入台股中文名對照（來自 daily_report 的 TW_NAMES）；失敗回空 dict。"""
    global _TW_NAMES_CACHE
    if _TW_NAMES_CACHE is None:
        try:
            from src.pipeline.daily_report import TW_NAMES
            _TW_NAMES_CACHE = dict(TW_NAMES)
        except Exception:
            _TW_NAMES_CACHE = {}
    return _TW_NAMES_CACHE


@app.get("/quotes")
def quotes(symbols: str = ""):
    """傳回多檔最新收盤價：/quotes?symbols=AAPL,MU,2330,SPY

    回傳 {"quotes":[{"symbol","resolved","price","currency","name","ok"}...]}。
    任一檔抓不到只把該檔標 ok=false，不讓整個請求失敗。
    """
    import time

    raw = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not raw:
        return {"quotes": []}

    try:
        import yfinance as yf
        from src.simple_signal import normalize_ticker
    except Exception as e:  # 套件缺失時降級
        logger.warning(f"/quotes 無法載入 yfinance：{e}")
        return {"quotes": [{"symbol": s, "ok": False, "error": "quote-engine-unavailable"} for s in raw]}

    now = time.time()
    out = []
    for sym in raw[:60]:  # 上限保護
        cached = _QUOTE_CACHE.get(sym)
        if cached and (now - cached["ts"] < _QUOTE_TTL):
            out.append({**{k: cached[k] for k in ("symbol", "resolved", "price", "currency", "name")}, "ok": True})
            continue
        try:
            market = "tw" if (sym.isdigit() or sym.endswith((".TW", ".TWO"))) else None
            resolved = normalize_ticker(sym, market)
            tk = yf.Ticker(resolved)
            df = tk.history(period="5d")
            if df is None or df.empty or "Close" not in df:
                raise ValueError("no price data")
            price = float(df["Close"].dropna().iloc[-1])
            info = {}
            try:
                info = tk.fast_info or {}
            except Exception:
                info = {}
            is_tw = resolved.endswith((".TW", ".TWO"))
            currency = (info.get("currency") if isinstance(info, dict) else None) or (
                "TWD" if is_tw else "USD"
            )
            # 台股優先用中文名對照；查不到才回退代號。
            name = sym
            if is_tw:
                base = resolved.split(".")[0]
                name = _tw_names().get(base) or _tw_names().get(resolved) or sym
            rec = {
                "symbol": sym,
                "resolved": resolved,
                "price": round(price, 4),
                "currency": currency,
                "name": name,
                "ts": now,
            }
            _QUOTE_CACHE[sym] = rec
            out.append({k: rec[k] for k in ("symbol", "resolved", "price", "currency", "name")} | {"ok": True})
        except Exception as e:
            logger.info(f"/quotes 抓不到 {sym}：{e}")
            out.append({"symbol": sym, "ok": False, "error": "no-data"})
    return {"quotes": out}


def _gooaye_scan_once():
    """股癌掃描的同步重活（網路 + DB），供背景更新在執行緒中呼叫。"""
    from app.backend.database.connection import SessionLocal
    from app.backend.database.models import PodcastEpisode, PodcastTicker
    from src.sentiment.audio_feed_adapter import AudioFeedAdapter

    adapter = AudioFeedAdapter(feed_url=GOOAYE_FEED, source_name="Gooaye")
    episode = adapter.fetch_recent_data()
    if episode and "id" in episode:
        db = SessionLocal()
        try:
            exists = db.query(PodcastEpisode).filter(PodcastEpisode.id == episode["id"]).first()
            if not exists:
                logger.info(f"🎙️ 自動發現股癌新集數：{episode['title']}")
                db.add(PodcastEpisode(
                    id=episode["id"], title=episode["title"],
                    published_date=episode.get("published", ""),
                    audio_url=episode.get("audio_url"),
                    transcript=episode.get("transcript", ""),
                ))
                db.flush()
                for op in (adapter.extract_opinions(episode) or []):
                    db.add(PodcastTicker(
                        episode_id=episode["id"], ticker=op["target_ticker"],
                        context=op["core_logic"], sentiment=op["sentiment_label"],
                        sentiment_score=op["sentiment_score"], confidence=op["confidence_rating"],
                    ))
                db.commit()
        finally:
            db.close()


async def _gooaye_updater():
    """每 2 小時自動掃股癌 RSS；偵測到新集數就（mock fallback）抽取點名並寫入 DB。
    重活在執行緒中跑，避免阻塞 event loop 拖垮服務(502)。"""
    await asyncio.sleep(8)
    while True:
        try:
            await asyncio.to_thread(_gooaye_scan_once)
        except Exception as e:
            logger.warning(f"股癌自動更新降級：{e}")
        await asyncio.sleep(7200)  # 2 小時


async def _nicolas_updater():
    """每 2 小時自動追蹤『尼可拉斯楊Live』最新集數（需設環境變數 NICOLAS_FEED；未設則安全降級、
    沿用種子/人工觀點）。寫進 docs/data/nicolas_opinions.json。"""
    await asyncio.sleep(12)
    while True:
        try:
            from src.pipeline.daily_report import scan_nicolas
            # 在執行緒中跑（含 YouTube feed/字幕網路 I/O），避免阻塞 asyncio event loop 而拖垮整個服務(502)。
            status = await asyncio.to_thread(scan_nicolas)
            if status.get("updated"):
                logger.info(f"🎯 自動發現尼可拉斯楊新集數：{status.get('episode_title')}")
        except Exception as e:
            logger.warning(f"尼可拉斯楊自動更新降級：{e}")
        await asyncio.sleep(7200)  # 2 小時


@app.on_event("startup")
async def _startup():
    asyncio.create_task(_gooaye_updater())
    asyncio.create_task(_nicolas_updater())


# /daily：每日台美股 Top 50 靜態頁（與 GitHub Pages 同一份 docs/）。先掛，才不會被 "/" 接走。
_DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs")
if os.path.isdir(_DOCS):
    app.mount("/daily", StaticFiles(directory=_DOCS, html=True), name="daily")
else:
    logger.warning(f"docs 不存在（{_DOCS}）；/daily 不可用。")

# 服務前端 build（必須放在所有 API 路由與 /daily 之後；mount 在 "/" 會接住其餘路徑）。
_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "dist")
if os.path.isdir(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="frontend")
else:
    logger.warning(f"前端 dist 不存在（{_DIST}）；只提供 API。請先 build 前端或用 Dockerfile.web 部署。")
