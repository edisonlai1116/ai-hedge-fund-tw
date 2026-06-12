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

from app.backend.database.connection import engine
from app.backend.database.models import Base
from app.backend.routes.simple_signals import router as simple_signals_router
from app.backend.routes.sentiment import router as sentiment_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web_app")

GOOAYE_FEED = "https://feeds.soundon.fm/podcasts/954689a5-3096-43a4-a80b-7810b219cef3.xml"

app = FastAPI(title="AI Hedge Fund (免 Key Web)", version="1.0.0")

# 建立資料表（sentiment 路由需要）
Base.metadata.create_all(bind=engine)

# 同源服務前端 → 理論上免 CORS；仍開放以防前端被單獨部署。
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(simple_signals_router)
app.include_router(sentiment_router)


@app.get("/healthz")
def healthz():
    return {"ok": True}


async def _gooaye_updater():
    """每 2 小時自動掃股癌 RSS；偵測到新集數就（mock fallback）抽取點名並寫入 DB。"""
    await asyncio.sleep(8)
    while True:
        try:
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
        except Exception as e:
            logger.warning(f"股癌自動更新降級：{e}")
        await asyncio.sleep(7200)  # 2 小時


@app.on_event("startup")
async def _startup():
    asyncio.create_task(_gooaye_updater())


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
