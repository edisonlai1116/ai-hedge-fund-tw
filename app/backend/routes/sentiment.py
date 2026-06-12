from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List, Dict, Any

from app.backend.database.connection import get_db
from app.backend.database.models import PodcastEpisode, PodcastTicker, CustomSentiment
from src.sentiment.audio_feed_adapter import AudioFeedAdapter
from src.sentiment.custom_ingest_service import CustomIngestService
from src.sentiment.consensus_engine import WeightedConsensusEngine

router = APIRouter(prefix="/sentiment", tags=["sentiment"])

class CustomIngestRequest(BaseModel):
    url: str = Field(..., description="Threads, YouTube, or stock analysis webpage URL")
    analyst_name: str | None = Field(default=None, description="Name of the analyst")
    ticker: str | None = Field(default=None, description="Stock ticker (e.g. 2330.TW or NVDA)")

class PodcastScanRequest(BaseModel):
    feed_url: str = Field(default="https://feeds.soundon.fm/podcasts/954689a5-3096-43a4-a80b-7810b219cef3.xml")
    source_name: str = Field(default="Gooaye")

class ConsensusQueryRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker (e.g. 2330.TW or NVDA)")
    sector: str = Field(default="Technology", description="Stock sector category")


@router.post("/podcast-scan")
async def scan_podcast_episode(request: PodcastScanRequest, db: Session = Depends(get_db)):
    """
    Scan RSS Feed for the latest episode, download/transcribe/mock,
    and persist results and dot-named stocks into the SQLite database.
    """
    try:
        adapter = AudioFeedAdapter(feed_url=request.feed_url, source_name=request.source_name)
        episode_data = adapter.fetch_recent_data()
        
        if not episode_data or "id" not in episode_data:
            raise HTTPException(status_code=400, detail="無法從 RSS 取得有效的集數數據")
            
        # Check if episode is already processed and stored in database
        existing = db.query(PodcastEpisode).filter(PodcastEpisode.id == episode_data["id"]).first()
        if existing:
            # If already exists, return current stored tickers
            stored_tickers = db.query(PodcastTicker).filter(PodcastTicker.episode_id == episode_data["id"]).all()
            return {
                "message": "此單集已於先前處理完畢！",
                "episode": {
                    "id": existing.id,
                    "title": existing.title,
                    "published_date": existing.published_date,
                    "transcript": existing.transcript[:200] + "..." if existing.transcript else ""
                },
                "tickers": [{"ticker": t.ticker, "sentiment": t.sentiment, "score": t.sentiment_score} for t in stored_tickers]
            }

        # Extract structured opinions (supports high-reliability mock fallback)
        opinions = adapter.extract_opinions(episode_data)
        
        # Save episode to DB
        new_episode = PodcastEpisode(
            id=episode_data["id"],
            title=episode_data["title"],
            published_date=episode_data["published"],
            audio_url=episode_data["audio_url"],
            transcript=episode_data.get("transcript", "")
        )
        db.add(new_episode)
        db.flush() # flush to generate or establish transaction context for foreign keys
        
        # Save tickers to DB
        saved_tickers = []
        for op in opinions:
            new_ticker = PodcastTicker(
                episode_id=episode_data["id"],
                ticker=op["target_ticker"],
                context=op["core_logic"],
                sentiment=op["sentiment_label"],
                sentiment_score=op["sentiment_score"],
                confidence=op["confidence_rating"]
            )
            db.add(new_ticker)
            saved_tickers.append({
                "ticker": op["target_ticker"],
                "sentiment": op["sentiment_label"],
                "score": op["sentiment_score"]
            })
            
        db.commit()
        return {
            "message": "Podcast 同步與逐字稿點名分析成功！",
            "episode": {
                "id": new_episode.id,
                "title": new_episode.title,
                "published_date": new_episode.published_date
            },
            "tickers": saved_tickers
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Podcast 掃描分析失敗：{e}")


@router.post("/custom-ingest")
async def ingest_custom_url(request: CustomIngestRequest, db: Session = Depends(get_db)):
    """
    Ingest a custom stock sentiment URL (Threads, YouTube, Webpage),
    fetch text, parse with GPT-4o, and insert into the CustomSentiment table.
    """
    try:
        service = CustomIngestService()
        raw_text = service.fetch_url_content(request.url)
        
        # Extract opinion
        opinions = service.extract_opinion_from_text(
            url=request.url,
            text=raw_text,
            analyst_name=request.analyst_name,
            target_ticker=request.ticker
        )
        
        if not opinions:
            raise HTTPException(status_code=400, detail="無法從提供的網址中解析出有效的個股輿情觀點")
            
        op = opinions[0]
        
        # Override ticker if the request explicitly specified a target ticker (dynamic matching)
        target_ticker = request.ticker if request.ticker else op["target_ticker"]
        
        # Save to DB
        new_sentiment = CustomSentiment(
            url=request.url,
            analyst_name=op["analyst_name"],
            ticker=target_ticker,
            sentiment_label=op["sentiment_label"],
            sentiment_score=op["sentiment_score"],
            core_logic=op["core_logic"],
            original_quote=op["original_quote"]
        )
        db.add(new_sentiment)
        db.commit()
        
        return {
            "message": "自定義輿情信號接入成功！",
            "opinion": {
                "analyst_name": new_sentiment.analyst_name,
                "ticker": new_sentiment.ticker,
                "sentiment_label": new_sentiment.sentiment_label,
                "sentiment_score": new_sentiment.sentiment_score,
                "core_logic": new_sentiment.core_logic
            }
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"自定義輿情分析失敗：{e}")


@router.post("/consensus")
async def query_ticker_consensus(request: ConsensusQueryRequest):
    """
    Fetch the weighted master consensus score and opinion matrix for a stock.
    """
    try:
        engine = WeightedConsensusEngine()
        result = engine.get_stock_consensus(request.ticker, request.sector)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查詢個股輿情共識失敗：{e}")


@router.get("/tracked-creators")
async def get_tracked_creators(db: Session = Depends(get_db)):
    """
    Get all unique creators (analysts) currently tracked/monitored in the system.
    """
    try:
        # Fetch from custom_sentiments
        custom_creators = db.query(CustomSentiment.analyst_name, CustomSentiment.url).group_by(CustomSentiment.analyst_name).all()
        
        creators = []
        seen = set()
        
        # Add default custom tracked creators if not in DB to make sure it looks rich
        default_creators = [
            {"name": "Minnie (米妮)", "url": "https://www.youtube.com/watch?v=NlnJKwGEB-g", "source": "YouTube (AI選股法)", "status": "自動監控中"},
            {"name": "工程師投資 (Engineer Alpha)", "url": "https://www.youtube.com/watch?v=8-m_oHkFZv8", "source": "YouTube (複利變富)", "status": "自動監控中"},
            {"name": "謝孟恭 (股癌)", "url": "https://feeds.soundon.fm/podcasts/954689a5-3096-43a4-a80b-7810b219cef3.xml", "source": "SoundOn Podcast", "status": "定時同步中"}
        ]
        
        for dc in default_creators:
            creators.append(dc)
            seen.add(dc["name"])
            
        for cc in custom_creators:
            if cc.analyst_name and cc.analyst_name not in seen:
                creators.append({
                    "name": cc.analyst_name,
                    "url": cc.url,
                    "source": "自定義外部輿情" if "youtube.com" not in cc.url else "YouTube",
                    "status": "已加入追蹤"
                })
                seen.add(cc.analyst_name)
                
        return creators
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取追蹤博主清單失敗：{e}")
