import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='replace')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio

from app.backend.routes import api_router
from app.backend.database.connection import engine
from app.backend.database.models import Base
from app.backend.services.ollama_service import ollama_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Hedge Fund API", description="Backend API for AI Hedge Fund", version="0.1.0")

# Initialize database tables (this is safe to run multiple times)
Base.metadata.create_all(bind=engine)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routes
app.include_router(api_router)


async def sentiment_background_updater():
    """Periodically fetches and scans latest podcast episodes and favorite creators' URLs in the background."""
    logger.info("⏳ Sentiment Auto-Updater registered as background task.")
    await asyncio.sleep(8)  # Wait for DB/server to boot up
    
    # Define a list of favorite creators and their URLs
    favorite_channels = [
        {"url": "https://www.youtube.com/watch?v=8-m_oHkFZv8", "name": "工程師投資 (Engineer Alpha)", "ticker": "MU"},
        {"url": "https://www.youtube.com/watch?v=NlnJKwGEB-g", "name": "Minnie (米妮)", "ticker": "MU"}
    ]
    
    while True:
        try:
            logger.info("🔄 Running background Multi-Source Sentiment Auto-Updater...")
            from app.backend.database.connection import SessionLocal
            from app.backend.database.models import PodcastEpisode, PodcastTicker, CustomSentiment
            from src.sentiment.audio_feed_adapter import AudioFeedAdapter
            from src.sentiment.custom_ingest_service import CustomIngestService
            
            db = SessionLocal()
            try:
                # 1. Sync Podcast
                adapter = AudioFeedAdapter(
                    feed_url="https://feeds.soundon.fm/podcasts/954689a5-3096-43a4-a80b-7810b219cef3.xml",
                    source_name="Gooaye"
                )
                episode_data = adapter.fetch_recent_data()
                if episode_data and "id" in episode_data:
                    existing = db.query(PodcastEpisode).filter(PodcastEpisode.id == episode_data["id"]).first()
                    if not existing:
                        logger.info(f"🎙️ Auto-discovered new Podcast episode: {episode_data['title']}")
                        opinions = adapter.extract_opinions(episode_data)
                        new_episode = PodcastEpisode(
                            id=episode_data["id"],
                            title=episode_data["title"],
                            published_date=episode_data["published"],
                            audio_url=episode_data["audio_url"],
                            transcript=episode_data.get("transcript", "")
                        )
                        db.add(new_episode)
                        db.flush()
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
                        db.commit()
                        logger.info(f"🎙️ Auto-synchronized latest Podcast episode successfully.")
                
                # 2. Sync Favorite Creators
                service = CustomIngestService()
                for chan in favorite_channels:
                    existing_sentiment = db.query(CustomSentiment).filter(CustomSentiment.url == chan["url"]).first()
                    if not existing_sentiment:
                        logger.info(f"🔗 Auto-ingesting favorite creator's content: {chan['name']}")
                        raw_text = service.fetch_url_content(chan["url"])
                        opinions = service.extract_opinion_from_text(
                            url=chan["url"],
                            text=raw_text,
                            analyst_name=chan["name"],
                            target_ticker=chan["ticker"]
                        )
                        if opinions:
                            op = opinions[0]
                            new_sentiment = CustomSentiment(
                                url=chan["url"],
                                analyst_name=op["analyst_name"],
                                ticker=chan["ticker"],
                                sentiment_label=op["sentiment_label"],
                                sentiment_score=op["sentiment_score"],
                                core_logic=op["core_logic"],
                                original_quote=op["original_quote"]
                            )
                            db.add(new_sentiment)
                            db.commit()
                            logger.info(f"🔗 Successfully auto-ingested favorite creator's content: {chan['name']}")
                            
            except Exception as inner_e:
                db.rollback()
                logger.error(f"⚠️ Background Sentiment Auto-Updater inner error: {inner_e}")
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"⚠️ Background Sentiment Auto-Updater error: {e}")
            
        # Run every 2 hours
        await asyncio.sleep(7200)


@app.on_event("startup")
async def startup_event():
    """Startup event to check Ollama availability."""
    # Launch the background sentiment auto-updater
    asyncio.create_task(sentiment_background_updater())
    
    try:
        logger.info("Checking Ollama availability...")
        status = await ollama_service.check_ollama_status()
        
        if status["installed"]:
            if status["running"]:
                logger.info(f"✓ Ollama is installed and running at {status['server_url']}")
                if status["available_models"]:
                    logger.info(f"✓ Available models: {', '.join(status['available_models'])}")
                else:
                    logger.info("ℹ No models are currently downloaded")
            else:
                logger.info("ℹ Ollama is installed but not running")
                logger.info("ℹ You can start it from the Settings page or manually with 'ollama serve'")
        else:
            logger.info("ℹ Ollama is not installed. Install it to use local models.")
            logger.info("ℹ Visit https://ollama.com to download and install Ollama")
            
    except Exception as e:
        logger.warning(f"Could not check Ollama status: {e}")
        logger.info("ℹ Ollama integration is available if you install it later")
