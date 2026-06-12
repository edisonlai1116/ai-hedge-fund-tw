import os
import sqlite3
from typing import List, Dict, Any

# 直接以 stdlib 解析 DB 路徑（= app/backend/hedge_fund.db），不匯入 connection.py，
# 避免把 SQLAlchemy/engine 副作用拖進純讀取的共識查詢路徑（讓 CLI 分析與本引擎可獨立運作）。
DATABASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "app", "backend", "hedge_fund.db",
)


class WeightedConsensusEngine:
    """
    Consensus engine to fetch qualitative opinions (podcast and custom)
    and compute a dynamically-weighted Market Social Consensus Score for tickers.
    """
    
    def __init__(self, db_path: str = str(DATABASE_PATH)):
        self.db_path = db_path

    def get_stock_consensus(self, ticker: str, sector: str = "Technology") -> Dict[str, Any]:
        """
        Fetch all recent analyst opinions (from database) and compute
        the weighted market consensus score.
        """
        opinions = self._fetch_all_opinions(ticker)
        
        # Default fallback if no opinions exist
        if not opinions:
            return {
                "consensus_score": 50,
                "consensus_label": "中性觀察",
                "opinions": []
            }
            
        # Define dynamic analyst weights based on sector
        weights = {
            "Gooaye": 0.40,
            "StatementDog": 0.30,
            "Custom Ingestion Source": 0.20,
            "Howard_Marks": 0.10
        }
        
        if sector == "Financials" or sector == "金融保險業":
            weights = {
                "Howard_Marks": 0.40,
                "MacroMicro": 0.30,
                "Custom Ingestion Source": 0.20,
                "Gooaye": 0.10
            }
            
        weighted_sum = 0.0
        total_weight = 0.0
        
        parsed_opinions = []
        for op in opinions:
            source = op.get("source_name", "Gooaye")
            weight = weights.get(source, 0.15)
            score = op.get("sentiment_score", 50)
            
            weighted_sum += score * weight
            total_weight += weight
            
            parsed_opinions.append({
                "source_name": source,
                "analyst_name": op.get("analyst_name", "專家"),
                "analyst_style": op.get("analyst_style", "Expert"),
                "sentiment_label": op.get("sentiment_label", "Neutral"),
                "sentiment_score": score,
                "core_logic": op.get("core_logic", ""),
                "original_quote": op.get("original_quote", "")
            })
            
        consensus_score = round(weighted_sum / total_weight) if total_weight > 0 else 50
        
        # Classify consensus label
        if consensus_score >= 80:
            consensus_label = "強力多頭共識"
        elif consensus_score >= 65:
            consensus_label = "中性偏多共識"
        elif consensus_score >= 45:
            consensus_label = "中性觀望"
        else:
            consensus_label = "防守避險共識"
            
        return {
            "consensus_score": consensus_score,
            "consensus_label": consensus_label,
            "opinions": parsed_opinions
        }

    def _fetch_all_opinions(self, ticker: str) -> List[Dict[str, Any]]:
        """Fetch both podcast_tickers and custom_sentiments from SQLite database."""
        opinions = []
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 1. Fetch from podcast_tickers
            # Note: Ticker could be stored as 2330 or 2330.TW, let's normalize or fetch both
            simple_ticker = ticker.split(".")[0]
            cursor.execute(
                "SELECT * FROM podcast_tickers WHERE ticker = ? OR ticker LIKE ?",
                (ticker, f"{simple_ticker}%")
            )
            rows = cursor.fetchall()
            for r in rows:
                opinions.append({
                    "source_name": "Gooaye",
                    "analyst_name": "謝孟恭 (股癌)",
                    "analyst_style": "Technology_Momentum",
                    "sentiment_label": r["sentiment"],
                    "sentiment_score": r["sentiment_score"] or 50,
                    "core_logic": r["context"] or "",
                    "original_quote": r["context"] or ""
                })
                
            # 2. Fetch from custom_sentiments
            cursor.execute(
                "SELECT * FROM custom_sentiments WHERE ticker = ? OR ticker LIKE ?",
                (ticker, f"{simple_ticker}%")
            )
            rows = cursor.fetchall()
            for r in rows:
                opinions.append({
                    "source_name": "Custom Ingestion Source",
                    "analyst_name": r["analyst_name"] or "自定義分析師",
                    "analyst_style": "Custom_Expert",
                    "sentiment_label": r["sentiment_label"],
                    "sentiment_score": r["sentiment_score"],
                    "core_logic": r["core_logic"] or "",
                    "original_quote": r["original_quote"] or ""
                })
                
            conn.close()
        except Exception as e:
            print(f"[Error] Failed to query SQLite database: {e}")
            
        return opinions
