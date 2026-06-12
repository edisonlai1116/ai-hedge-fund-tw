import json
import os
import sqlite3
from typing import List, Dict, Any

# 直接以 stdlib 解析 DB 路徑（= app/backend/hedge_fund.db），不匯入 connection.py，
# 避免把 SQLAlchemy/engine 副作用拖進純讀取的共識查詢路徑（讓 CLI 分析與本引擎可獨立運作）。
DATABASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "app", "backend", "hedge_fund.db",
)

# 來源權重（依產業動態調整）。抽成模組常數，供 DB 路徑與 JSON 路徑共用。
DEFAULT_WEIGHTS = {
    "Gooaye": 0.40,
    "StatementDog": 0.30,
    "Custom Ingestion Source": 0.20,
    "Howard_Marks": 0.10,
}
FINANCIALS_WEIGHTS = {
    "Howard_Marks": 0.40,
    "MacroMicro": 0.30,
    "Custom Ingestion Source": 0.20,
    "Gooaye": 0.10,
}


def _classify_label(score: int) -> str:
    if score >= 80:
        return "強力多頭共識"
    if score >= 65:
        return "中性偏多共識"
    if score >= 45:
        return "中性觀望"
    return "防守避險共識"


def score_opinions(opinions: List[Dict[str, Any]], sector: str = "Technology") -> Dict[str, Any]:
    """把一組 opinion dict 加權成共識分數。純函式（無 I/O），DB 與 JSON 路徑共用。

    opinion 需含 source_name / sentiment_score；其餘欄位選填。無 opinion → 中性 50。
    """
    if not opinions:
        return {"consensus_score": 50, "consensus_label": "中性觀察", "opinions": []}

    weights = FINANCIALS_WEIGHTS if sector in ("Financials", "金融保險業") else DEFAULT_WEIGHTS

    weighted_sum = 0.0
    total_weight = 0.0
    parsed = []
    for op in opinions:
        source = op.get("source_name", "Gooaye")
        weight = weights.get(source, 0.15)
        score = op.get("sentiment_score", 50) or 50
        weighted_sum += score * weight
        total_weight += weight
        parsed.append({
            "source_name": source,
            "analyst_name": op.get("analyst_name", "專家"),
            "analyst_style": op.get("analyst_style", "Expert"),
            "sentiment_label": op.get("sentiment_label", "Neutral"),
            "sentiment_score": score,
            "core_logic": op.get("core_logic", ""),
            "original_quote": op.get("original_quote", ""),
        })

    consensus_score = round(weighted_sum / total_weight) if total_weight > 0 else 50
    return {
        "consensus_score": consensus_score,
        "consensus_label": _classify_label(consensus_score),
        "opinions": parsed,
    }


def consensus_from_json(opinions_json_path: str, ticker: str, sector: str = "Technology") -> Dict[str, Any]:
    """從 JSON 檔（list[opinion]，每筆含 target_ticker）算單檔共識。供 CI 無 DB 時使用。"""
    try:
        with open(opinions_json_path, encoding="utf-8") as f:
            all_ops = json.load(f)
    except Exception:
        all_ops = []
    simple = ticker.split(".")[0].upper()
    matched = [
        o for o in all_ops
        if str(o.get("target_ticker", "")).split(".")[0].upper() == simple
    ]
    return score_opinions(matched, sector)


class WeightedConsensusEngine:
    """彙整 podcast 與 custom 輿情，計算加權市場共識分數（DB 後端）。"""

    def __init__(self, db_path: str = str(DATABASE_PATH)):
        self.db_path = db_path

    def get_stock_consensus(self, ticker: str, sector: str = "Technology") -> Dict[str, Any]:
        """從資料庫取近期 opinions 並加權成共識分數。"""
        return score_opinions(self._fetch_all_opinions(ticker), sector)

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
