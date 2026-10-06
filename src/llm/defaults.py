"""預設 LLM：免費的 Google Gemini 3.8 Flash（需 GOOGLE_API_KEY，免費額度即可）。

可用環境變數覆寫：LLM_MODEL、LLM_PROVIDER。
LLM 在本系統只負責「解讀文字」（衝突、情境），不決定任何分數或買賣動作，因此用免費模型即可。
免費額度有每分鐘請求上限 → 呼叫間隔至少 LLM_MIN_INTERVAL_SEC 秒（預設 6.5 秒，約 9 次/分鐘）。
"""
import os

DEFAULT_MODEL_NAME = os.environ.get("LLM_MODEL", "gemini-3.8-flash")
DEFAULT_MODEL_PROVIDER = os.environ.get("LLM_PROVIDER", "Google")
MIN_INTERVAL_SEC = float(os.environ.get("LLM_MIN_INTERVAL_SEC", "6.5"))
