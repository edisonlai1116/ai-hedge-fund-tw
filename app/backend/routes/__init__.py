import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)

# Main API router
api_router = APIRouter()


def _include(module_name: str, **kwargs) -> None:
    """逐一載入子路由並容錯：某個路由若缺依賴（例如免 Key 部署沒裝 langchain，
    hedge_fund 會 import 失敗）就跳過並記錄，不讓整個 routes 套件 import 失敗。
    有裝齊依賴時行為與原本完全一致。"""
    try:
        mod = __import__(f"app.backend.routes.{module_name}", fromlist=["router"])
        api_router.include_router(mod.router, **kwargs)
    except Exception as e:
        logger.warning(f"[routes] 略過 '{module_name}'（缺依賴或載入失敗）：{e}")


_include("health", tags=["health"])
_include("hedge_fund", tags=["hedge-fund"])
_include("storage", tags=["storage"])
_include("flows", tags=["flows"])
_include("flow_runs", tags=["flow-runs"])
_include("ollama", tags=["ollama"])
_include("language_models", tags=["language-models"])
_include("api_keys", tags=["api-keys"])
_include("simple_signals")
_include("sentiment")
