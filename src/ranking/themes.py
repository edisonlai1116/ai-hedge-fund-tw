"""主題 / AI 分類表。

AI 曝險（ai_exposure）與 AI 需求敏感度（ai_demand_sensitivity）是「專家先驗」，依公司營收結構與
產品定位給定，**不是**由回測擬合出來的參數；因為是用今天的認知判斷，回測時不使用這兩個分數
（避免後見之明），只在即時評分中使用。

valuation_class：
  A High Growth / AI   B Growth   C Mature   D Cyclical   E Financial   F Utility / Power
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ThemeInfo:
    theme: str
    sub_theme: str
    ai_exposure: int               # 0~100：營收與 AI 資本支出的直接關聯程度
    ai_demand_sensitivity: int     # 0~100：AI 需求變化對營收/獲利的彈性
    valuation_class: str           # A~F
    industry_etf: str              # 產業基準
    note: str = ""


THEMES: Dict[str, ThemeInfo] = {
    # ---- AI Compute ----
    "NVDA": ThemeInfo("AI Compute", "GPU", 95, 95, "A", "SMH"),
    "AMD": ThemeInfo("AI Compute", "GPU/CPU", 70, 75, "A", "SMH"),
    "INTC": ThemeInfo("AI Compute", "CPU / foundry", 35, 40, "D", "SMH", "轉型中，獲利循環性高"),
    "ARM": ThemeInfo("AI Compute", "CPU IP", 65, 60, "A", "SMH"),
    "AVGO": ThemeInfo("AI Compute", "custom ASIC / networking", 80, 80, "A", "SMH"),
    "MRVL": ThemeInfo("AI Compute", "custom ASIC / optical DSP", 75, 85, "A", "SMH"),
    "TSM": ThemeInfo("AI Compute", "foundry", 70, 70, "B", "SMH"),
    "QCOM": ThemeInfo("AI Compute", "edge SoC", 25, 30, "C", "SMH"),
    # ---- Memory ----
    "MU": ThemeInfo("Memory", "DRAM / HBM", 70, 85, "D", "SMH"),
    "SNDK": ThemeInfo("Memory", "NAND / SSD", 50, 70, "D", "SMH"),
    "RMBS": ThemeInfo("Memory", "memory interface", 60, 65, "B", "SMH"),
    # ---- Networking ----
    "ANET": ThemeInfo("Networking", "Ethernet switching", 80, 80, "A", "IGV"),
    "CSCO": ThemeInfo("Networking", "switching / routing", 30, 30, "C", "XLK"),
    "CRDO": ThemeInfo("Networking", "AEC / connectivity", 90, 95, "A", "SMH"),
    "ALAB": ThemeInfo("Networking", "PCIe / CXL connectivity", 95, 95, "A", "SMH"),
    "CIEN": ThemeInfo("Optical", "optical networking", 55, 65, "B", "XLK"),
    # ---- Optical ----
    "LITE": ThemeInfo("Optical", "lasers / transceivers", 75, 90, "A", "XLK"),
    "COHR": ThemeInfo("Optical", "transceivers / lasers", 65, 80, "A", "XLK"),
    "FN": ThemeInfo("Optical", "optical manufacturing", 65, 75, "B", "XLK"),
    "AAOI": ThemeInfo("Optical", "transceivers", 80, 95, "A", "XLK"),
    # ---- Storage ----
    "STX": ThemeInfo("Storage", "HDD", 55, 70, "D", "XLK"),
    "WDC": ThemeInfo("Storage", "HDD", 55, 70, "D", "XLK"),
    "NTAP": ThemeInfo("Storage", "enterprise storage", 40, 45, "C", "XLK"),
    "PSTG": ThemeInfo("Storage", "enterprise flash", 45, 50, "B", "XLK"),
    # ---- Power ----
    "VST": ThemeInfo("Power", "nuclear / gas generation", 60, 65, "F", "XLU"),
    "CEG": ThemeInfo("Power", "nuclear generation", 65, 65, "F", "XLU"),
    "NRG": ThemeInfo("Power", "gas generation / retail", 45, 50, "F", "XLU"),
    "TLN": ThemeInfo("Power", "nuclear / gas generation", 65, 70, "F", "XLU"),
    "OKLO": ThemeInfo("Power", "advanced nuclear (pre-revenue)", 70, 80, "A", "XLU", "尚無營收，估值不可用"),
    "GEV": ThemeInfo("Power", "electrical equipment / turbines", 55, 60, "B", "XLI"),
    "ETN": ThemeInfo("Power", "electrical equipment", 45, 45, "B", "XLI"),
    "BE": ThemeInfo("Power", "fuel cells / onsite power", 60, 75, "A", "XLI"),
    "PWR": ThemeInfo("Power", "grid construction", 40, 45, "B", "XLI"),
    # ---- Cooling ----
    "VRT": ThemeInfo("Cooling", "liquid cooling / thermal / power", 80, 80, "A", "XLI"),
    # ---- Semiconductor Equipment ----
    "AMAT": ThemeInfo("Semiconductor Equipment", "wafer fab equipment", 50, 60, "D", "SMH"),
    "LRCX": ThemeInfo("Semiconductor Equipment", "wafer fab equipment", 50, 60, "D", "SMH"),
    "KLAC": ThemeInfo("Semiconductor Equipment", "process control", 50, 55, "D", "SMH"),
    "ASML": ThemeInfo("Semiconductor Equipment", "lithography", 50, 55, "B", "SMH"),
    "TER": ThemeInfo("Semiconductor Equipment", "test", 55, 70, "D", "SMH"),
    # ---- AI Platform / Software ----
    "MSFT": ThemeInfo("AI Platform", "hyperscaler", 60, 50, "B", "XLK"),
    "GOOGL": ThemeInfo("AI Platform", "hyperscaler", 60, 50, "B", "XLC"),
    "AMZN": ThemeInfo("AI Platform", "hyperscaler", 50, 45, "B", "XLY"),
    "META": ThemeInfo("AI Platform", "AI capex / apps", 55, 45, "B", "XLC"),
    "ORCL": ThemeInfo("AI Platform", "cloud infrastructure", 60, 65, "B", "XLK"),
    "DELL": ThemeInfo("AI Compute", "AI servers", 55, 70, "D", "XLK"),
    "SMCI": ThemeInfo("AI Compute", "AI servers", 85, 95, "D", "XLK"),
    "HPE": ThemeInfo("AI Compute", "AI servers / networking", 35, 45, "C", "XLK"),
    "PLTR": ThemeInfo("AI Software", "AI applications", 70, 60, "A", "IGV"),
    "CRWV": ThemeInfo("AI Platform", "GPU cloud", 95, 95, "A", "XLK", "負債高、上市時間短"),
    "NBIS": ThemeInfo("AI Platform", "GPU cloud", 90, 95, "A", "XLK"),
}

# 台股 AI 供應鏈（代號不含 .TW/.TWO）
TW_THEMES: Dict[str, ThemeInfo] = {
    "2330": ThemeInfo("AI Compute", "foundry / CoWoS", 70, 70, "B", "0050.TW"),
    "2454": ThemeInfo("AI Compute", "SoC / ASIC", 45, 55, "B", "0050.TW"),
    "3443": ThemeInfo("AI Compute", "ASIC design service", 80, 90, "A", "0050.TW"),
    "3661": ThemeInfo("AI Compute", "ASIC design service", 85, 90, "A", "0050.TW"),
    "2382": ThemeInfo("AI Compute", "AI servers", 70, 80, "B", "0050.TW"),
    "6669": ThemeInfo("AI Compute", "AI servers", 80, 85, "A", "0050.TW"),
    "2317": ThemeInfo("AI Compute", "AI servers / EMS", 45, 55, "C", "0050.TW"),
    "3231": ThemeInfo("AI Compute", "AI servers", 60, 70, "B", "0050.TW"),
    "2356": ThemeInfo("AI Compute", "servers", 45, 55, "C", "0050.TW"),
    "2376": ThemeInfo("AI Compute", "servers / boards", 45, 55, "D", "0050.TW"),
    "2377": ThemeInfo("AI Compute", "servers / boards", 35, 45, "D", "0050.TW"),
    "3017": ThemeInfo("Cooling", "thermal management", 75, 85, "A", "0050.TW"),
    "3324": ThemeInfo("Cooling", "liquid cooling", 75, 85, "A", "0050.TW"),
    "3653": ThemeInfo("Cooling", "thermal management", 70, 80, "A", "0050.TW"),
    "2308": ThemeInfo("Power", "data center power", 60, 65, "B", "0050.TW"),
    "2301": ThemeInfo("Power", "power supplies", 45, 55, "C", "0050.TW"),
    "2345": ThemeInfo("Networking", "switching", 70, 80, "A", "0050.TW"),
    "2383": ThemeInfo("Networking", "CCL / high-speed materials", 70, 80, "A", "0050.TW"),
    "6274": ThemeInfo("Networking", "CCL / high-speed materials", 70, 80, "A", "0050.TW"),
    "3037": ThemeInfo("Networking", "ABF substrate / PCB", 55, 70, "D", "0050.TW"),
    "2368": ThemeInfo("Networking", "server PCB", 65, 75, "B", "0050.TW"),
    "3044": ThemeInfo("Networking", "PCB", 45, 55, "C", "0050.TW"),
    "2408": ThemeInfo("Memory", "DRAM", 40, 70, "D", "0050.TW"),
    "2344": ThemeInfo("Memory", "DRAM / NOR", 30, 60, "D", "0050.TW"),
    "8299": ThemeInfo("Memory", "NAND controller / SSD", 45, 65, "D", "0050.TW"),
    "3711": ThemeInfo("Semiconductor Equipment", "advanced packaging / test", 55, 65, "D", "0050.TW"),
    "2449": ThemeInfo("Semiconductor Equipment", "test", 55, 65, "B", "0050.TW"),
    "6488": ThemeInfo("Semiconductor Equipment", "wafers", 25, 35, "D", "0050.TW"),
    "3533": ThemeInfo("Networking", "connectors", 55, 65, "B", "0050.TW"),
    "2059": ThemeInfo("AI Compute", "server rails", 65, 75, "B", "0050.TW"),
    "5274": ThemeInfo("AI Compute", "BMC chips", 75, 80, "A", "0050.TW"),
}

SECTOR_ETF_BY_YF_SECTOR = {
    "Technology": "XLK", "Utilities": "XLU", "Energy": "XLE", "Industrials": "XLI",
    "Communication Services": "XLC", "Consumer Cyclical": "XLY", "Financial Services": "XLF",
    "Healthcare": "XLV", "Basic Materials": "XLB", "Consumer Defensive": "XLP", "Real Estate": "XLRE",
}
CYCLICAL_INDUSTRY_KEYWORDS = ("Semiconductor Equipment", "Memory", "Steel", "Oil", "Gas E&P", "Auto", "Chemicals",
                              "Computer Hardware", "Mining")


def theme_of(symbol: str) -> Optional[ThemeInfo]:
    s = symbol.upper()
    if s.endswith((".TW", ".TWO")):
        return TW_THEMES.get(s.split(".")[0])
    return THEMES.get(s)


def classify_valuation(symbol: str, info: Optional[dict] = None) -> str:
    """A~F 估值類別：主題表優先；否則依 yfinance sector/industry 與成長率推定。"""
    t = theme_of(symbol)
    if t:
        return t.valuation_class
    info = info or {}
    sector, industry = info.get("sector") or "", info.get("industry") or ""
    if sector == "Utilities":
        return "F"
    if sector == "Financial Services":
        return "E"
    if any(k.lower() in industry.lower() for k in CYCLICAL_INDUSTRY_KEYWORDS) or sector in ("Energy", "Basic Materials"):
        return "D"
    g = info.get("revenueGrowth")
    if g is not None and g >= 0.25:
        return "A"
    if g is not None and g >= 0.10:
        return "B"
    return "C"


def industry_benchmark(symbol: str, info: Optional[dict] = None) -> str:
    t = theme_of(symbol)
    if t:
        return t.industry_etf
    if symbol.upper().endswith((".TW", ".TWO")):
        return "0050.TW"
    return SECTOR_ETF_BY_YF_SECTOR.get((info or {}).get("sector", ""), "SPY")


def peers_of(symbol: str, universe: List[str]) -> List[str]:
    """同主題的其他股票（產業動能比較用）。"""
    t = theme_of(symbol)
    if not t:
        return []
    return [u for u in universe if u != symbol and (theme_of(u) or ThemeInfo("", "", 0, 0, "C", "")).theme == t.theme]


ACCEPTANCE_TICKERS = ["AVGO", "VST", "CEG", "NRG", "LITE", "BE", "TER", "STX", "AMAT", "RMBS", "CRDO", "VRT", "ARM", "INTC"]
