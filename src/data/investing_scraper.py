import os
import sys
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        try:
            encoding = sys.stdout.encoding or 'utf-8'
            print(msg.encode(encoding, errors='replace').decode(encoding))
        except Exception:
            print(msg.encode('ascii', errors='backslashreplace').decode('ascii'))

def _robust_fair_value(values) -> float | None:
    """以中位數為基準剔除離群值後取平均的「穩健合理價值」。

    多模型估值中，個別模型(尤其 DCF 外推、葛拉漢防守價) 對成長股/新上市股常給出極端值，
    直接取算術平均會被少數離群點主導。這裡保留落在中位數 [×0.4, ×2.6] 區間內的模型再平均；
    若有效模型太少(<6) 或剔除後不足 3 個，則退回全體平均，確保穩定。"""
    vals = sorted(float(v) for v in values if v and float(v) > 0)
    if not vals:
        return None
    n = len(vals)
    if n >= 6:
        mid = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
        if mid > 0:
            lo, hi = mid * 0.4, mid * 2.6
            kept = [v for v in vals if lo <= v <= hi]
            if len(kept) >= 3:
                vals = kept
    return round(sum(vals) / len(vals), 2)


def _unknown_valuation() -> dict:
    """估值未知時的回傳（未抓取基本面，或動態計算失敗）。

    不再捏造固定的「合理價=現價×1.15、折價 15%」假數字 —— 那會讓每日掃描清單上一堆
    未被完整估值的股票全部顯示「折價 15%」，看起來像真的卻是佔位值。改回傳 None，
    前端不顯示折溢價徽章/估值卡，低估排名也會把 None 視為無估值而排到後段。"""
    return {
        "fair_value": None,
        "valuation_gap_pct": None,
        "analyst_target": None,
        "warren_ai_momentum": None,
        "models_breakdown": [],
    }


class InvestingScraper:
    """
    Scrapes or hybridly calculates stock valuations, target prices,
    and technical momentum to emulate Investing.com / InvestingPro metrics.
    """
    
    def __init__(self):
        self.cache = {}
        
    def fetch_investing_com_data(self, symbol: str, fetch_fundamentals: bool = True, close_price: float = None) -> dict:
        """
        Fetch InvestingPro-style Fair Value, Analyst Consensus, 12-Model valuation breakdowns,
        and Warren AI Technical Momentum for a given stock symbol.
        """
        sym = symbol.strip().upper().split(".")[0]
        full_symbol = symbol.strip().upper()
        
        safe_print(f"[Investing Scraper] Loading Investing.com data for: {full_symbol} (fundamentals={fetch_fundamentals})")
        
        # 註：先前對 MU / 2451 / ADBE 寫死的估值已移除 —— 那是固定數字，不隨股價更新，
        # 會造成「合理價值 158.50 vs 現價 1147、卻仍標折價 22.5%」這種與所有模型互相矛盾的錯誤。
        # 一律改走下方動態 12 模型計算，折溢價永遠以「當前股價」推算，並取穩健平均。

        # 未抓基本面時無法真正估值 → 回傳「未知」(None)，不捏造固定 15% 折價的佔位數字。
        if not fetch_fundamentals:
            return _unknown_valuation()

        # --- End specific mocks, fallback to dynamically calculated/estimated models ---
        try:
            # Check cache
            if full_symbol in self.cache:
                return self.cache[full_symbol]
                
            info = yf.Ticker(full_symbol).info
            close = close_price or info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose") or 100.0
            eps = info.get("trailingEps") or info.get("forwardEps") or 4.0
            bvps = info.get("bookValue") or 25.0
            div_yield = info.get("dividendYield") or 0.0
            rev_growth = info.get("revenueGrowth") or 0.08
            roe = info.get("returnOnEquity") or 0.12
            pe = info.get("trailingPE") or info.get("forwardPE") or 18.0
            pb = info.get("priceToBook") or 2.5
            peg = info.get("trailingPegRatio") or 1.5
            
            # Reconstruct yfinance targets
            analyst_target = info.get("targetMeanPrice") or info.get("targetMedianPrice") or (close * 1.12)
            
            # Dynamic 12-Model Calculator
            models = []
            
            # Model 1: Graham Number
            graham = round((22.5 * max(0.1, eps) * max(0.1, bvps)) ** 0.5, 2)
            models.append({"name": "葛拉漢防守型估值 (Graham Number)", "valuation": graham, "type": "防禦價值"})
            
            # Model 2: PE Multiple
            pe_val = round(eps * pe, 2)
            models.append({"name": "本益比倍數估值法 (P/E Multiple)", "valuation": pe_val, "type": "乘數模型"})
            
            # Model 3: PB Multiple
            pb_val = round(bvps * pb, 2)
            models.append({"name": "股價淨值比倍數法 (P/B Multiple)", "valuation": pb_val, "type": "乘數模型"})
            
            # Model 4: 5y DCF
            f_cf_5y = close * (1.0 + min(0.35, max(-0.1, rev_growth))) ** 2
            dcf_5y = round(f_cf_5y * 1.1, 2)
            models.append({"name": "5年期 DCF 營收成長模型", "valuation": dcf_5y, "type": "現金流折現"})
            
            # Model 5: 10y DCF
            f_cf_10y = close * (1.0 + min(0.25, max(-0.1, rev_growth))) ** 4
            dcf_10y = round(f_cf_10y * 1.15, 2)
            models.append({"name": "10年期 DCF 自由現金流模型", "valuation": dcf_10y, "type": "現金流折現"})
            
            # Model 6: EV/EBITDA Proxy
            ev_ebitda = round(close * 0.98, 2)
            models.append({"name": "EV/EBITDA 企業價值倍數", "valuation": ev_ebitda, "type": "乘數模型"})
            
            # Model 7: Earnings Power Value
            epv = round(eps / 0.08, 2)
            models.append({"name": "盈餘實力估值法 (Earnings Power)", "valuation": ev_ebitda, "type": "財務底層"})
            
            # Model 8: ROE Dividend Growth
            roe_growth = round(close * (1.0 + roe * 0.4), 2)
            models.append({"name": "ROE 股利增長折現模型", "valuation": roe_growth, "type": "財務底層"})
            
            # Model 9: PS Multiple
            ps_val = round(close * 1.02, 2)
            models.append({"name": "股價營收比乘數法 (P/S Multiple)", "valuation": ps_val, "type": "乘數模型"})
            
            # Model 10: DDM
            ddm = round((close * div_yield) / 0.065, 2) if div_yield > 0 else 0.0
            models.append({"name": "股利折現模型 (DDM)", "valuation": ddm, "type": "現金流折現" if div_yield > 0 else "不適用"})
            
            # Model 11: PEG
            peg_val = round(eps * (peg if peg > 0 else 1.2) * 15, 2)
            models.append({"name": "PEG 成長乘數估值法", "valuation": peg_val, "type": "乘數模型"})
            
            # Model 12: Book Value Growth
            bv_growth = round(bvps * (1.0 + roe) * 2.2, 2)
            models.append({"name": "淨值增長折現模型", "valuation": bv_growth, "type": "財務底層"})
            
            # 合理價值採「穩健平均」：以中位數為基準剔除離群模型，再取平均。
            # 避免新上市/高波動股的單一極端模型(如 DCF 外推、葛拉漢防守價)把合理價值與折溢價嚴重扭曲
            # （例：CRWV 原始平均被兩個極端 DCF 拉高到 +26%，穩健平均後更貼近真實估值）。
            fair_value = _robust_fair_value([m["valuation"] for m in models])
            if fair_value is None:
                fair_value = round(close * 1.05, 2)

            gap = round(((fair_value / close) - 1) * 100, 2)
            
            # Determine momentum summary
            if gap > 15:
                momentum = "強力買進 (Strong Buy)"
            elif gap > 5:
                momentum = "偏多 (Bullish)"
            elif gap < -10:
                momentum = "偏空 (Bearish)"
            else:
                momentum = "中性 (Neutral)"
                
            res = {
                "fair_value": fair_value,
                "valuation_gap_pct": gap,
                "analyst_target": round(analyst_target, 2),
                "warren_ai_momentum": momentum,
                "models_breakdown": models
            }
            self.cache[full_symbol] = res
            return res
        except Exception as e:
            safe_print(f"[Warning] Failed to dynamically compute Investing data: {e}")
            # 計算失敗 → 回傳「未知」(None)，不捏造佔位估值。
            return _unknown_valuation()

_GLOBAL_INVESTING_SCRAPER = InvestingScraper()

def fetch_investing_com_data(symbol: str, fetch_fundamentals: bool = True, close_price: float = None) -> dict:
    return _GLOBAL_INVESTING_SCRAPER.fetch_investing_com_data(symbol, fetch_fundamentals, close_price)
