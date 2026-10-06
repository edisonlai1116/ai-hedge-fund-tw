"""Cross-sectional ranking 系統：同一批股票互相比較，回答「今天哪一支的 forward risk/reward 最好」。

模組（皆為確定性計算；LLM 只做解讀，不決定分數）：
  themes       主題/AI 分類（AI 曝險、需求敏感度、產業基準、估值類別）
  features     價格/量能特徵（point-in-time，可回測）
  fundamentals 品質、成長、財報加速、分類估值（只有「現在」的資料 → 只用於即時評分，不進回測）
  catalysts    近 30 天新聞事件（即時）
  shocks       單日大跌診斷 → POTENTIAL_OVERSOLD
  regime       市場狀態 BULL / NEUTRAL / RISK_OFF / RISK_ON_ROTATION
  scoring      Opportunity Score、狀態、進場區、輪動、部位大小
"""
