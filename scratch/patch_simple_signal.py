import os

target_file = r'src/simple_signal.py'
with open(target_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Replace score_from_signal
old_score = """def score_from_signal(signal: str) -> int:
    if signal == "偏多":
        return 1
    if signal == "偏空":
        return -1
    return 0"""

new_score = """def score_from_signal(signal: str) -> int:
    if "多" in signal or "Bullish" in signal:
        return 1
    if "空" in signal or "Bearish" in signal:
        return -1
    return 0"""

if old_score in content:
    content = content.replace(old_score, new_score)
    print("SUCCESS: Patched score_from_signal")
else:
    if "def score_from_signal(" in content:
        print("score_from_signal already modified or present.")

# 2. Replace build_agent_opinions
idx_start = content.find("def build_agent_opinions(")
idx_end = content.find("def build_horizon_views(")

if idx_start != -1 and idx_end != -1:
    old_build_opinions = content[idx_start:idx_end]
    new_build_opinions = """def build_agent_opinions(
    frame: pd.DataFrame,
    symbol: str = "AAPL",
    fundamental_score: int = 0,
    graham_number: float | None = None,
    latest_close: float = 0.0,
    forward_pe: float | None = None,
    investingpro_fair_value: float | None = None,
    valuation_gap_pct: float | None = None,
    analyst_target_price: float | None = None,
    warren_ai_momentum: str | None = None,
    ai_chain_layer: str | None = None,
    critical_bottleneck: str | None = None
) -> list[AgentOpinion]:
    latest = frame.iloc[-1]
    prev = frame.iloc[-2]

    close = float(latest["Close"])
    ma10 = float(latest["MA10"])
    ma20 = float(latest["MA20"])
    ma50 = float(latest["MA50"])
    rsi5 = float(latest["RSI5"])
    rsi14 = float(latest["RSI14"])
    atr14 = float(latest["ATR14"])
    support20 = float(frame.tail(20)["Low"].min())
    resistance20 = float(frame.tail(20)["High"].max())
    avg_volume = float(latest["VOL20"]) if not np.isnan(latest["VOL20"]) else float(latest["Volume"])
    volume_ratio = float(latest["Volume"]) / avg_volume if avg_volume else 1.0
    atr_pct = (atr14 / close) * 100 if close else 0.0

    # MACD Indicators
    macd_val = float(latest["MACD"]) if "MACD" in latest else 0.0
    macd_sig = float(latest["MACD_Signal"]) if "MACD_Signal" in latest else 0.0
    macd_hist = float(latest["MACD_Hist"]) if "MACD_Hist" in latest else 0.0
    prev_macd_hist = float(prev["MACD_Hist"]) if "MACD_Hist" in prev else 0.0
    bb_width = float(latest["BB_Width"]) if "BB_Width" in latest else 0.0

    # Trend calculation incorporating MACD crossover
    trend_score = int(close > ma20) + int(ma20 > ma50) + int(float(latest["MA20"]) >= float(prev["MA20"]))
    trend_score += int(macd_val > macd_sig)
    trend_score -= int(close < ma20) + int(ma20 < ma50) + int(macd_val < macd_sig)

    # Momentum incorporating MACD histogram changes
    momentum_score = int(rsi14 > 56) + int(rsi5 > 56) + int(close > float(prev["Close"]))
    momentum_score += int(macd_hist > prev_macd_hist)
    momentum_score -= int(rsi14 < 44) + int(rsi5 < 44) + int(macd_hist < prev_macd_hist)

    # Breakout incorporating Bollinger Width expansions
    breakout_score = int(close >= resistance20 * 0.99) + int(volume_ratio > 1.15) + int(bb_width > 0.15)
    breakout_score -= int(close <= support20 * 1.01) + int(volume_ratio < 0.9)

    mean_reversion_score = int(rsi5 < 35) + int(close < ma10 - 0.7 * atr14)
    mean_reversion_score -= int(rsi5 > 67) + int(close > ma10 + 0.8 * atr14)

    risk_score = int(atr_pct > 4.5) + int(close < ma50) + int(rsi14 < 40)

    macd_desc = "黃金交叉" if macd_val > macd_sig else "死亡交叉"
    bb_desc = "波動擴張" if bb_width > 0.15 else "波動收縮"

    # Master analyst agent signals logic
    gap = valuation_gap_pct if valuation_gap_pct is not None else 0.0

    # 1. Warren Buffett
    buffett_sig = "Neutral"
    buffett_conf = 70
    buffett_sum = "基本面穩健度適中，估值處於合理區間，保持觀望態度。"
    if fundamental_score >= 6 and (graham_number is not None and latest_close <= graham_number * 1.2 or gap >= 5.0):
        buffett_sig = "Bullish"
        buffett_conf = 85
        buffett_sum = f"財務強健 (F-Score: {fundamental_score}/9)，具備足夠的安全邊際與護城河，建議偏多配置。"
    elif fundamental_score <= 3:
        buffett_sig = "Bearish"
        buffett_conf = 90
        buffett_sum = f"財務實力虛弱 (F-Score: {fundamental_score}/9)，缺乏穩定盈餘與護城河，建議迴避。"

    # 2. Charlie Munger
    munger_sig = "Neutral"
    munger_conf = 75
    munger_sum = "業務模式與財務健康度平穩，未見極度突出的高回報率優勢。"
    if fundamental_score >= 7:
        munger_sig = "Bullish"
        munger_conf = 88
        munger_sum = f"典型的超高質量企業 (F-Score: {fundamental_score}/9)，護城河極深，高資本回報率具備強確定性。"
    elif fundamental_score <= 3:
        munger_sig = "Bearish"
        munger_conf = 92
        munger_sum = "企業質量低劣，盈餘波動大且債務結構不佳，完全不符合合理估值買入偉大企業的原則。"

    # 3. Ben Graham
    graham_sig = "Neutral"
    graham_conf = 70
    graham_sum = "價格貼近合理價值，安全邊際不足，保持中性觀察。"
    if (graham_number is not None and latest_close < graham_number) or gap >= 20.0 or (forward_pe is not None and 0 < forward_pe <= 15.0):
        graham_sig = "Bullish"
        graham_conf = 90
        graham_sum = f"股價顯著低於防守價或低本益比 (PE: {forward_pe or 0:.1f})，提供了極佳的安全邊際保護。"
    elif (graham_number is not None and latest_close > graham_number * 1.5) and (forward_pe is not None and forward_pe > 30.0):
        graham_sig = "Bearish"
        graham_conf = 85
        graham_sum = f"估值過度透支 (PE: {forward_pe:.1f}) 且大幅高於資產防守價值，不具備安全邊際。"

    # 4. Aswath Damodaran
    damodaran_sig = "Neutral"
    damodaran_conf = 75
    damodaran_sum = f"最新定額折現模型顯示，當前市場定價已充分反映其內在價值。"
    if gap >= 15.0:
        damodaran_sig = "Bullish"
        damodaran_conf = 85
        damodaran_sum = f"折現現金流與經典乘數模型測算合理價值為 ${investingpro_fair_value or 0:.2f}，安全邊際折價高達 {gap:.1f}%。"
    elif gap <= -15.0:
        damodaran_sig = "Bearish"
        damodaran_conf = 85
        damodaran_sum = f"當前股價高於模型合理價值 (${investingpro_fair_value or 0:.2f}) 約 {abs(gap):.1f}%，估值明顯溢價。"

    # 5. Cathie Wood
    wood_sig = "Neutral"
    wood_conf = 60
    wood_sum = "未見顯著的顛覆性創新或硬核科技增長主線，暫不列為核心追蹤目標。"
    if ai_chain_layer is not None or symbol in ["NVDA", "AAPL", "MSFT", "PLTR", "MU", "2330.TW", "3017.TW"]:
        wood_sig = "Bullish"
        wood_conf = 90
        wood_sum = f"屬於硬核 AI 科技或 AI 基礎設施關鍵鏈條 ({ai_chain_layer or '創新主線'})，具備爆發性長線增長潛力。"
    elif forward_pe is not None and forward_pe < 10.0 and fundamental_score >= 7:
        wood_sig = "Bearish"
        wood_conf = 80
        wood_sum = "屬於傳統低增長週期股，缺乏長線創新高確定性，並非時代顛覆性核心資產。"

    # 6. Nassim Taleb
    taleb_sig = "Neutral"
    taleb_conf = 70
    taleb_sum = "資產負債表與估值水平尚可，未見顯著的反脆弱或脆弱特徵。"
    if fundamental_score >= 6 and (forward_pe is not None and forward_pe <= 20.0):
        taleb_sig = "Bullish"
        taleb_conf = 80
        taleb_sum = "擁有強勁的流動性與防禦性估值，具備顯著的反脆弱抗震能力。"
    elif (forward_pe is not None and forward_pe > 40.0) or fundamental_score <= 3:
        taleb_sig = "Bearish"
        taleb_conf = 95
        taleb_sum = f"高估值 (PE: {forward_pe or 0:.1f}) 或財務極度脆弱，極易受尾部黑天鵝事件衝擊，屬於高風險Fragile標的。"

    # 7. Peter Lynch
    lynch_sig = "Neutral"
    lynch_conf = 70
    lynch_sum = "增長率與本益比匹配度適中，處於中性合理區間。"
    if (forward_pe is not None and 0 < forward_pe <= 25.0) and fundamental_score >= 5:
        lynch_sig = "Bullish"
        lynch_conf = 85
        lynch_sum = f"增長前景良好且估值合理 (PE: {forward_pe:.1f})，是典型的 GARP (合理價格增長) 投資首選。"
    elif forward_pe is not None and forward_pe > 50.0:
        lynch_sig = "Bearish"
        lynch_conf = 80
        lynch_sum = f"前瞻本益比已高達 {forward_pe:.1f} 倍，增長速度難以維持如此高企的估值倍數。"

    # 8. Michael Burry
    burry_sig = "Neutral"
    burry_conf = 65
    burry_sum = "市場情緒與多空力量均衡，未見極端的非對稱套利機會。"
    if gap >= 25.0 or rsi14 < 35.0:
        burry_sig = "Bullish"
        burry_conf = 85
        burry_sum = "股價技術面嚴重超跌或被市場極度恐慌性低估，提供了非對稱的多頭切入契機。"
    elif (forward_pe is not None and forward_pe > 45.0) or rsi14 > 70.0:
        burry_sig = "Bearish"
        burry_conf = 90
        burry_sum = f"市場情緒極度亢奮且估值溢價嚴重 (PE: {forward_pe or 0:.1f})，防範泡沫破裂與高位套牢風險。"

    # 9. Stanley Druckenmiller
    druckenmiller_sig = "Neutral"
    druckenmiller_conf = 70
    druckenmiller_sum = "技術面均線與價格纏繞，未形成明確的單邊趨勢信號。"
    if close > ma20 > ma50:
        druckenmiller_sig = "Bullish"
        druckenmiller_conf = 85
        druckenmiller_sum = "價格站上均線且呈多頭排列，技術動能強勁且資金持續流入，順勢做多。"
    elif close < ma20 < ma50:
        druckenmiller_sig = "Bearish"
        druckenmiller_conf = 85
        druckenmiller_sum = "弱勢空頭排列且跌破生命線，市場缺乏流動性支持，建議順勢看空。"

    # 10. Bill Ackman
    ackman_sig = "Neutral"
    ackman_conf = 70
    ackman_sum = "業務預測性中等，缺乏頂級龍頭獨佔性特徵。"
    if fundamental_score >= 7 and symbol in ["MSFT", "AAPL", "GOOGL", "AMZN", "META", "2330.TW"]:
        ackman_sig = "Bullish"
        ackman_conf = 90
        ackman_sum = "典型的大市值高護城河龍頭，業務極具可預測性且擁有強大的自由現金流生成能力。"
    elif fundamental_score <= 3:
        ackman_sig = "Bearish"
        ackman_conf = 85
        ackman_sum = f"基本面不穩定 (F-Score: {fundamental_score}/9)，業務結構繁雜且可預測性低，缺乏長線配置價值。"

    return [
        AgentOpinion(
            key="trend_agent",
            name="趨勢代理",
            signal=classify_signal(trend_score),
            confidence=confidence_from_score(trend_score),
            summary=f"價格與均線排列顯示 {classify_signal(trend_score)}，MACD呈 {macd_desc}，趨勢延續中。",
        ),
        AgentOpinion(
            key="momentum_agent",
            name="動能代理",
            signal=classify_signal(momentum_score),
            confidence=confidence_from_score(momentum_score),
            summary=f"RSI 與最近價格動能偏 {classify_signal(momentum_score)}，MACD柱狀圖動能偏 {'增強' if macd_hist > prev_macd_hist else '減弱'}。",
        ),
        AgentOpinion(
            key="breakout_agent",
            name="突破代理",
            signal=classify_signal(breakout_score),
            confidence=confidence_from_score(breakout_score, 50),
            summary=f"股價距離 20 日高點不遠，量比 {volume_ratio:.2f} 倍，布林頻寬呈 {bb_desc} ({bb_width:.1%})。",
        ),
        AgentOpinion(
            key="mean_reversion_agent",
            name="回檔承接代理",
            signal=classify_signal(mean_reversion_score),
            confidence=confidence_from_score(mean_reversion_score, 50),
            summary=f"觀察是否出現回檔後可承接的位置，目前偏 {classify_signal(mean_reversion_score)}。",
        ),
        AgentOpinion(
            key="risk_agent",
            name="風險代理",
            signal="偏空" if risk_score >= 2 else "中性",
            confidence=confidence_from_score(risk_score, 60),
            summary=f"ATR 波動率 {atr_pct:.2f}%，風險評估為 {'偏高' if risk_score >= 2 else '可控'}。",
        ),
        AgentOpinion(
            key="warren_buffett",
            name="Warren Buffett (巴菲特)",
            signal=buffett_sig,
            confidence=buffett_conf,
            summary=buffett_sum
        ),
        AgentOpinion(
            key="charlie_munger",
            name="Charlie Munger (蒙格)",
            signal=munger_sig,
            confidence=munger_conf,
            summary=munger_sum
        ),
        AgentOpinion(
            key="ben_graham",
            name="Ben Graham (葛拉漢)",
            signal=graham_sig,
            confidence=graham_conf,
            summary=graham_sum
        ),
        AgentOpinion(
            key="aswath_damodaran",
            name="Aswath Damodaran (達莫達蘭)",
            signal=damodaran_sig,
            confidence=damodaran_conf,
            summary=damodaran_sum
        ),
        AgentOpinion(
            key="cathie_wood",
            name="Cathie Wood (凱薩琳伍德)",
            signal=wood_sig,
            confidence=wood_conf,
            summary=wood_sum
        ),
        AgentOpinion(
            key="nassim_taleb",
            name="Nassim Taleb (塔雷伯)",
            signal=taleb_sig,
            confidence=taleb_conf,
            summary=taleb_sum
        ),
        AgentOpinion(
            key="peter_lynch",
            name="Peter Lynch (彼得林區)",
            signal=lynch_sig,
            confidence=lynch_conf,
            summary=lynch_sum
        ),
        AgentOpinion(
            key="michael_burry",
            name="Michael Burry (麥克貝瑞)",
            signal=burry_sig,
            confidence=burry_conf,
            summary=burry_sum
        ),
        AgentOpinion(
            key="stanley_druckenmiller",
            name="Stanley Druckenmiller (德魯肯米勒)",
            signal=druckenmiller_sig,
            confidence=druckenmiller_conf,
            summary=druckenmiller_sum
        ),
        AgentOpinion(
            key="bill_ackman",
            name="Bill Ackman (艾克曼)",
            signal=ackman_sig,
            confidence=ackman_conf,
            summary=ackman_sum
        ),
    ]

"""
    content = content.replace(old_build_opinions, new_build_opinions)
    print("SUCCESS: Patched build_agent_opinions")
else:
    print("ERROR: Failed to find build_agent_opinions indices")

# 3. Replace compute_rule_score
idx_score_start = content.find("def compute_rule_score(")
idx_score_end = content.find("def enforce_position_value(")

if idx_score_start != -1 and idx_score_end != -1:
    old_rule_score = content[idx_score_start:idx_score_end]
    new_rule_score = """def compute_rule_score(agents: list[dict], horizons: list[dict]) -> tuple[int, str]:
    agent_weights = {
        "trend_agent": 6,
        "momentum_agent": 5,
        "breakout_agent": 3,
        "mean_reversion_agent": 3,
        "risk_agent": 4,
        "warren_buffett": 8,
        "charlie_munger": 8,
        "ben_graham": 7,
        "aswath_damodaran": 7,
        "cathie_wood": 6,
        "nassim_taleb": 7,
        "peter_lynch": 6,
        "michael_burry": 7,
        "stanley_druckenmiller": 6,
        "bill_ackman": 6,
    }
    horizon_weights = {"短線": 4, "中線": 7, "長線": 9}
    score = 50
    for agent in agents:
        score += score_from_signal(agent["signal"]) * agent_weights.get(agent["key"], 4)
    for horizon in horizons:
        score += score_from_signal(horizon["bias"]) * horizon_weights.get(horizon["horizon"], 4)
    final_score = clamp_score(score)
    return final_score, score_to_strength(final_score)

"""
    content = content.replace(old_rule_score, new_rule_score)
    print("SUCCESS: Patched compute_rule_score")
else:
    print("ERROR: Failed to find compute_rule_score indices")

# 4. Replace derive_today_plan
idx_plan_start = content.find("def derive_today_plan(")
idx_plan_end = content.find("def map_ai_chain_and_bottleneck(")

if idx_plan_start != -1 and idx_plan_end != -1:
    old_plan = content[idx_plan_start:idx_plan_end]
    new_plan = """def derive_today_plan(
    latest_close: float,
    ma50: float,
    atr14: float,
    rsi14: float,
    bias: str,
    buy_strength: str,
    buy_zone: str,
    sell_zone: str,
    stop_loss: str,
    candlestick_pattern: str = "無",
    fundamental_score: int = 0,
    valuation_gap_pct: float | None = None,
) -> tuple[str, str, str, str, str, str, float, float, int, str, float]:
    buy_low, buy_high = parse_range(buy_zone)
    sell_low, sell_high = parse_range(sell_zone)
    stop_low, stop_high = parse_range(stop_loss)

    entry_zone = format_range(max(buy_low, latest_close - 0.45 * atr14), min(buy_high, latest_close + 0.15 * atr14))
    entry_mid = range_mid(entry_zone)
    target_mid = range_mid(sell_zone)
    stop_mid = (stop_low + stop_high) / 2
    expected_return_pct = ((target_mid / entry_mid) - 1) * 100 if entry_mid > 0 else 0.0
    risk_pct = ((entry_mid - stop_mid) / entry_mid) * 100 if entry_mid > 0 else 0.0
    reward_ratio = expected_return_pct / risk_pct if risk_pct > 0 else 0.0

    near_buy_zone = latest_close <= buy_high * 1.01
    slightly_extended = latest_close <= buy_high * 1.04 and rsi14 < 63

    # Value investor allowance: if it is heavily undervalued with strong fundamentals, allow buying even in downtrend
    is_strong_value = (fundamental_score >= 5 and valuation_gap_pct is not None and valuation_gap_pct >= 10.0)
    min_return_threshold = 12.0 if is_strong_value else MIN_POSITION_RETURN_PCT

    # Estimate winning probability p
    p_map = {
        "強力買進": 0.64,
        "可分批買進": 0.60,
        "觀察偏多": 0.56,
        "先觀察": 0.52,
        "不建議進場": 0.42
    }
    p = p_map.get(buy_strength, 0.50)

    # Calculate Kelly percentage (Half-Kelly for safety)
    if reward_ratio > 0:
        kelly_raw = p - (1.0 - p) / reward_ratio
        kelly_position_pct = max(0.0, min(0.30, kelly_raw / 2.0))  # Cap at 30% max for single position safety
    else:
        kelly_position_pct = 0.0

    candle_bonus = f" (偵測到K線訊號: {candlestick_pattern})" if candlestick_pattern != "無" else ""

    if (bias == "偏空" and not is_strong_value) or buy_strength == "不建議進場":
        today_action = NO_BUY
        today_note = f"趨勢偏弱，今天不建議新倉進場。{candle_bonus}"
    elif expected_return_pct < min_return_threshold:
        today_action = NO_BUY
        today_note = f"目前可用的報酬空間不足 {min_return_threshold:.1f}%，今天先不要急著買。{candle_bonus}"
    elif near_buy_zone:
        today_action = BUY_NOW
        today_note = f"現價仍在可執行承接區附近，建議直接分批掛單，凱利公式建議倉位 {kelly_position_pct:.1%}。{candle_bonus}"
    elif slightly_extended:
        today_action = BUY_SMALL
        today_note = f"價格略高於理想承接區，若要進場建議先買小量，凱利公式建議倉位 {kelly_position_pct:.1%}。{candle_bonus}"
        entry_zone = format_range(max(latest_close * 0.992, latest_close - 0.35 * atr14), min(latest_close * 1.002, latest_close + 0.05 * atr14))
        entry_mid = range_mid(entry_zone)
        expected_return_pct = ((target_mid / entry_mid) - 1) * 100 if entry_mid > 0 else 0.0
        risk_pct = ((entry_mid - stop_mid) / entry_mid) * 100 if entry_mid > 0 else 0.0
        reward_ratio = expected_return_pct / risk_pct if risk_pct > 0 else 0.0
    else:
        today_action = WAIT_PULLBACK
        today_note = f"目前離理想買點有點遠，更適合等回檔再接。{candle_bonus}"

    sell_mid = (sell_low + sell_high) / 2
    if bias == "偏空" and latest_close < ma50:
        exit_action = SELL_NOW
        exit_zone = format_range(max(latest_close * 0.997, latest_close - 0.2 * atr14), max(latest_close * 1.006, latest_close))
    else:
        exit_action = SELL_SMALL if rsi14 > 68 else HOLD
        exit_zone = sell_zone

    exit_note = f"持股續抱中，目標賣點區設在 {sell_zone}。{candle_bonus}"
    if exit_action == SELL_NOW:
        exit_note = f"技術均線已跌破生命線，建議在現價附近 {exit_zone} 執行全額停損或紀律退場以保全資金。{candle_bonus}"
    elif exit_action == SELL_SMALL:
        exit_note = f"短線 RSI ({rsi14:.1f}) 已進入超買過熱區，建議在 {exit_zone} 分批掛單鎖定利潤。{candle_bonus}"

    return (
        today_action,
        entry_zone,
        today_note,
        exit_action,
        exit_zone,
        exit_note,
        expected_return_pct,
        reward_ratio,
        holding_days_estimate,
        holding_window,
        kelly_position_pct,
    )

"""
    content = content.replace(old_plan, new_plan)
    print("SUCCESS: Patched derive_today_plan")
else:
    print("ERROR: Failed to find derive_today_plan indices")

# 5. Insert early fetch in build_report
target_early_fetch = """    # Call Investing.com data service early to make it available to all master agents
    investing_data = fetch_investing_com_data(symbol, fetch_fundamentals=fetch_fundamentals, close_price=latest_close)
    investingpro_fair_value = investing_data.get("fair_value")
    valuation_gap_pct = investing_data.get("valuation_gap_pct")
    analyst_target_price = investing_data.get("analyst_target")
    warren_ai_momentum = investing_data.get("warren_ai_momentum")
    investingpro_models = investing_data.get("models_breakdown")"""

if "investing_data = fetch_investing_com_data(" not in content:
    trend_up_str = "    trend_up = latest_close > ma20 > ma50"
    if trend_up_str in content:
        content = content.replace(trend_up_str, target_early_fetch + "\n\n" + trend_up_str)
        print("SUCCESS: Patched early fetch in build_report")
    else:
        print("ERROR: Failed to find trend_up for early fetch insertion")
else:
    print("investing_data already fetched early or present.")

# 6. Replace build_agent_opinions call in build_report
old_call = "    agents = [asdict(agent) for agent in build_agent_opinions(frame)]"
new_call = """    agents = [
        asdict(agent) for agent in build_agent_opinions(
            frame=frame,
            symbol=symbol,
            fundamental_score=fundamental_score,
            graham_number=graham_number,
            latest_close=latest_close,
            forward_pe=forward_pe,
            investingpro_fair_value=investingpro_fair_value,
            valuation_gap_pct=valuation_gap_pct,
            analyst_target_price=analyst_target_price,
            warren_ai_momentum=warren_ai_momentum,
            ai_chain_layer=ai_chain_layer,
            critical_bottleneck=critical_bottleneck
        )
    ]"""

if old_call in content:
    content = content.replace(old_call, new_call)
    print("SUCCESS: Patched build_agent_opinions call")
else:
    if "build_agent_opinions(frame=frame" in content or "build_agent_opinions(" in content and "symbol=symbol" in content:
         print("build_agent_opinions call already patched.")
    else:
         print("ERROR: Failed to find old build_agent_opinions call")

# 7. Replace derive_today_plan call in build_report
old_plan_call = """    (
        today_action,
        today_entry_zone,
        today_note,
        today_exit_action,
        today_exit_zone,
        today_exit_note,
        expected_return_pct,
        reward_ratio,
        holding_days_estimate,
        holding_window,
        kelly_position_pct,
    ) = derive_today_plan(
        latest_close,
        ma50,
        atr14,
        rsi14,
        bias,
        buy_strength,
        buy_zone,
        sell_zone,
        stop_loss,
        candlestick_pattern,
    )"""

new_plan_call = """    (
        today_action,
        today_entry_zone,
        today_note,
        today_exit_action,
        today_exit_zone,
        today_exit_note,
        expected_return_pct,
        reward_ratio,
        holding_days_estimate,
        holding_window,
        kelly_position_pct,
    ) = derive_today_plan(
        latest_close=latest_close,
        ma50=ma50,
        atr14=atr14,
        rsi14=rsi14,
        bias=bias,
        buy_strength=buy_strength,
        buy_zone=buy_zone,
        sell_zone=sell_zone,
        stop_loss=stop_loss,
        candlestick_pattern=candlestick_pattern,
        fundamental_score=fundamental_score,
        valuation_gap_pct=valuation_gap_pct,
    )"""

if old_plan_call in content:
    content = content.replace(old_plan_call, new_plan_call)
    print("SUCCESS: Patched derive_today_plan call")
else:
    if "derive_today_plan(" in content and "fundamental_score=fundamental_score" in content:
         print("derive_today_plan call already patched.")
    else:
         print("ERROR: Failed to find old derive_today_plan call")

# 8. Replace duplicate late fetch
old_late_fetch = """    # Call Investing.com data service
    investing_data = fetch_investing_com_data(symbol, fetch_fundamentals=fetch_fundamentals, close_price=latest_close)"""

new_late_fetch = """    # Re-use pre-fetched Investing.com data from early scan
    pass"""

if old_late_fetch in content:
    content = content.replace(old_late_fetch, new_late_fetch)
    print("SUCCESS: Patched duplicate late fetch")
else:
    print("Duplicate late fetch already patched or not found.")

# Write back
with open(target_file, 'w', encoding='utf-8') as f:
    f.write(content)
print("PATCH COMPLETED SUCCESSFULLY!")
