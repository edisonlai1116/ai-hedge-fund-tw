"""
Completely rewrite the _attach_backtest and _build_holding_verdict functions
in simple_signals.py, fixing all encoding corruption.
"""

filepath = "app/backend/routes/simple_signals.py"

with open(filepath, "rb") as f:
    raw = f.read()

text = raw.decode("utf-8")
lines = text.split("\n")

# Lines 254-333 (0-indexed: 253-332) contain the broken functions
# We will replace them with correct UTF-8 content

# Find boundaries
start_idx = 253  # Line 254 (0-indexed)
end_idx = 333    # Line 334 (exclusive, 0-indexed)

# Confirm we have the right range
print(f"Replacing lines {start_idx+1} to {end_idx}")
print(f"First line: {repr(lines[start_idx][:80])}")
print(f"Last line:  {repr(lines[end_idx-1][:80])}")
print(f"Line after: {repr(lines[end_idx][:80])}")

correct_block = '''\
def _attach_backtest(report, frame) -> None:
    pass


def _build_holding_verdict(
    report, cost_basis: float | None
) -> tuple[str, str, str, str]:
    close = report.latest_close
    ma20 = report.ma20
    ma50 = report.ma50
    rsi = report.rsi14
    bias = report.bias
    pnl_pct = None if cost_basis in (None, 0) else ((close / cost_basis) - 1) * 100

    # 多層趨勢驗證：只要收盤價在 MA50 的 1.5% 以內且非偏空，就視為長期多頭
    is_long_term_bullish = (close >= ma50 * 0.985) and (bias != "偏空")

    # 稍微回落至 MA20 以下，但仍在 MA50 防線之上
    is_minor_pullback = (close < ma20) and (close >= ma50 * 0.97)

    if is_long_term_bullish:
        if is_minor_pullback:
            # 長期趨勢強，短線回檔不急著出場
            return (
                "強勢續抱",
                "低",
                "0%",
                f"大趨勢依然健康（收盤貼近 MA50: {ma50:.2f} 元以上），目前只是正常回檔範圍，"
                f"尚未出現三死叉訊號。建議守穩生命線 MA50（{ma50:.2f} 元）續抱。",
            )

        # 長期偏多，僅在 RSI 極度超買時執行獲利了結
        if rsi >= 78 and report.today_exit_action in {"今天賣出", "今天可小量賣"}:
            if pnl_pct is not None and pnl_pct >= 20:
                return (
                    "分批獲利",
                    "中",
                    "30%",
                    f"本波段已進入超買區 (RSI: {rsi:.1f}) 且出現高點，"
                    f"目前獲利豐厚，建議先獲利了結 30% 部位，留倉。",
                )
            return (
                "觀察減碼",
                "中",
                "20%",
                f"本波段技術指標已進入超買 (RSI: {rsi:.1f})，"
                f"建議分批，守住 20% 核心部位等待回調，剩餘續抱 MA50 以上。",
            )

    # 嚴格防線驗證：跌破 MA50 超過 3.5% 才算真正破位
    is_price_broken = close < ma50 * 0.965
    is_structural_break = (close < ma50) and (ma20 < ma50)
    is_defensive_break = is_price_broken or is_structural_break

    # 解析停損價
    stop_val = None
    try:
        if report.stop_loss and isinstance(report.stop_loss, str):
            stop_val = float(report.stop_loss.split("-")[0].strip())
        elif report.stop_loss:
            stop_val = float(report.stop_loss)
    except Exception:
        pass

    is_stop_broken = stop_val is not None and close < stop_val
    is_bearish_break = (bias == "偏空") and (close < ma50)

    if is_defensive_break or is_stop_broken or is_bearish_break:
        # 嚴格防守性賣出
        if pnl_pct is not None and pnl_pct < 0:
            stop_info = f"且觸及防守價（{stop_val:.2f} 元）" if is_stop_broken else ""
            return (
                "停損出場",
                "高",
                "100%",
                f"股價已確實跌破中期防守線 MA50（{ma50:.2f} 元）{stop_info}，"
                f"結構性趨勢已轉走弱。為規避下行風險，建議全面停損，保存實力。",
            )
        return (
            "獲利了結",
            "高",
            "100%",
            f"股價確認跌破中期防守線 MA50（{ma50:.2f} 元）並觸及防守價，"
            f"多頭結構遭到破壞。建議將現有獲利部位全面獲利了結，落袋為安。",
        )

    # 中間情況
    if report.today_exit_action == "今天賣出":
        if pnl_pct is not None and pnl_pct < 0:
            return "今天賣出", "高", "100%", "趨勢已轉弱，若有持股建議今天直接退場。"
        return "今天賣出", "高", "100%", "價格進入高檔獲利區，今天優先落袋。"

    if report.today_exit_action == "今天可小量賣":
        if pnl_pct is not None and pnl_pct >= 20:
            return "分賣一半", "中", "50%", "已有顯著獲利，這裡先賣一半比較穩健。"
        return "分賣三成", "中", "30%", "技術面偏強，先減碼一成，剩下繼續觀察。"

    # 預設：穩健續抱
    return (
        "續抱觀察",
        "低",
        "0%",
        f"中期上升軌道未被破壞，目前股價在合理波動範圍內，"
        f"建議守穩關鍵支撐 MA50（{ma50:.2f} 元）防線續抱。",
    )
'''

new_lines = correct_block.split("\n")

# Replace lines start_idx..end_idx with new_lines
rebuilt = lines[:start_idx] + new_lines + lines[end_idx:]

result = "\n".join(rebuilt)

# Validate
import ast
try:
    ast.parse(result)
    print("AST parse: OK")
except SyntaxError as e:
    print(f"SyntaxError line {e.lineno}: {e.msg}")
    print(repr(e.text))
    import sys; sys.exit(1)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(result)
print("File saved OK.")
