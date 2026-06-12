"""Skill：holdings — 解析持股成本檔（每行 `代號 成本 股數`），輸出 positions artifact（含來源追溯）。

格式（對應專案根目錄的 `股票成本.txt`）：
    alab 120 46
    2330 1000 5
- 代號：英數（美股 AAPL、台股 2330 / 00403A 皆可）。
- 成本：每股平均成本，可含千分位逗號與小數。
- 股數：整數或小數（零股）。
**不靜默丟資料**：非空非註解(#)行若無法解析成 3 欄乾淨持股 → 記入 `skipped` 並 fail-loud（exit≠0），
除非加 `--allow-unparsed`。純標準庫；路徑由 --in/--out 參數化，無硬編碼。
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
import kit  # noqa: E402

DEPS = [{"kind": "python", "min": "3.8"}]
WHO = "holdings"
# 數字：可含千分位逗號與小數（成本/股數共用）。
_NUM = re.compile(r"^-?\d[\d,]*(?:\.\d+)?$")


def _to_num(tok):
    if not _NUM.match(tok or ""):
        return None
    try:
        return float(tok.replace(",", ""))
    except ValueError:
        return None


def run(inp, out, allow_unparsed=False):
    positions, trace, skipped = [], [], []
    with open(inp, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            raw = line.rstrip("\n")
            s = raw.strip()
            if not s or s.startswith("#"):
                continue                                  # 空行/註解：非資料
            parts = s.split()
            cost = _to_num(parts[1]) if len(parts) >= 3 else None
            shares = _to_num(parts[2]) if len(parts) >= 3 else None
            if len(parts) >= 3 and cost is not None and shares is not None:
                ticker = parts[0]
                cost_value = round(cost * shares, 2)
                positions.append({
                    "ticker": ticker,
                    "avg_cost": cost,
                    "shares": shares,
                    "cost_value": cost_value,
                })
                # 來源追溯：成本與股數都溯回輸入行（gate 用 str(value) 比對）。
                trace.append({"value": parts[1], "source": os.path.basename(inp), "locator": f"line {i} ({ticker} avg_cost)"})
                trace.append({"value": parts[2], "source": os.path.basename(inp), "locator": f"line {i} ({ticker} shares)"})
            else:
                skipped.append({"line": i, "text": raw[:100]})

    total_cost_value = round(sum(p["cost_value"] for p in positions), 2)
    data = {
        "positions": positions,
        "position_count": len(positions),
        "total_cost_value": total_cost_value,
        "skipped": skipped,
    }
    kit.write_artifact(kit.artifact("positions@1", WHO, data, trace), out)

    if skipped and not allow_unparsed:
        det = "\n  - ".join(f"line {x['line']}: {x['text']}" for x in skipped[:8])
        print(f"ERROR: [{WHO}] {len(skipped)} 行無法解析為「代號 成本 股數」（已記入 skipped、未丟棄）。"
              f"請修正輸入或加 --allow-unparsed：\n  - {det}", file=sys.stderr)
        raise SystemExit(3)
    if skipped:
        print(f"WARNING: [{WHO}] {len(skipped)} 行無法解析（已記入 skipped；--allow-unparsed 放行）", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=WHO)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--allow-unparsed", action="store_true",
                    help="允許無法解析的行（記入 skipped、不中止）")
    a = ap.parse_args()
    kit.require_deps(DEPS, who=WHO)
    run(a.inp, a.out, a.allow_unparsed)
    print(f"[{WHO}] OK → {a.out}")


if __name__ == "__main__":
    main()
