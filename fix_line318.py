"""Fix line 318 in simple_signals.py by replacing the corrupted line with correct UTF-8."""

filepath = "app/backend/routes/simple_signals.py"

# Read as bytes
with open(filepath, "rb") as f:
    raw = f.read()

# Split into lines (bytes)
lines = raw.split(b"\n")
print(f"Total lines: {len(lines)}")
print(f"Line 318 (raw): {lines[317]!r}")

# The line should be the "Default robust hold" return statement.
# From context: it returns ("續抱觀察", "低", "0%", f"...MA50...")
# Reconstruct with correct UTF-8:
correct_line = (
    '    return "續抱觀察", "低", "0%", f"中期上升軌道未被破壞，目前股價在合理波動範圍內，建議守穩關鍵支撐 MA50（{ma50:.2f} 元）防線續抱。"'
).encode("utf-8")

lines[317] = correct_line
fixed = b"\n".join(lines)

# Verify
try:
    decoded = fixed.decode("utf-8")
    print("UTF-8 decode: OK")
    import ast
    ast.parse(decoded)
    print("AST parse: OK")
except Exception as e:
    print(f"Error: {e}")
    import sys; sys.exit(1)

with open(filepath, "wb") as f:
    f.write(fixed)
print("File saved.")
