"""Fix the encoding corruption in simple_signals.py line 318."""
import re

filepath = "app/backend/routes/simple_signals.py"

with open(filepath, "rb") as f:
    raw = f.read()

# The bad sequence is \x83\xbd which should be \xe8\x83\xbd (能)
# The byte 0x83 alone is invalid UTF-8; 0xe8 0x83 0xbd = 能
fixed = raw.replace(b"\x83\xbd", b"\xe8\x83\xbd")

# Verify the fix works
try:
    decoded = fixed.decode("utf-8")
    print("UTF-8 decode after fix: OK")
except UnicodeDecodeError as e:
    print(f"Still broken: {e}")

with open(filepath, "wb") as f:
    f.write(fixed)

print("File written.")

# Show the fixed line
lines = decoded.split("\n")
print(f"Line 318: {lines[317]}")
