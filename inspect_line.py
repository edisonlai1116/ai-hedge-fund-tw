"""Fix corrupted line 318 in simple_signals.py."""

filepath = "app/backend/routes/simple_signals.py"

with open(filepath, encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")

# Show lines 310-325 for context
for i in range(309, 325):
    print(f"{i+1}: {repr(lines[i])}")
