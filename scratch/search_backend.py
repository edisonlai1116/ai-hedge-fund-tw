import os

target = 'app/backend/routes/simple_signals.py'
with open(target, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "holding_days_estimate" in line or "holding_window" in line:
        print(f"Line {i+1}: {line.strip()}")
        # print 5 lines around it
        start = max(0, i-5)
        end = min(len(lines), i+6)
        for j in range(start, end):
            print(f"  {j+1}: {lines[j].strip()}")
