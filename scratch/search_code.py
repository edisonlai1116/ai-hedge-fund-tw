import os

target_file = 'src/simple_signal.py'
with open(target_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "holding_days_estimate" in line or "holding_window" in line:
        print(f"Line {i+1}: {line.strip()}")
