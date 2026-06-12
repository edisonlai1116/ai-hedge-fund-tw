import os

log_path = r"C:\Users\User\.gemini\antigravity\brain\223a01c6-2106-4a97-9c20-833d6c2ad4d1\.system_generated\logs\transcript.jsonl"
print(f"Scanning log for all mentions of holding_days_estimate...")

if os.path.exists(log_path):
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f):
            if "holding_days_estimate" in line:
                # Find all occurrences of holding_days_estimate and print their contexts
                idx = 0
                while True:
                    idx = line.find("holding_days_estimate", idx)
                    if idx == -1:
                        break
                    context = line[max(0, idx-100):idx+150]
                    # Check if this context looks like an assignment
                    if "=" in context and "report." not in context and "asdict" not in context:
                        print(f"Line {line_num+1}: context:")
                        print(f"  {context.strip()}")
                        print("-" * 50)
                    idx += len("holding_days_estimate")
else:
    print("Log file does not exist.")
