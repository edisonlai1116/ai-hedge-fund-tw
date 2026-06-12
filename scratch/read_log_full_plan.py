import os
import json

log_path = r"C:\Users\User\.gemini\antigravity\brain\223a01c6-2106-4a97-9c20-833d6c2ad4d1\.system_generated\logs\transcript.jsonl"

if os.path.exists(log_path):
    print("Log file exists!")
    found_any = False
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                data = json.loads(line)
                step = data.get('step_index')
                if step is not None and step < 1400:
                    content = str(data)
                    if "derive_today_plan" in content and "simple_signal.py" in content:
                        print(f"Step {step} contains 'simple_signal.py' and 'derive_today_plan'.")
                        idx = content.find("def derive_today_plan(")
                        if idx != -1:
                            print("Found code in content!")
                            print(content[idx:idx+2500])
                            print("="*80)
                            found_any = True
                            break
            except Exception as e:
                pass
    if not found_any:
        print("No earlier step found.")
else:
    print("Log file does not exist.")
