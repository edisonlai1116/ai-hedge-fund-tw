import os
import sys

def safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        try:
            encoding = sys.stdout.encoding or 'utf-8'
            print(msg.encode(encoding, errors='replace').decode(encoding))
        except Exception:
            print(msg.encode('ascii', errors='backslashreplace').decode('ascii'))

src_dir = r"c:\Users\User\Desktop\codex\ai-hedge-fund-main\app\frontend\src"
keywords = ["提示詞", "助手", "Minnie", "Nicholas", "米妮", "尼可拉斯"]

safe_print(f"Scanning directory: {src_dir}")

for root, dirs, files in os.walk(src_dir):
    for file in files:
        if file.endswith((".tsx", ".ts", ".js", ".jsx")):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Check for keywords
                found_kws = []
                for kw in keywords:
                    if kw in content:
                        found_kws.append(kw)
                
                if found_kws:
                    safe_print(f"File: {filepath} matches keywords: {found_kws}")
                    # Let's print the line numbers
                    lines = content.splitlines()
                    for idx, line in enumerate(lines):
                        for kw in keywords:
                            if kw in line:
                                safe_print(f"  Line {idx+1}: {line.strip()[:100]}")
            except Exception as e:
                safe_print(f"Failed to read {filepath}: {e}")
