import os

def search_dir(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') or file.endswith('.txt') or file.endswith('.md'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    if "holding_days_estimate" in content:
                        print(f"Found 'holding_days_estimate' in: {path}")
                        # Print lines containing it
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if "holding_days_estimate" in line:
                                print(f"  Line {i+1}: {line.strip()}")
                except Exception as e:
                    pass

search_dir('.')
