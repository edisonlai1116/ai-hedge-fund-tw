import os

def search_dir(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    if "def derive_today_plan" in content:
                        print(f"Found in: {path}")
                        # Print signature and first few lines of the function
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if "def derive_today_plan" in line:
                                print(f"  Lines {i+1}-{i+30}:")
                                for j in range(i, min(i+35, len(lines))):
                                    print(f"    {lines[j]}")
                except Exception as e:
                    pass

search_dir('.')
