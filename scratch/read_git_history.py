import os
import subprocess

print("Checking for .git folder...")
if os.path.exists('.git'):
    print(".git folder exists!")
    # Try running git via full path if possible, or just using git commands via subprocess
    try:
        # Check if git is available in standard Windows paths
        git_paths = [
            "git",
            r"C:\Program Files\Git\bin\git.exe",
            r"C:\Program Files (x86)\Git\bin\git.exe",
            r"C:\Users\User\AppData\Local\Programs\Git\bin\git.exe"
        ]
        
        for gp in git_paths:
            try:
                res = subprocess.run([gp, "log", "-n", "3", "--oneline"], capture_output=True, text=True)
                if res.returncode == 0:
                    print(f"Git found at: {gp}")
                    # Show previous version of src/simple_signal.py
                    res_show = subprocess.run([gp, "show", "HEAD:src/simple_signal.py"], capture_output=True, text=True, encoding='utf-8', errors='ignore')
                    if res_show.returncode == 0:
                        print("Successfully read HEAD:src/simple_signal.py!")
                        # Find derive_today_plan in the previous version
                        lines = res_show.stdout.split('\n')
                        found = False
                        for i, line in enumerate(lines):
                            if "def derive_today_plan(" in line:
                                found = True
                                print(f"Found derive_today_plan in history at line {i+1}:")
                                for j in range(i, min(i+40, len(lines))):
                                    print(lines[j])
                                break
                        if not found:
                            print("derive_today_plan not found in HEAD version.")
                    
                    # Try HEAD~1 if not found in HEAD
                    res_show2 = subprocess.run([gp, "show", "HEAD~1:src/simple_signal.py"], capture_output=True, text=True, encoding='utf-8', errors='ignore')
                    if res_show2.returncode == 0:
                        print("Successfully read HEAD~1:src/simple_signal.py!")
                        lines = res_show2.stdout.split('\n')
                        for i, line in enumerate(lines):
                            if "def derive_today_plan(" in line:
                                print(f"Found in HEAD~1 at line {i+1}:")
                                for j in range(i, min(i+40, len(lines))):
                                    print(lines[j])
                                break
                    break
            except Exception as e:
                pass
    except Exception as e:
        print(f"Subprocess git search failed: {e}")
else:
    print(".git folder does not exist.")
