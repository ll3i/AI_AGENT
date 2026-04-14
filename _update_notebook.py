import json

# Read the run_solution_gpu.py file
with open("C:/Users/SSAFY/Desktop/AI_AGENT/run_solution_gpu.py", "r", encoding="utf-8") as f:
    py_content = f.read()

# Work line by line to apply the substitution
lines = py_content.splitlines(keepends=True)
result_lines = []
i = 0
replaced = False
while i < len(lines):
    # Check if we're at the 3-line block to replace
    if (i + 2 < len(lines) and
        lines[i] == "# 스크립트 위치 기준으로 작업 디렉토리 설정\n" and
        lines[i+1] == "_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))\n" and
        lines[i+2] == "os.chdir(_SCRIPT_DIR)\n"):
        result_lines.append('os.chdir(r"C:/Users/SSAFY/Desktop/AI_AGENT")\n')
        i += 3
        replaced = True
    else:
        result_lines.append(lines[i])
        i += 1

modified_content = "".join(result_lines)
assert replaced, "ERROR: replacement block not found!"
assert "_SCRIPT_DIR" not in modified_content, "ERROR: _SCRIPT_DIR still present!"
assert 'os.chdir(r"C:/Users/SSAFY/Desktop/AI_AGENT")' in modified_content, "ERROR: new chdir not found!"
print(f"Replacement verified. Total lines: {len(result_lines)}")

# Convert to list of strings (each line ending with \n, except the last)
source_lines = result_lines[:]
if source_lines and source_lines[-1].endswith("\n"):
    source_lines[-1] = source_lines[-1][:-1]

print(f"Source lines count: {len(source_lines)}")
print(f"Lines 13-15 (around replacement): {source_lines[12:15]}")

# Read the notebook
with open("C:/Users/SSAFY/Desktop/AI_AGENT/안경즈팀.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"\nNotebook cells count: {len(nb['cells'])}")
print(f"Cell 0 type: {nb['cells'][0]['cell_type']}")
print(f"Cell 1 type: {nb['cells'][1]['cell_type']}")

# Replace cell index 1 source
nb['cells'][1]['source'] = source_lines

# Write back
with open("C:/Users/SSAFY/Desktop/AI_AGENT/안경즈팀.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\nNotebook written successfully.")

# --- Verification ---
with open("C:/Users/SSAFY/Desktop/AI_AGENT/안경즈팀.ipynb", "r", encoding="utf-8") as f:
    nb2 = json.load(f)

cell1_src = "".join(nb2['cells'][1]['source'])

check1 = "def _b64_to_pil" in cell1_src
check2 = "RESIZE_TO = (512, 512)" in cell1_src
check3 = "[DEBUG]" in cell1_src

print(f"\n--- VERIFICATION ---")
print(f"1. _b64_to_pil present: {check1}")
print(f"2. RESIZE_TO = (512, 512) present: {check2}")
print(f"3. [DEBUG] print statement present: {check3}")
print(f"_SCRIPT_DIR absent: {'_SCRIPT_DIR' not in cell1_src}")
print(f"os.chdir new line present: {'os.chdir(r\"C:/Users/SSAFY/Desktop/AI_AGENT\")' in cell1_src}")

all_ok = check1 and check2 and check3
print(f"\nAll checks passed: {all_ok}")
