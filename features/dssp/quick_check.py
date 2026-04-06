import os
from pathlib import Path

def count_files(directory):
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

base_dir = Path("结果")
dssp_count = count_files(base_dir / "3714DSSP")
standard_count = count_files(base_dir / "3714标准mmCIF")
experimental_count = count_files(base_dir / "3714实验mmCIF")

total = 3714
print(f"DSSP: {dssp_count}/{total} ({dssp_count/total*100:.1f}%)")
print(f"标准mmCIF: {standard_count}/{total} ({standard_count/total*100:.1f}%)")
print(f"实验mmCIF: {experimental_count}/{total} ({experimental_count/total*100:.1f}%)")
