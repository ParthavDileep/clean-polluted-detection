import os
from pathlib import Path

base = Path("dataset")

folders = {
    "Train - Clean": base / "train" / "clean",
    "Train - Polluted": base / "train" / "polluted",
    "Val - Clean": base / "val" / "clean",
    "Val - Polluted": base / "val" / "polluted",
}

total = 0
print("=" * 45)
print("       DATASET STATUS CHECK")
print("=" * 45)

for name, path in folders.items():
    if path.exists():
        count = (
            len(list(path.glob("*.jpg")))
            + len(list(path.glob("*.jpeg")))
            + len(list(path.glob("*.png")))
        )
        print(f"  {name:25s} : {count} images")
        total += count
    else:
        print(f"  {name:25s} : Folder Missing")

print("=" * 45)
print(f"  {'TOTAL':25s} : {total} images")
print("=" * 45)

if total == 0:
    print("\nNo images found!")
elif total < 200:
    print("\nLow count - recommend more images")
else:
    print("\nDataset looks good! Ready for Step 3")