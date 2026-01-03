import json
from pathlib import Path

path = Path("data/raw/polymer_ner/Train.json")

# Read raw text
text = path.read_text(encoding="utf-8").strip()

print("=== First 300 characters of file ===")
print(text[:300])
print("\n")

# Try JSON Lines
print("=== Trying JSON Lines ===")
try:
    first_line = text.splitlines()[0]
    obj = json.loads(first_line)
    print("JSON Lines detected. Keys:", obj.keys())
except Exception as e:
    print("Not JSON Lines:", e)

# Try full JSON
print("\n=== Trying full JSON ===")
try:
    obj = json.loads(text)
    print("Full JSON detected. Type:", type(obj))
except Exception as e:
    print("Not full JSON:", e)
