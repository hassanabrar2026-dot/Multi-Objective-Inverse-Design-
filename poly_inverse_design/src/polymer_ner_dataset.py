import json
import csv
from pathlib import Path
from src.polymer_ner_model import load_polymer_ner

# Path to your dataset folder
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "polymer_ner"


# ------------------------------------------------------------
# 1. Load JSON Lines (NDJSON)
# ------------------------------------------------------------
def load_json(filename: str):
    """
    Load a JSON Lines (NDJSON) file where each line is a JSON object.
    Example line:
    {"words": [...], "ner": [...]}
    """
    path = DATA_DIR / filename
    lines = path.read_text(encoding="utf-8").strip().splitlines()

    data = []
    for line in lines:
        if line.strip():
            data.append(json.loads(line))
    return data


# ------------------------------------------------------------
# 2. Convert list of tokens into readable text
# ------------------------------------------------------------
def words_to_text(words_list):
    """
    Convert list of tokens into a readable sentence.
    Handles spacing around punctuation.
    """
    text = ""
    for w in words_list:
        if w in [".", ",", ")", "]", "}", ":", ";", "!", "?"]:
            text += w
        elif w in ["(", "[", "{"]:
            text += " " + w
        elif w == "-":
            text += "-"
        else:
            text += " " + w
    return text.strip()


# ------------------------------------------------------------
# 3. Extract entities using PolymerNER
# ------------------------------------------------------------
def extract_entities(text: str, ner_pipe):
    ents = ner_pipe(text)
    return [
        {
            "entity_group": e["entity_group"],
            "word": e["word"],
            "start": e["start"],
            "end": e["end"]
        }
        for e in ents
    ]


# ------------------------------------------------------------
# 4. Process a JSON file and save NER results to CSV
# ------------------------------------------------------------
def process_json_file(json_filename: str, output_csv: str, max_docs=None):
    data = load_json(json_filename)
    ner_pipe = load_polymer_ner(device="cpu")

    rows = []
    docs = data if max_docs is None else data[:max_docs]

    for i, entry in enumerate(docs):
        text = words_to_text(entry["words"])
        ents = extract_entities(text, ner_pipe)

        for e in ents:
            rows.append({
                "doc_index": i,
                "entity_group": e["entity_group"],
                "entity_text": e["word"],
                "start": e["start"],
                "end": e["end"],
                "full_text": text
            })

    # Save CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output_csv}  ({len(rows)} rows)")
