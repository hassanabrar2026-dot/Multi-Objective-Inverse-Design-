import random
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOCAB_PATH = PROJECT_ROOT / "data" / "processed" / "polymer_vocab.csv"

def build_ngram_model(words, n=3):
    model = {}
    for word in words:
        padded = "~" * (n - 1) + word + "~"
        for i in range(len(padded) - n + 1):
            gram = padded[i:i+n-1]
            next_char = padded[i+n-1]
            model.setdefault(gram, []).append(next_char)
    return model

def generate_word(model, n=3, max_len=40):
    gram = "~" * (n - 1)
    result = ""
    for _ in range(max_len):
        next_chars = model.get(gram, None)
        if not next_chars:
            break
        next_char = random.choice(next_chars)
        if next_char == "~":
            break
        result += next_char
        gram = gram[1:] + next_char
    return result

if __name__ == "__main__":
    df = pd.read_csv(VOCAB_PATH)
    polymer_list = df["polymer_name"].tolist()

    model = build_ngram_model(polymer_list, n=3)

    generated = []
    for _ in range(200):
        name = generate_word(model, n=3)
        if len(name) > 3:
            generated.append(name)

    out_path = PROJECT_ROOT / "data" / "processed" / "generated_polymers.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for name in generated:
            f.write(name + "\n")

    print(f"Generated {len(generated)} polymer candidates")
    print("Examples:")
    for g in generated[:20]:
        print("  ", g)
