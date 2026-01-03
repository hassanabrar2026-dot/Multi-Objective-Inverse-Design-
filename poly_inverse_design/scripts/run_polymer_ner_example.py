import sys
import os

# Add project root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.polymer_ner_model import run_ner_on_text

if __name__ == "__main__":
    text = "Polyethylene has a glass transition temperature of -100 °C"
    output = run_ner_on_text(text, device="cpu")

    print("Input text:")
    print(text)
    print("\nNER output:")
    for ent in output:
        print(ent)
