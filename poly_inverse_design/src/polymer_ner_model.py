from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

MODEL_NAME = "pranav-s/PolymerNER"

def load_polymer_ner(device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=512)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

    device_arg = -1 if device == "cpu" else 0

    ner_pipe = pipeline(
        task="ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device_arg
    )
    return ner_pipe

def run_ner_on_text(text: str, device: str = "cpu"):
    ner_pipe = load_polymer_ner(device=device)
    return ner_pipe(text)
