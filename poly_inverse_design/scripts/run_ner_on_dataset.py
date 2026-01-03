import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.polymer_ner_dataset import process_json_file

if __name__ == "__main__":
    process_json_file(
        json_filename="Train.json",
        output_csv="ner_train.csv",
        max_docs=50
    )

    process_json_file(
        json_filename="Dev.json",
        output_csv="ner_dev.csv",
        max_docs=50
    )

    process_json_file(
        json_filename="Test.json",
        output_csv="ner_test.csv",
        max_docs=50
    )
