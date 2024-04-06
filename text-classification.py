from optimum.intel import OVModelForSequenceClassification
from transformers import AutoTokenizer, pipeline
import os


def clear_screen():
    if os.name == 'nt':  # for windows
        _ = os.system('cls')
    else:  # for mac and linux(here, os.name is 'posix')
        _ = os.system('clear')

def main():
    model_id = "distilbert-base-uncased-finetuned-sst-2-english"
    model = OVModelForSequenceClassification.from_pretrained(model_id, export=True)
    model.save_pretrained("distilbert-base-uncased-finetuned-sst-2-english-ov")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    clear_screen()
    while True:
        text = input("Please enter text to classify (or 'q' to quit): ")
        if text.lower() == 'q':
            break
        results = pipe(text)
        print(results)

if __name__ == "__main__":
    main()
