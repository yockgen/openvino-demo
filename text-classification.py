from optimum.intel import OVModelForSequenceClassification
from transformers import AutoTokenizer, pipeline

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
model = OVModelForSequenceClassification.from_pretrained(model_id, export=True)
model.save_pretrained ("distilbert-base-uncased-finetuned-sst-2-english-ov")

tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
results = pipe ("He is an outlaw.")
print (results)