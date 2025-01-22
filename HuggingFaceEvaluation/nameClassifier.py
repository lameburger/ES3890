from transformers import pipeline

text = "Hugging Face Inc. is based in New York City and was founded by Julien Chaumond and Clement Delangue."

ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
outputs = ner_pipeline(text)

for entity in outputs:
    print(f"Entity: {entity['word']} | Type: {entity['entity']} | Confidence: {entity['score']:.4f}")

# Output:
# Some weights of the model checkpoint at dbmdz/bert-large-cased-finetuned-conll03-english were not used when initializing BertForTokenClassification: ['bert.pooler.dense.bias', 'bert.pooler.dense.weight']
# - This IS expected if you are initializing BertForTokenClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing BertForTokenClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# tokenizer_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 60.0/60.0 [00:00<00:00, 1.33MB/s]
# vocab.txt: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 213k/213k [00:00<00:00, 3.42MB/s]
# Device set to use mps:0
# Entity: Hu | Type: I-ORG | Confidence: 0.9989
# Entity: ##gging | Type: I-ORG | Confidence: 0.9723
# Entity: Face | Type: I-ORG | Confidence: 0.9967
# Entity: Inc | Type: I-ORG | Confidence: 0.9992
# Entity: New | Type: I-LOC | Confidence: 0.9992
# Entity: York | Type: I-LOC | Confidence: 0.9991
# Entity: City | Type: I-LOC | Confidence: 0.9994
# Entity: Julien | Type: I-PER | Confidence: 0.9995
# Entity: Cha | Type: I-PER | Confidence: 0.9993
# Entity: ##um | Type: I-PER | Confidence: 0.9783
# Entity: ##ond | Type: I-PER | Confidence: 0.9874
# Entity: Clement | Type: I-PER | Confidence: 0.9993
# Entity: Del | Type: I-PER | Confidence: 0.9982
# Entity: ##ang | Type: I-PER | Confidence: 0.9930
# Entity: ##ue | Type: I-PER | Confidence: 0.9965