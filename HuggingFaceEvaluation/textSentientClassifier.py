from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english") # model
inputs = ["I love this!", "This is terrible."]
outputs = classifier(inputs)
for i, output in enumerate(outputs):
    print(f"Input: {inputs[i]} | Sentiment: {output['label']} | Score: {output['score']:.4f}")


# Output:
# model.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 268M/268M [00:10<00:00, 25.4MB/s]
# tokenizer_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 48.0/48.0 [00:00<00:00, 1.03MB/s]
# vocab.txt: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 232k/232k [00:00<00:00, 4.19MB/s]
# Device set to use mps:0
# Input: I love this! | Sentiment: POSITIVE | Score: 0.9999
# Input: This is terrible. | Sentiment: NEGATIVE | Score: 0.9996