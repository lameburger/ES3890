from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad") # model
context = "The class that Lane's in is ES3890."
question = "What class is Lane in?"
output = qa_pipeline(question=question, context=context)
print(f"Answer: {output['answer']} | Confidence: {output['score']:.4f}")

# Output:
# config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 451/451 [00:00<00:00, 1.33MB/s]
# model.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 265M/265M [00:10<00:00, 25.4MB/s]
# tokenizer_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 48.0/48.0 [00:00<00:00, 777kB/s]
# vocab.txt: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 232k/232k [00:00<00:00, 3.37MB/s]
# tokenizer.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 466k/466k [00:00<00:00, 10.4MB/s]
# Device set to use mps:0
# Answer: tools for natural language processing | Confidence: 0.6592
# (base) Mack:ES3890 laneburgett$ /opt/anaconda3/bin/python /Users/laneburgett/Desktop/Lane/Vanderbilt/ES3890/HuggingFaceEvaluation/questionAnswering.py
# Device set to use mps:0
# Answer: ES3890 | Confidence: 0.9877