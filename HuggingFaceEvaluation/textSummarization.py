from transformers import pipeline

text = """
The best way to get into the Hank dorm is by taking 18th avanue and then taking a quick left or right into the parking lot. You will then enter the
lobby to be greeted by security which will escort you to your room.
"""

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6") # model
output = summarizer(text, max_length=50, min_length=25, do_sample=False)
print(f"Summary: {output[0]['summary_text']}")

# Output:
# config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.80k/1.80k [00:00<00:00, 4.49MB/s]
# pytorch_model.bin: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.22G/1.22G [00:30<00:00, 40.1MB/s]
# tokenizer_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26.0/26.0 [00:00<00:00, 620kB/s]
# vocab.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 899k/899k [00:00<00:00, 6.13MB/s]
# merges.txt: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 456k/456k [00:00<00:00, 16.7MB/s]
#                                                                                                                                                                                                                       Device set to use mps:0%|█▎                                                                                                                                                        | 10.5M/1.22G [00:00<00:52, 23.2MB/s]
#                                                                                                                                                                                                                       Summary:  The best way to get into the Hank dorm is by taking 18th avanue and then taking a quick left or right into the parking lot . Security will escort you to your room . 