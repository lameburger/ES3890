from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2") # model
prompt = "The quick brown fox jumped over the"
outputs = generator(prompt, max_length=50, num_return_sequences=1)
print(f"Generated Text: {outputs[0]['generated_text']}")

# Output:
# config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 762/762 [00:00<00:00, 1.23MB/s]
# model.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 353M/353M [00:14<00:00, 23.7MB/s]
# generation_config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 124/124 [00:00<00:00, 487kB/s]
# tokenizer_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26.0/26.0 [00:00<00:00, 205kB/s]
# vocab.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.04M/1.04M [00:00<00:00, 8.23MB/s]
# merges.txt: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 456k/456k [00:00<00:00, 24.7MB/s]
# tokenizer.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.36M/1.36M [00:00<00:00, 25.5MB/s]
# Device set to use mps:0
# Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
# Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
# Generated Text: The quick brown fox jumped over the backseat and then he jumped off the seats in the room and then suddenly dropped onto the floor, then a little guy grabbed his pants off, then he jumped off the seats in the room and then another guy grabbed
# 
# Also, the response is fairly humorous.