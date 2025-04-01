from datasets import load_dataset
from transformers import AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-7B-Instruct-1M"
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset("spyroot/cornell_sum_movie_dialog")

print(dataset.column_names)

def build_chats(batch):
    message_batch = []
    messages = []
    user_line = True
    old_name = ""
    new_name = ""
    for utt in batch["utterance"]:
        for line in utt["lines"]:
            line_split = line.split('\n')
            if user_line:
                new_name = line_split[0]
                if old_name == "":
                    old_name = new_name
                message_batch.append({"role": "user", "content": "".join(line_split[1:])})
                user_line = False
            else:
                message_batch.append({"role": "assistant", "content": "".join(line_split[1:])})
                user_line = True
                if len(message_batch) == 10 or new_name != old_name:
                    old_name = new_name
                    messages.append(message_batch)
                    message_batch = []

    if len(message_batch) > 0:
        messages.append(message_batch)

    return {"chats": messages}
            

chat_dataset = dataset["train"].map(build_chats, batched=True, remove_columns=dataset["train"].column_names, batch_size=50)

chat_dataset.save_to_disk("cornell_sum_movie_dialog_chat")



TEMPLATE_START = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
def tokenize_chats(batch):
    tokens = []
    mask = []
    for i in range(len(batch["chats"])):
        text = tokenizer.apply_chat_template([batch["chats"][i]], tokenize=False, add_generation_prompt=False)
        text = text.replace(TEMPLATE_START, "")
        cur_tokens = tokenizer.encode(text)
        cur_mask = [1 if batch["chats"][i]["role"] == "assistant" else 0 for _ in range(len(cur_tokens) - 2)] + [0, 0]
        tokens.extend(cur_tokens)
        mask.extend(cur_mask)
    return {"input_ids": tokens, "mask": mask, "labels": tokens[1:] + [tokenizer.eos_token_id]}


chat_dataset = chat_dataset.map(tokenize_chats, batched=False, remove_columns=chat_dataset.column_names)

chat_dataset.save_to_disk("cornell_sum_movie_dialog_chat_tokenized")


chat_dataset.set_format(type="torch", columns=["input_ids", "mask", "labels"])

def mask_labels(chat):
    mask_inverse = (chat["mask"] - 1) * -1
    negative_hundred = mask_inverse * -100
    new_labels = chat["mask"] * chat["labels"] + negative_hundred
    return {"input_ids": chat["input_ids"], "mask": chat["mask"], "labels": chat["labels"], "mask_labels": new_labels}

chat_dataset = chat_dataset.map(mask_labels, batched=False)

print(chat_dataset[0])

chat_dataset = chat_dataset.train_test_split(test_size=0.1)

chat_dataset.save_to_disk("cornell_sum_movie_dialog_chat_tokenized_masked")