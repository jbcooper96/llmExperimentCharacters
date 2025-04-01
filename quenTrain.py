from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available())
model_name = "Qwen/Qwen2.5-1.5B-Instruct"


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype= torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.to(device)

print(type(model).__name__)

dataset = load_from_disk("./cornell_sum_movie_dialog_chat_tokenized_masked")

def collate_fn(batch):
    max_len = 0
    for item in batch: 
        if item["input_ids"].shape[0] > max_len:
            max_len = item["input_ids"].shape[0]
    
    for item in batch: 
        padding_len = max_len - item["input_ids"].shape[0]
        item["input_ids"] = torch.nn.functional.pad(item["input_ids"], (0, padding_len), value=tokenizer.pad_token_id)
        item["labels"] = torch.nn.functional.pad(item["labels"], (0, padding_len), value=-100)


    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch])
    }

dataset = dataset.with_format("torch", device=torch.device("cpu"))
dataloader = DataLoader(dataset["train"], batch_size=5, shuffle=True, collate_fn=collate_fn)
testLoader = DataLoader(dataset["test"], batch_size=5, shuffle=False, collate_fn=collate_fn)

LEARNING_RATE = 1e-5
EPOCHS = 10 

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)



for epoch in range(EPOCHS): 
    model.train()
    for batch_number, batch in enumerate(dataloader): 
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        optimizer.zero_grad()

        outputs = model(input_ids=input_ids.to(device), labels=labels.to(device))
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        if batch_number % 100 == 0: 
            print(f"Epoch: {epoch}, Batch: {batch_number}, Loss: {loss.item()}")

    if epoch % 5 == 0: 
        model.save_pretrained(f"Qwen2.5-1.5B-Instruct-{epoch}")

    print("Epoch: ", epoch, " completed")
    total_loss = 0
    batch_count = 0
    model.eval()
    for batch_number, batch in enumerate(testLoader): 
        
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        with torch.no_grad():
            outputs = model(input_ids=input_ids.to(device), labels=labels.to(device))
        loss = outputs.loss
        total_loss += loss.item()
        batch_count += 1

    print(f"Test Loss: {total_loss / batch_count}")

model.save_pretrained(f"Qwen2.5-1.5B-Instruct-final")