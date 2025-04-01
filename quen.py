from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

print(torch.cuda.is_available())
model_name = "Qwen/Qwen2.5-7B-Instruct-1M"

CHAT_TEMPLATE = "{%- for message in messages %}\n   {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n     {%- endfor %}\n {%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"


tokenizer_name = "Qwen/Qwen2.5-1.5B-Instruct"

#model_name = "Qwen2.5-1.5B-Instruct-final"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype= torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

messages = [{"role": "system", "content": "You are a medieval bartender named Arthur. Your best friend is the blacksmith named Aldous who owns the shop next to yours. You are very good at making drinks and you love to tell stories. You are very friendly and you love to help people."}]

while True:
    prompt = input("Enter prompt: ")
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)

    messages.append({"role": "assistant", "content": response})
