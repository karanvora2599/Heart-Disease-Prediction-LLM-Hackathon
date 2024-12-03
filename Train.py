import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForSeq2Seq, Trainer
from datasets import Dataset, load_dataset

device = "cuda"  # Use CUDA if available
model_name = "HuggingFaceTB/SmolLM-135M"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Set a padding token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token

# Preprocessing function
def preprocess(examples):
    # Combine 'instruction' and 'input' for each example in the batch
    combined_inputs = [
        f"{instruction}\n\n{input_text}"
        for instruction, input_text in zip(examples['instruction'], examples['input'])
    ]

    # Tokenize the combined inputs
    inputs = tokenizer(
        combined_inputs,
        truncation=True,
        max_length=512,
        padding="max_length",
    )

    # Tokenize the responses (labels)
    labels = tokenizer(
        examples["response"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

    # Assign labels
    inputs["labels"] = labels["input_ids"]

    # Verify lengths (optional)
    for input_ids, label_ids in zip(inputs["input_ids"], inputs["labels"]):
        assert len(input_ids) == len(label_ids), "Input and label lengths do not match!"

    return inputs

# Load the JSON dataset
dataset = load_dataset("json", data_files="dataset/AlpacaFormat_Master.json")

# Preprocess the dataset
tokenized_dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=["instruction", "input", "response"]
)

# Check tokenized data
# print(dataset['train'][0])

train_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1)["train"]
eval_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1)["test"]

args = TrainingArguments(
    output_dir="SmolLM",
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    eval_strategy="steps",
    eval_steps=250,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=500,
    fp16=True,
    push_to_hub=False,
)

training_args = TrainingArguments(
    output_dir="./results",            # Directory to save model checkpoints and logs
    evaluation_strategy="epoch",      # Evaluate after every epoch
    learning_rate=5e-5,               # Learning rate
    per_device_train_batch_size=8,    # Batch size for training
    per_device_eval_batch_size=8,     # Batch size for evaluation
    num_train_epochs=1,               # Number of epochs
    weight_decay=0.01,                # Weight decay for regularization
    save_strategy="epoch",            # Save model checkpoint every epoch
    logging_dir="./logs",             # Directory for logging
    logging_steps=10,                 # Log every 10 steps
    save_total_limit=2,               # Limit the number of saved checkpoints
    fp16=True,                        # Use mixed precision for faster training
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

trainer.evaluate()


trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")