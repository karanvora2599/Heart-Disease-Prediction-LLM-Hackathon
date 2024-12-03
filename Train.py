import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer,
    set_seed
)
from datasets import load_dataset
import evaluate  # Use the 'evaluate' library instead of 'datasets.load_metric'
import wandb  # Import wandb

# Initialize wandb
wandb.init(
    project="SmolLM-FineTuning",
    config={
        "model_name": "HuggingFaceTB/SmolLM-135M",
        "batch_size": 8,
        "learning_rate": 5e-5,
        "epochs": 3,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 8,
        "max_length": 512
    },
    name="SmolLM-FineTuning-Run",  # Optional: Name your wandb run
)

# Set a seed for reproducibility
set_seed(42)

device = "cuda:0"  # Use CUDA if available
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

# Split the dataset into training and evaluation sets
train_test_split = tokenized_dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Define TrainingArguments with wandb integration and updated parameters
training_args = TrainingArguments(
    output_dir="./results",                      # Directory to save model checkpoints and logs
    eval_strategy="steps",                       # Updated from 'evaluation_strategy'
    eval_steps=250,                              # Number of steps between evaluations
    learning_rate=5e-5,                          # Learning rate
    per_device_train_batch_size=8,               # Batch size for training
    per_device_eval_batch_size=8,                # Batch size for evaluation
    num_train_epochs=3,                          # Number of epochs
    weight_decay=0.01,                           # Weight decay for regularization
    save_strategy="steps",                       # Save checkpoint every `save_steps`
    save_steps=500,                              # Number of steps between saves
    save_total_limit=2,                          # Limit the total amount of checkpoints
    logging_dir="./logs",                        # Directory for logging
    logging_steps=10,                            # Log every 10 steps
    gradient_accumulation_steps=8,               # Accumulate gradients over 8 steps
    fp16=True,                                   # Use mixed precision
    report_to=["wandb"],                         # Enable wandb reporting
    run_name="SmolLM-FineTuning",                # Name of the wandb run
    dataloader_num_workers=4,                    # Number of subprocesses for data loading
    load_best_model_at_end=True,                 # Load the best model when finished training
    metric_for_best_model="perplexity",           # Use perplexity to evaluate the best model
    greater_is_better=False                      # Lower perplexity is better
)

# Initialize the data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# Load the metric using the 'evaluate' library
perplexity_metric = evaluate.load("perplexity")

# Define evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Shift logits and labels for perplexity computation
    # Note: This is a simplified example; adjust based on your specific use case
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    # Compute perplexity
    perplexity = perplexity_metric.compute(predictions=predictions, references=labels)
    return {"perplexity": perplexity["perplexity"]}

# Initialize the Trainer with wandb integration
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,  # Add your metrics here
)

# Start training
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save the fine-tuned model and tokenizer
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Finish the wandb run
wandb.finish()