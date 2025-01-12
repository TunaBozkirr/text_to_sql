from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq

# Load the Spider dataset
dataset = load_dataset("spider")

# Load the model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Preprocessing function
def preprocess_function(examples):
    inputs = examples["question"]
    targets = examples["query"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)  # Reduced max_length
    labels = tokenizer(targets, max_length=128, truncation=True).input_ids
    model_inputs["labels"] = labels
    return model_inputs

# Tokenize dataset and use a smaller subset
small_train = dataset["train"].shuffle(seed=42).select(range(1000))  # Use only 1000 samples for training
small_val = dataset["validation"].shuffle(seed=42).select(range(200))  # Use only 200 samples for validation
tokenized_datasets = {
    "train": small_train.map(preprocess_function, batched=True),
    "validation": small_val.map(preprocess_function, batched=True)
}

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# Training arguments optimized for low resources
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,  # Higher learning rate for fewer epochs
    per_device_train_batch_size=4,  # Smaller batch size
    per_device_eval_batch_size=4,
    num_train_epochs=2,  # Fewer epochs
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    logging_dir="./logs",
    gradient_accumulation_steps=4,  # Simulate larger batch size
    dataloader_num_workers=2,  # Fewer workers to reduce memory usage
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the model
OUTPUT_DIR = "./trained_model"
print("Saving the model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training completed and model saved!")

