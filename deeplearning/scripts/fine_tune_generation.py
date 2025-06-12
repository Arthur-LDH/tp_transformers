import transformers
print(f"[INFO] Using Transformers library version: {transformers.__version__}")
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import os
import math

# Configuration
MODEL_NAME = "distilgpt2"
DATASET_NAME = "wikitext"
DATASET_CONFIG_NAME = "wikitext-2-raw-v1" # Using wikitext-2 for language modeling
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "text_generator_fine_tuned")
LOGGING_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "generation_fine_tuning_logs")
NUM_TRAIN_EPOCHS = 1 # Keep low for quick testing, increase for better results
PER_DEVICE_TRAIN_BATCH_SIZE = 4 # Adjust based on your GPU memory
PER_DEVICE_EVAL_BATCH_SIZE = 4  # Adjust based on your GPU memory
LEARNING_RATE = 2e-5
BLOCK_SIZE = 128 # Context window size for the model

# Set to a smaller number for faster testing, None for full dataset
MAX_TRAIN_SAMPLES = 1000 
MAX_EVAL_SAMPLES = 200

def main():
    print(f"Starting fine-tuning for Causal LM: {MODEL_NAME} on {DATASET_NAME} ({DATASET_CONFIG_NAME})")
    print(f"Output directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)

    # 1. Charger le tokenizer et le modèle
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Set pad_token_id if not present (GPT-2 typically doesn't have a pad token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer.pad_token to tokenizer.eos_token: {tokenizer.eos_token}")

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.pad_token_id # Ensure model config matches

    # 2. Charger et prétraiter le dataset
    print(f"Loading dataset: {DATASET_NAME} ({DATASET_CONFIG_NAME})")
    raw_datasets = load_dataset(DATASET_NAME, DATASET_CONFIG_NAME)

    # Filter out empty lines that might be present in wikitext
    raw_datasets = raw_datasets.filter(lambda example: len(example['text']) > 0)

    def tokenize_function(examples):
        # Tokenize all texts
        return tokenizer(examples['text'])

    print("Tokenizing datasets...")
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["text"])

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= BLOCK_SIZE:
            total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    print(f"Grouping texts into blocks of size {BLOCK_SIZE}...")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        # batch_size=1000, # Adjust batch_size for map if memory issues occur
        # num_proc=4, # Adjust num_proc based on your CPU cores
    )
    
    train_dataset = lm_datasets['train']
    eval_dataset = lm_datasets['validation'] # wikitext-2 has 'validation' instead of 'test' for eval

    if MAX_TRAIN_SAMPLES:
        print(f"Subsetting train dataset to {MAX_TRAIN_SAMPLES} samples.")
        train_dataset = train_dataset.select(range(min(MAX_TRAIN_SAMPLES, len(train_dataset))))
    if MAX_EVAL_SAMPLES:
        print(f"Subsetting eval dataset to {MAX_EVAL_SAMPLES} samples.")
        eval_dataset = eval_dataset.select(range(min(MAX_EVAL_SAMPLES, len(eval_dataset))))

    print(f"Using {len(train_dataset)} samples for training and {len(eval_dataset)} for evaluation.")

    # Data collator for language modeling.
    # It will take care of randomly masking tokens if MLM is set to true (not for Causal LM).
    # For Causal LM, it simply batches the data and creates labels.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 3. Définir les arguments d'entraînement
    print("Defining training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch", # Corrected from evaluation_strategy
        save_strategy="epoch",
        logging_dir=LOGGING_DIR,
        logging_steps=100,
        load_best_model_at_end=True, 
        metric_for_best_model="loss", # Perplexity is often used, loss is a good proxy
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
    )

    # 4. Initialiser le Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        # tokenizer=tokenizer, # Not strictly needed here as data_collator has it
    )

    # 5. Entraîner le modèle
    print("Starting training...")
    train_result = trainer.train()
    trainer.save_model()  # Also saves tokenizer and training_args

    metrics = train_result.metrics
    max_train_samples = MAX_TRAIN_SAMPLES if MAX_TRAIN_SAMPLES else len(train_dataset)
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # 6. Évaluer le modèle
    print("Evaluating model...")
    eval_metrics = trainer.evaluate()
    
    try:
        perplexity = math.exp(eval_metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    eval_metrics["perplexity"] = perplexity
    
    max_eval_samples = MAX_EVAL_SAMPLES if MAX_EVAL_SAMPLES else len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)


    # 7. Sauvegarder le modèle et le tokenizer fine-tunés (already done by trainer.save_model())
    print(f"Fine-tuned model and tokenizer saved to {OUTPUT_DIR}")
    
    print("Fine-tuning for Causal LM complete.")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"To use this model, you can load it with: AutoModelForCausalLM.from_pretrained('{os.path.abspath(OUTPUT_DIR)}') and AutoTokenizer.from_pretrained('{os.path.abspath(OUTPUT_DIR)}')")

if __name__ == "__main__":
    main()
