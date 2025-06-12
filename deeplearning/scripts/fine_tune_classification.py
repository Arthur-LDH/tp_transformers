import transformers
print(f"[INFO] Using Transformers library version: {transformers.__version__}")
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import inspect # Import inspect module

# Configuration
MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "imdb" # Using IMDB for sentiment classification (positive/negative)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "sentiment_classifier_fine_tuned")
LOGGING_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "fine_tuning_logs")
NUM_TRAIN_EPOCHS = 1 # Keep low for quick testing, increase for better results
PER_DEVICE_TRAIN_BATCH_SIZE = 8 # Adjust based on your GPU memory
PER_DEVICE_EVAL_BATCH_SIZE = 16 # Adjust based on your GPU memory
LEARNING_RATE = 2e-5
MAX_SAMPLES_TRAIN = 2000 # For faster run during TP, use more (e.g. 25000 for full IMDB train)
MAX_SAMPLES_TEST = 500  # For faster run during TP, use more (e.g. 25000 for full IMDB test)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    print(f"Starting fine-tuning for {MODEL_NAME} on {DATASET_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)

    # Print the source file of TrainingArguments
    try:
        print(f"[INFO] TrainingArguments class loaded from: {inspect.getfile(TrainingArguments)}")
        print(f"[INFO] TrainingArguments module: {TrainingArguments.__module__}")
    except TypeError:
        print("[INFO] Could not determine file for TrainingArguments (possibly a built-in or dynamically generated class).")
    except AttributeError:
        print("[INFO] Could not determine module for TrainingArguments.")


    # 1. Charger le tokenizer et le modèle
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2) # IMDB has 2 labels (pos/neg)

    # 2. Charger et prétraiter le dataset
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)

    # Select a subset for faster training/evaluation if MAX_SAMPLES are set
    train_dataset = dataset['train'].shuffle(seed=42)
    if MAX_SAMPLES_TRAIN:
        train_dataset = train_dataset.select(range(MAX_SAMPLES_TRAIN))
    
    eval_dataset = dataset['test'].shuffle(seed=42)
    if MAX_SAMPLES_TEST:
        eval_dataset = eval_dataset.select(range(MAX_SAMPLES_TEST))

    print(f"Using {len(train_dataset)} samples for training and {len(eval_dataset)} for evaluation.")

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

    print("Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    # Remove columns that the model doesn't expect
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(["text"])
    tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(["text"])
    tokenized_train_dataset = tokenized_train_dataset.rename_column("label", "labels")
    tokenized_eval_dataset = tokenized_eval_dataset.rename_column("label", "labels")
    tokenized_train_dataset.set_format("torch")
    tokenized_eval_dataset.set_format("torch")

    # 3. Définir les arguments d'entraînement
    print("Defining training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        evaluation_strategy="epoch", # Evaluate at the end of each epoch
        save_strategy="epoch",       # Save model at the end of each epoch
        logging_dir=LOGGING_DIR,
        logging_steps=100,           # Log every 100 steps
        load_best_model_at_end=True, # Load the best model found during training
        metric_for_best_model="f1",  # Use f1 score to determine the best model
        report_to="tensorboard",     # Can also use "wandb" if configured
        fp16=torch.cuda.is_available(), # Use mixed precision if CUDA is available
        push_to_hub=False # Do not push to Hugging Face Hub
    )

    # 4. Initialiser le Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer # Pass tokenizer for saving purposes
    )

    # 5. Entraîner le modèle
    print("Starting training...")
    trainer.train()

    # 6. Évaluer le modèle
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # 7. Sauvegarder le modèle et le tokenizer fine-tunés
    print(f"Saving fine-tuned model and tokenizer to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR) # Saves model, tokenizer, and training args
    # tokenizer.save_pretrained(OUTPUT_DIR) # Trainer already saves the tokenizer if provided

    print("Fine-tuning complete.")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"To use this model in your API, update 'model_name_or_path' in 'deeplearning/config/classification_config.json' to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
