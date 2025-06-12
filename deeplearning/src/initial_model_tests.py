'''Initial tests for loading Hugging Face Transformer models.'''

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM

def load_distilbert():
    '''Loads the distilbert-base-uncased model and tokenizer.'''
    model_name = "distilbert-base-uncased"
    print(f"Loading model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        print(f"Successfully loaded {model_name}")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None, None

def load_distilgpt2():
    '''Loads the distilgpt2 model and tokenizer.'''
    model_name = "distilgpt2"
    print(f"Loading model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print(f"Successfully loaded {model_name}")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None, None

if __name__ == "__main__":
    print("--- Testing DistilBERT loading ---")
    distilbert_model, distilbert_tokenizer = load_distilbert()
    if distilbert_model and distilbert_tokenizer:
        print(f"DistilBERT model type: {type(distilbert_model)}")
        print(f"DistilBERT tokenizer type: {type(distilbert_tokenizer)}")
        # Add a simple test if needed, e.g., tokenizing a sentence
        # inputs = distilbert_tokenizer("Hello, world!", return_tensors="pt")
        # print(f"Tokenized input for DistilBERT: {inputs}")

    print("\n--- Testing DistilGPT2 loading ---")
    distilgpt2_model, distilgpt2_tokenizer = load_distilgpt2()
    if distilgpt2_model and distilgpt2_tokenizer:
        print(f"DistilGPT2 model type: {type(distilgpt2_model)}")
        print(f"DistilGPT2 tokenizer type: {type(distilgpt2_tokenizer)}")
        # Add a simple test if needed, e.g., tokenizing a sentence
        # inputs = distilgpt2_tokenizer("Once upon a time", return_tensors="pt")
        # print(f"Tokenized input for DistilGPT2: {inputs}")

    print("\nScript finished.")
