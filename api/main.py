from fastapi import FastAPI
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, pipeline
from pydantic import BaseModel
import torch # PyTorch is a dependency of transformers

# Global variables to hold the models and tokenizers
# These will be loaded at startup
classification_model = None
classification_tokenizer = None
generation_model = None
generation_tokenizer = None
# Using pipelines for easier inference
text_classifier = None
text_generator = None

app = FastAPI()

class TextIn(BaseModel):
    text: str

class GenerationIn(BaseModel):
    prompt: str
    max_length: int = 50

@app.on_event("startup")
async def load_models():
    global classification_model, classification_tokenizer, generation_model, generation_tokenizer
    global text_classifier, text_generator

    # Load DistilBERT for sequence classification
    cls_model_name = "distilbert-base-uncased"
    print(f"Loading classification model: {cls_model_name}")
    try:
        classification_tokenizer = AutoTokenizer.from_pretrained(cls_model_name)
        classification_model = AutoModelForSequenceClassification.from_pretrained(cls_model_name)
        # Initialize a pipeline for easier use, though the model is not fine-tuned yet
        text_classifier = pipeline("sentiment-analysis", model=classification_model, tokenizer=classification_tokenizer) # Using sentiment-analysis as a placeholder task
        print(f"Successfully loaded {cls_model_name} and created classification pipeline.")
    except Exception as e:
        print(f"Error loading classification model {cls_model_name}: {e}")

    # Load DistilGPT2 for text generation
    gen_model_name = "distilgpt2"
    print(f"Loading generation model: {gen_model_name}")
    try:
        generation_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
        # Ensure pad_token_id is set for open-ended generation if not already present
        if generation_tokenizer.pad_token_id is None:
            generation_tokenizer.pad_token_id = generation_tokenizer.eos_token_id
            print(f"Set pad_token_id to eos_token_id for {gen_model_name} tokenizer.")

        generation_model = AutoModelForCausalLM.from_pretrained(gen_model_name)
        # Set the pad_token_id in the model config as well if using the model directly for generation
        generation_model.config.pad_token_id = generation_model.config.eos_token_id
        
        text_generator = pipeline("text-generation", model=generation_model, tokenizer=generation_tokenizer)
        print(f"Successfully loaded {gen_model_name} and created generation pipeline.")
    except Exception as e:
        print(f"Error loading generation model {gen_model_name}: {e}")


@app.get("/")
async def root():
    return {"message": "Welcome to the Transformers API. Use /classify or /generate endpoints."}

@app.post("/classify/")
async def classify_text(item: TextIn):
    if text_classifier is None:
        return {"error": "Classification model not loaded."}
    try:
        # The model is not fine-tuned, so results might not be meaningful for a specific task yet.
        # The pipeline will attempt to classify based on its pre-training or default head.
        # For distilbert-base-uncased, the default sentiment-analysis pipeline might assume 2 labels (e.g. POSITIVE/NEGATIVE)
        # and the raw model output would be logits for these.
        results = text_classifier(item.text)
        return {"input_text": item.text, "classification_results": results}
    except Exception as e:
        return {"error": f"Error during classification: {str(e)}"}

@app.post("/generate/")
async def generate_text(item: GenerationIn):
    if text_generator is None:
        return {"error": "Generation model not loaded."}
    try:
        # Generate text using the pipeline
        generated_sequences = text_generator(item.prompt, max_length=item.max_length, num_return_sequences=1)
        return {"prompt": item.prompt, "generated_text": generated_sequences[0]['generated_text']}
    except Exception as e:
        return {"error": f"Error during text generation: {str(e)}"}

# To run this FastAPI application, you would typically use:
# uvicorn api.main:app --reload
# (Assuming your file is main.py inside an 'api' directory)
