from fastapi import FastAPI, HTTPException
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, pipeline
from pydantic import BaseModel
import torch # PyTorch is a dependency of transformers
import json # For loading config from file
import os # For path manipulation

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

# Define the path to the config directory
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "deeplearning", "config")
GENERATION_CONFIG_PATH = os.path.join(CONFIG_DIR, "generation_config.json")
CLASSIFICATION_CONFIG_PATH = os.path.join(CONFIG_DIR, "classification_config.json") # New

# Load generation config from file
def load_generation_config():
    if os.path.exists(GENERATION_CONFIG_PATH):
        with open(GENERATION_CONFIG_PATH, 'r') as f:
            return json.load(f)
    # Default config if file not found or error
    return {
        "max_length": 50,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
        "no_repeat_ngram_size": 2,
        "do_sample": True,
        "num_beams": 1 # Added num_beams default
    }

# Load classification config from file (New)
def load_classification_config():
    if os.path.exists(CLASSIFICATION_CONFIG_PATH):
        with open(CLASSIFICATION_CONFIG_PATH, 'r') as f:
            return json.load(f)
    # Default config if file not found or error
    return {
        "model_name_or_path": "distilbert-base-uncased",
        "return_all_scores": False
    }

classification_params_config = load_classification_config() # New
generation_params_config = load_generation_config()

class ClassificationIn(BaseModel): # Renamed from TextIn and updated
    text: str
    return_all_scores: bool = classification_params_config.get("return_all_scores", False)

class GenerationIn(BaseModel):
    prompt: str
    max_length: int = generation_params_config.get("max_length", 50)
    temperature: float = generation_params_config.get("temperature", 0.7)
    top_k: int = generation_params_config.get("top_k", 50)
    top_p: float = generation_params_config.get("top_p", 0.95)
    no_repeat_ngram_size: int = generation_params_config.get("no_repeat_ngram_size", 2)
    do_sample: bool = generation_params_config.get("do_sample", True)
    num_beams: int = generation_params_config.get("num_beams", 1) # Added num_beams

@app.on_event("startup")
async def load_models():
    global classification_model, classification_tokenizer, generation_model, generation_tokenizer
    global text_classifier, text_generator, classification_params_config # Added classification_params_config

    # Load DistilBERT for sequence classification
    cls_model_name = classification_params_config.get("model_name_or_path", "distilbert-base-uncased")
    print(f"Loading classification model: {cls_model_name}")
    try:
        classification_tokenizer = AutoTokenizer.from_pretrained(cls_model_name)
        classification_model = AutoModelForSequenceClassification.from_pretrained(cls_model_name)
        text_classifier = pipeline(
            "sentiment-analysis",
            model=classification_model,
            tokenizer=classification_tokenizer
            # return_all_scores can be passed dynamically during inference
        )
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
        generation_model.config.pad_token_id = generation_model.config.eos_token_id # This is important for beam search

        text_generator = pipeline(
            "text-generation",
            model=generation_model,
            tokenizer=generation_tokenizer
        )
        print(f"Successfully loaded {gen_model_name} and created generation pipeline.")
    except Exception as e:
        print(f"Error loading generation model {gen_model_name}: {e}")

@app.post("/classify/")
async def classify_text(item: ClassificationIn):
    if text_classifier is None:
        return {"error": "Classification model not loaded."}
    try:
        results = text_classifier(item.text, return_all_scores=item.return_all_scores)
        return {"input_text": item.text, "classification_results": results}
    except Exception as e:
        return {"error": f"Error during classification: {str(e)}"}

@app.post("/generate/")
async def generate_text(item: GenerationIn):
    if not text_generator:
        raise HTTPException(status_code=503, detail="Text generation model not loaded")
    
    # Parameters for the pipeline
    generation_args = {
        "max_length": item.max_length,
        "temperature": item.temperature,
        "top_k": item.top_k,
        "top_p": item.top_p,
        "no_repeat_ngram_size": item.no_repeat_ngram_size,
        "do_sample": item.do_sample,
        "num_beams": item.num_beams # Added num_beams
    }
    
    # Ensure pad_token_id is set; pipeline should handle this, but good to be aware
    # For some models, especially when using beam search, pad_token_id must be set.
    # The pipeline usually sets it, but if using model.generate directly, it's crucial.
    # if generation_model.config.pad_token_id is None:
    #     generation_model.config.pad_token_id = generation_tokenizer.eos_token_id

    try:
        # Filter out None values or use pipeline defaults
        # The pipeline handles default values for parameters not provided.
        # We are explicitly passing all, so this is more about ensuring correct types.
        
        # If not sampling, some parameters like temperature, top_k, top_p might not be used or behave differently.
        # Beam search (num_beams > 1) also interacts with do_sample.
        # If do_sample is False and num_beams > 1, it's beam search.
        # If do_sample is True and num_beams > 1, it's beam search with sampling.
        # If do_sample is False and num_beams == 1, it's greedy search.
        
        # The pipeline abstracts much of this, but it's good to be mindful.
        # Forcing pad_token_id for the model directly, as pipeline might not always propagate it as expected for all scenarios
        text_generator.model.config.pad_token_id = text_generator.tokenizer.eos_token_id


        generated_texts = text_generator(item.prompt, **generation_args)
        return {"generated_text": generated_texts[0]["generated_text"]} # Assuming pipeline returns a list of dicts
    except Exception as e:
        print(f"Error during text generation: {e}")
        raise HTTPException(status_code=500, detail=f"Error during text generation: {str(e)}")

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "API is running. Models should be loaded if startup was successful."}

# To run this FastAPI application, you would typically use:
# uvicorn api.main:app --reload
# (Assuming your file is main.py inside an 'api' directory)
