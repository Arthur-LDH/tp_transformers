import gradio as gr
import requests
import json # For loading default config for Gradio UI
import os

# Define the path to the config directory (relative to this file)
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "deeplearning", "config")
GENERATION_CONFIG_PATH = os.path.join(CONFIG_DIR, "generation_config.json")
CLASSIFICATION_CONFIG_PATH = os.path.join(CONFIG_DIR, "classification_config.json")

# Load default generation params for Gradio UI
def load_default_gen_params():
    if os.path.exists(GENERATION_CONFIG_PATH):
        with open(GENERATION_CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {
        "max_length": 50,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
        "no_repeat_ngram_size": 2,
        "do_sample": True,
        "num_beams": 1  # Default value for UI if not in config
    }

# Load default classification params for Gradio UI
def load_default_class_params():
    if os.path.exists(CLASSIFICATION_CONFIG_PATH):
        with open(CLASSIFICATION_CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {
        "return_all_scores": False
    }

default_gen_params = load_default_gen_params()
default_class_params = load_default_class_params()

def classification(prompt, return_all_scores):
    payload = {"text": prompt, "return_all_scores": return_all_scores}
    response = requests.post("http://localhost:8000/classify/", json=payload)
    return response.json()

def generation(prompt_text, max_length, temperature, top_k, top_p, no_repeat_ngram_size, do_sample, num_beams): # Add num_beams
    payload = {
        "prompt": prompt_text,
        "max_length": int(max_length),
        "temperature": float(temperature),
        "top_k": int(top_k),
        "top_p": float(top_p),
        "no_repeat_ngram_size": int(no_repeat_ngram_size),
        "do_sample": bool(do_sample),
        "num_beams": int(num_beams)  # Add num_beams to payload
    }
    response = requests.post("http://localhost:8000/generate/", json=payload)
    return response.json()

# Page d'accueil
with gr.Blocks() as accueil:
    gr.Markdown("# Bienvenue")

# Page 1 - Classification sentiment
with gr.Blocks() as page1:
    gr.Markdown("# Classification de Sentiment")
    with gr.Row():
        classification_prompt = gr.Textbox(label="Texte à analyser", lines=3, scale=3)
        with gr.Column(scale=1):
            return_all_scores_checkbox = gr.Checkbox(label="Retourner tous les scores", value=default_class_params.get("return_all_scores", False))
    classificiation_result = gr.JSON(label="Résultat") # Changed to JSON for better display of list of dicts
    bouton = gr.Button("Analyser")
    bouton.click(classification, inputs=[classification_prompt, return_all_scores_checkbox], outputs=classificiation_result)

# Page 2 - Génération de texte
with gr.Blocks() as page2:
    gr.Markdown("# Génération de Texte")
    generation_prompt = gr.Textbox(label="Prompt initial", lines=3)
    with gr.Accordion("Paramètres de Génération", open=False):
        max_length_slider = gr.Slider(minimum=10, maximum=500, step=10, label="Max Length", value=default_gen_params.get("max_length", 50))
        temperature_slider = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, label="Temperature", value=default_gen_params.get("temperature", 0.7))
        top_k_slider = gr.Slider(minimum=0, maximum=100, step=1, label="Top K", value=default_gen_params.get("top_k", 50))
        top_p_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label="Top P", value=default_gen_params.get("top_p", 0.95))
        no_repeat_ngram_size_slider = gr.Slider(minimum=0, maximum=5, step=1, label="No Repeat Ngram Size", value=default_gen_params.get("no_repeat_ngram_size", 2))
        num_beams_slider = gr.Slider(minimum=1, maximum=10, step=1, label="Num Beams", value=default_gen_params.get("num_beams", 1)) # Add num_beams slider
        do_sample_checkbox = gr.Checkbox(label="Do Sample", value=default_gen_params.get("do_sample", True))
    
    generation_result = gr.JSON(label="Texte Généré")
    bouton2 = gr.Button("Générer")
    bouton2.click(generation, 
                  inputs=[
                      generation_prompt, 
                      max_length_slider, 
                      temperature_slider, 
                      top_k_slider, 
                      top_p_slider, 
                      no_repeat_ngram_size_slider,
                      do_sample_checkbox,
                      num_beams_slider # Add num_beams_slider to inputs
                  ], 
                  outputs=generation_result)

# Assemblage avec onglets
demo = gr.TabbedInterface(
    [accueil, page1, page2],
    ["Accueil", "Classification de sentiment", "Génération de texte"]
)

demo.launch()