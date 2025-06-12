import gradio as gr
import requests

def classification(prompt):
    response = requests.post("http://localhost:8000/classify/", json={"text": prompt})
    return response.json()

def generation(prompt_text):
    response = requests.post("http://localhost:8000/generate/", json={"prompt": prompt_text})
    return response.json()

# Page d'accueil
with gr.Blocks() as accueil:
    gr.Markdown("# Bienvenue")

# Page 1 - Classification sentiment
with gr.Blocks() as page1:
    gr.Markdown("# Classification de Sentiment")
    classification_prompt = gr.Textbox(label="prompt")
    classificiation_result = gr.Textbox(label="Résultat")
    bouton = gr.Button("Analyser")
    bouton.click(classification, inputs=classification_prompt, outputs=classificiation_result)

# Page 2 - Génération de texte
with gr.Blocks() as page2:
    gr.Markdown("# Generation de prompt")
    generation_prompt = gr.Textbox(label="prompt")
    generation_result = gr.Textbox(label="Résultat")
    bouton2 = gr.Button("Traiter")
    bouton2.click(generation, inputs=generation_prompt, outputs=generation_result)

# Assemblage avec onglets
demo = gr.TabbedInterface(
    [accueil, page1, page2],
    ["Accueil", "Classification de sentiment", "Génération de texte"]
)

demo.launch()