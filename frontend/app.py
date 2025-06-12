import gradio as gr
import requests


def classification(prompt):
    response = requests.post("http://localhost:8000/classification", json={"text": prompt})
    return response.json()


def generation(prompt, temperature, top_k, top_p, num_beams):
    payload = {
        "text": prompt,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "num_beams": num_beams
    }
    response = requests.post("http://localhost:8000/generation", json=payload)
    return response.json()

# Page 1 - Classification sentiment
with gr.Blocks() as page1:
    gr.Markdown("# Classification de Sentiment")
    with gr.Row():
        with gr.Column():
            classification_prompt = gr.Textbox(label="Prompt")
            bouton = gr.Button("Analyser")
        with gr.Column():
            classificiation_result = gr.Textbox(label="Résultat")
    bouton.click(classification, inputs=classification_prompt, outputs=classificiation_result)

# Page 2 - Génération de texte
with gr.Blocks() as page2:
    gr.Markdown("# Generation de prompt")

    with gr.Row():
        with gr.Column():
            generation_prompt = gr.Textbox(label="Prompt", lines=3)

            # Paramètres de génération
            with gr.Accordion("Paramètres de génération", open=True):
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature"
                )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Top-K"
                )
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top-P"
                )
                num_beams = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    label="Beam Search"
                )

            bouton2 = gr.Button("Générer")

        with gr.Column():
            generation_result = gr.Textbox(label="Résultat", lines=10)

    bouton2.click(
        generation,
        inputs=[generation_prompt, temperature, top_k, top_p, num_beams],
        outputs=generation_result
    )

# Assemblage avec onglets
demo = gr.TabbedInterface(
    [page1, page2],
    ["Classification de sentiment", "Génération de texte"]
)

demo.launch()