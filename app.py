import gradio as gr
import numpy as np
import os
from src.inference import YearPredictor

# Initialize predictor
MODEL_PATH = "models/best_model.pth"

if os.path.exists(MODEL_PATH):
    predictor = YearPredictor(MODEL_PATH)
else:
    print(f"Warning: Model not found at {MODEL_PATH}. Interface will fail on prediction.")
    predictor = None

def predict_year(image):
    if predictor is None:
        return "Error: Model not loaded."
    
    if image is None:
        return "Please upload an image."
        
    try:
        year = predictor.predict(image)
        return f"{year:.0f}"
    except Exception as e:
        return f"Error: {str(e)}"

# Define custom CSS for a premium look
custom_css = """
.container {
    max-width: 900px;
    margin: auto;
    padding-top: 20px;
}
h1 {
    text-align: center;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    color: #2d3748;
    margin-bottom: 10px;
}
.description {
    text-align: center;
    color: #718096;
    margin-bottom: 30px;
    font-size: 1.1em;
}
.footer {
    text-align: center;
    margin-top: 40px;
    color: #a0aec0;
    font-size: 0.8em;
}
"""

# Create Gradio Interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_classes="container"):
        gr.Markdown("# 📸 PhotoDate AI")
        gr.Markdown("### Discover the vintage of your photos with AI", elem_classes="description")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="numpy", label="Upload a Portrait")
                predict_btn = gr.Button("Estimate Year", variant="primary", size="lg")
            
            with gr.Column():
                output_year = gr.Label(label="Estimated Year", num_top_classes=0)
        
        predict_btn.click(fn=predict_year, inputs=input_image, outputs=output_year)
        
        gr.Markdown("---")
        gr.Markdown("Built with EfficientNetV2 & PyTorch | Trained on the Yearbook Dataset", elem_classes="footer")

if __name__ == "__main__":
    demo.launch()
