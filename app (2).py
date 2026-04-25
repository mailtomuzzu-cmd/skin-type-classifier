
import gradio as gr
from fastai.vision.all import *

learn = load_learner('skin_type_model.pkl')

categories = ('dry', 'normal', 'oily')

def classify_skin(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

app = gr.Interface(
    fn=classify_skin,
    inputs=gr.Image(type='pil'),
    outputs=gr.Label(num_top_classes=3),
    title="Skin Type Classifier",
    description="Upload a face photo to classify skin type as dry, normal or oily. ⚠️ This app is for educational purposes only. Not a substitute for professional medical advice."
)

app.launch()
