import gradio as gr
from fastai.vision.all import *
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

learn = load_learner('skin_type_model.pkl')
model = learn.model
model.eval()

categories = ['dry', 'normal', 'oily']

tfms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def classify_skin(img):
    img = img.convert('RGB')
    tensor = tfms(img).unsqueeze(0)
    with torch.no_grad():
        preds = torch.softmax(model(tensor), dim=1)[0]
    return dict(zip(categories, map(float, preds)))

app = gr.Interface(
    fn=classify_skin,
    inputs=gr.Image(type='pil'),
    outputs=gr.Label(num_top_classes=3),
    title="Skin Type Classifier",
    description="Upload a face photo to classify skin type. ⚠️ Educational purposes only. Not medical advice."
)

app.launch()
