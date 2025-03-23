
# https://github.com/fastai/fastbook/tree/master
import streamlit as st
import fastbook
#hide
from fastbook import *
import torch
from fastai.vision.all import *
import os

fastbook.setup_book()

from dotenv import load_dotenv

load_dotenv()


from fastai.vision.all import *
path = untar_data(URLs.PETS)/'images'

def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)


def main():
    st.title("Cats or Dogs")
    st.header("Upload an image of a cat or dog and I will tell you which one it is")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        img = PILImage.create(uploaded_file)
        is_cat, _, probs = learn.predict(img)
        st.image(img.to_thumb(128,128), caption=f'Prediction: {is_cat}, Probability: {probs[1].item():.6f}')
        st.write("Probability of being a cat:", probs[1].item())
        st.write("Probability of being a dog:", probs[0].item())

if __name__ == "__main__":
    main()
