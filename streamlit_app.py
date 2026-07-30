import streamlit as st
import numpy as np
import tensorflow as tf

from generic_model.autoencoder_model import load_autoencoder, get_grad_layer_name
from generic_model.autoencode_figures import plot_image_comparison

from PIL import Image

import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()
image_path = Path(os.getenv("PATH_DATASET"))

category="wood"
output_path = Path(__file__).parent.joinpath("output") / "generic_model" / "64-16-False-True-True-conv_dense-0-mae-mse"
model_file = output_path / f"{category}_autoencoder.keras"

def predict_images(images):
    autoencoder = load_autoencoder(model_file)

    images_to_process = []
    for image in images:
        image = np.array(image.resize((64,64)))
        images_to_process.append(image)
    images_to_process = np.array(images_to_process)
    images_to_process = images_to_process / 255.

    pred = autoencoder.predict(images_to_process)
    mse = tf.reduce_mean(tf.square(images_to_process - pred), axis=(1,2,3))
    fig = plot_image_comparison(images_to_process, None, autoencoder, get_grad_layer_name("conv_dense"))
    return fig

def image_grid(images):
    nb_colonnes=5

    for debut_ligne in range(0, len(images), nb_colonnes):
        colonnes = st.columns(nb_colonnes)

        images_ligne = images[debut_ligne:debut_ligne + nb_colonnes]

        for colonne, image in zip(colonnes, images_ligne):
            with colonne:
                st.image(image, use_container_width=True)

##### PAGE CONTENT #####
if "figure_compare" not in st.session_state:
    st.session_state.figure_compare = None

st.title("Détection d'anomalies")

uploaded_files = st.file_uploader(
    "Upload data", accept_multiple_files=True, type="image"
)
images=[]
for uploaded_file in uploaded_files:
    print("uploaded_file type =", type(uploaded_file))
    image = Image.open(uploaded_file)
    images.append(image)
image_grid(images)

#st.image(image)
if st.button("Analyser l'image", icon=":material/image_search:"):
    if images is not None and len(images) > 0:
        with st.spinner("Génération en cours..."):
            st.session_state.figure_compare = predict_images(images)
    else:
        st.session_state.figure_compare=None

if st.session_state.figure_compare is not None:
    st.pyplot(
        st.session_state.figure_compare
    )