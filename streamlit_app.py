import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from itertools import groupby

from generic_model.autoencoder_model import load_autoencoder, get_grad_layer_name
from generic_model.autoencode_figures import plot_image_comparison

from PIL import Image

import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()
image_path = Path(os.getenv("PATH_DATASET"))

category="wood"
output_path = Path(__file__).parent.joinpath("output") / "generic_model"
model_file = output_path / "64-16-False-True-True-conv_dense-0-mae-mse" / f"{category}_autoencoder.keras"

def predict_images(images, filenames, models_selected, models):
    print("MODEL FILES =", models_selected)

    position = 0
    for indice, groupe in groupby(models_selected):
        taille = len(list(groupe))
        image_batch = images[position:position + taille]
        #print(f"{taille} éléments avec {models.iloc[indice]["category"]}")
        position += taille
        model = models.iloc[indice]

        autoencoder = load_autoencoder(output_path / model["filepath"])

        print("NEW BATCH")
        images_to_process = []
        for image in image_batch:
            image = image.resize((model["resized_dimension"],model["resized_dimension"]))
            if model["grayscale"]:
                print("GRAYSCALE")
                image = image.convert('L')
            else:
                print("COLOR")
                image = image.convert('RGB')
            print("Image shape=", np.array(image).shape)
            images_to_process.append(np.array(image))
        images_to_process = np.array(images_to_process)
        images_to_process = images_to_process / 255.

        print(f"images_to_process shape = {images_to_process.shape}")
        pred = autoencoder.predict(images_to_process)
        axes = tuple(range(1, len(images_to_process.shape)))
        mse = tf.reduce_mean(tf.square(images_to_process - pred), axis=axes)
        fig = plot_image_comparison(images_to_process, filenames, autoencoder, get_grad_layer_name(model["model_type"]))
    return fig

def image_grid(image_batch, filenames, model_names):
    nb_colonnes=8

    for debut_ligne in range(0, len(images), nb_colonnes):
        colonnes = st.columns(nb_colonnes)

        images_ligne = images[debut_ligne:debut_ligne + nb_colonnes]

        colonne_nbr = 0
        for colonne, image in zip(colonnes, images_ligne):
            with colonne:
                st.image(image, width="stretch")
                st.selectbox(
                    "Modèle", 
                    options=range(len(model_names)),
                    format_func=lambda index: model_names[index], 
                    label_visibility="collapsed", 
                    key=f"model_selection_{debut_ligne + colonne_nbr}"
                )
            colonne_nbr +=1

def change_margin_top_selectbox():
    st.markdown(
    """
    <style>
    div[data-testid="stImage"] {
        margin-bottom: -1rem;
    }

    div[data-testid="stSelectbox"] {
        margin-top: 1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def load_model_list(train_result_filename):
    df = pd.read_csv(train_result_filename, sep=",")
    df["batch_size"] = df["batch_size"].astype(int)
    df["resized_dimension"] = df["resized_dimension"].str.extract(r'(\d+)').astype(int)
    df["folder"] = (
        df.iloc[:, 2:11]      # colonnes d'indice 2 à 10 inclus
        .astype(str)
        .agg("-".join, axis=1)
    )
    df["filepath"] = df["folder"] + "/" + df["category"] + "_autoencoder.keras"
    df["model_name"] = df['category'] + "(" + df['model_type'] + ")"

    df = df.sort_values(["category", "roc_auc"], ascending=[True, False])
    best_by_cat = df.loc[df.groupby(["category"])["roc_auc"].idxmax()]
    return best_by_cat

##### PAGE CONTENT #####
st.set_page_config(layout="wide")
change_margin_top_selectbox()

models = load_model_list(output_path / '0_train_results.csv')

if "figure_compare" not in st.session_state:
    st.session_state.figure_compare = None

st.title("Détection d'anomalies")

uploaded_files = st.file_uploader(
    "Envoi des images à analyser", accept_multiple_files=True, type="image"
)
images=[]
filenames = []
for uploaded_file in uploaded_files:
    image = Image.open(uploaded_file)
    images.append(image)
    filenames.append(uploaded_file.name)
image_grid(images, filenames, np.array(models['model_name']))

#st.image(image)
if st.button("Analyser l'image", icon=":material/image_search:"):
    if images is not None and len(images) > 0:
        with st.spinner("Génération en cours..."):
            model_selections=[]
            for i in range(len(images)):
                model_selections.append( st.session_state[f"model_selection_{i}"] )

            #print("Model selections =", model_selections)
            st.session_state.figure_compare = predict_images(images, filenames, models_selected=model_selections, models=models)
    else:
        st.session_state.figure_compare=None

if st.session_state.figure_compare is not None:
    st.pyplot(
        st.session_state.figure_compare
    )