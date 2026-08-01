import streamlit as st
import gc
import os

# TensorFlow doit voir cet env var avant son import. L'allocateur par defaut
# garde souvent la memoire GPU dans un pool interne au processus Streamlit.
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import numpy as np
import pandas as pd
import tensorflow as tf
from itertools import groupby

from generic_model.autoencoder_model import load_autoencoder, get_grad_layer_name
from generic_model.autoencode_figures import plot_image_comparison

from PIL import Image
import re

from dotenv import load_dotenv
from pathlib import Path
load_dotenv()
image_path = Path(os.getenv("PATH_DATASET"))

category="wood"
output_path = Path(__file__).parent.joinpath("output") / "generic_model"

for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

def predict_images(images, filenames, models_selected, models):
    print("MODEL FILES =", models_selected)

    figs=[]
    position = 0
    for indice, groupe in groupby(models_selected):
        if indice is not None:
            fig = None
            taille = len(list(groupe))
            image_batch = images[position:position + taille]
            #print(f"{taille} éléments avec {models.iloc[indice]["category"]}")
            position += taille
            model = models.iloc[indice]

            autoencoder = None
            images_to_process = None
            pred = None
            try:
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
                    #print("Image shape=", np.array(image).shape)
                    images_to_process.append(np.array(image))
                images_to_process = np.array(images_to_process, dtype=np.float32) / 255.0

                #print(f"images_to_process shape = {images_to_process.shape}")
                #pred = autoencoder.predict(images_to_process, batch_size=1, verbose=0)
                pred = autoencoder(
                    tf.convert_to_tensor(images_to_process),
                    training=False,
                ).numpy()

                if images_to_process.ndim == 3:
                    images_to_process = images_to_process[..., np.newaxis]
                if pred.ndim == 3:
                    pred = pred[..., np.newaxis]

                ### classification
                # Calcul de l'erreur de chaque image
                #axes = tuple(range(1, len(images_to_process.shape)))
                # if model['error_score'] == 'mse':
                #     error_values = tf.reduce_mean(tf.square(images_to_process - pred), axis=axes)
                # else:
                #     error_values = tf.reduce_mean(tf.abs(images_to_process - pred), axis=axes)
                # Récupération du seuil
                threshold = load_threshold(output_path / model["folder"], model["category"])

                fig = plot_image_comparison(
                    images_to_process,
                    filenames[position - taille:position],
                    autoencoder,
                    error_score = model["error_score"], 
                    threshold=threshold, 
                    grad_layer_name=get_grad_layer_name(model["model_type"]),
                    encoded_images=pred,
                )

                figs.append(fig)
            finally:
                del autoencoder, images_to_process, pred
                gc.collect()
                tf.keras.backend.clear_session()
                gc.collect()
    return figs

def detect_category(filename, model_names):
    categories= list(map(lambda s: s.split(' ')[0], model_names))
    index = next((i for i, cat in enumerate(categories) if cat in filename), None)
    return index

def image_grid(image_batch, filenames, model_names):
    nb_colonnes=8

    for debut_ligne in range(0, len(image_batch), nb_colonnes):
        colonnes = st.columns(nb_colonnes)

        images_ligne = image_batch[debut_ligne:debut_ligne + nb_colonnes]

        colonne_nbr = 0
        for colonne, image in zip(colonnes, images_ligne):
            with colonne:
                st.image(image, width="stretch")
                cat_index = detect_category(filenames[debut_ligne+colonne_nbr], model_names)
                st.selectbox(
                    "Modèle", 
                    options=range(len(model_names)),
                    format_func=lambda index: model_names[index], 
                    label_visibility="collapsed", 
                    key=f"model_selection_{debut_ligne + colonne_nbr}",
                    index=cat_index,
                    placeholder="Select model...",
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
    df["model_name"] = df['category'] + " (" + df['model_type'] + ")"

    df = df.sort_values(["category", "roc_auc"], ascending=[True, False])
    best_by_cat = df.loc[df.groupby(["category"])["roc_auc"].idxmax()]
    return best_by_cat

def load_threshold(model_path, category):
    #print(f"CALLING with {model_path} and {category}")
    with open(model_path / f"{category}_classification_report.txt", "r") as f:
        line = f.readline()
    #print("Line read:", line)
    threshold = re.search(r'\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)', line)
    if threshold is not None:
        threshold = float(threshold.group(1))
    #print("Threshold =", str(threshold))
    return threshold

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
            if st.session_state.figure_compare is not None:
                for fig in st.session_state.figure_compare:
                    fig.clf()
                st.session_state.figure_compare = None
                gc.collect()

            model_selections=[]
            for i in range(len(images)):
                model_selections.append( st.session_state[f"model_selection_{i}"] )

            #print("Model selections =", model_selections)
            st.session_state.figure_compare = predict_images(images, filenames, models_selected=model_selections, models=models)
    else:
        st.session_state.figure_compare=None

if st.session_state.figure_compare is not None:
    for fig in st.session_state.figure_compare:
        st.pyplot(
            fig
        )
