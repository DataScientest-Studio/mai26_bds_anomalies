import streamlit as st
import numpy as np

from dense_autoencoder.autoencoder_load_images import load_liste_images
from generic_model.autoencoder_model import load_autoencoder

import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()
image_path = Path(os.getenv("PATH_DATASET"))


def predict_image(image):
    #autoencoder = load_autoencoder(model_file)
    return 

##### PAGE CONTENT #####
st.title("Détection d'anomalies")

images, nb_channels = load_liste_images(image_path, resized_dimension=(256,256), category='wood', type='test', quality='good', include_augmented=False, limit_to=10)
image_w_h = int(np.sqrt(images.shape[1] / nb_channels))
image_dimensions = [ image_w_h, image_w_h, nb_channels ]
images = images.reshape(-1, *image_dimensions)
print(images[0].shape)
image = images[0]

st.image(image)
st.button("Analyser l'image", icon=":material/image_search:", on_click=predict_image, args=image)