# mai26_bds_anomalies

# Détection d'anomalies sur des pièces industrielles

Ce projet a pour objectif de détecter automatiquement la présence d'une anomalie sur une pièce industrielle à partir d'une image.

L'approche principale repose sur des **auto-encodeurs entraînés uniquement sur des images sans défaut**. Le modèle apprend à reconstruire l'apparence normale d'une catégorie de pièces. Lorsqu'une image contient une anomalie, sa reconstruction est généralement moins fidèle et l'erreur entre l'image originale et l'image reconstruite devient plus importante.

Le projet compare plusieurs architectures d'auto-encodeurs classiques et variationnels, avec ou sans apprentissage par transfert, sur les jeux de données **MVTec** et **RAD**.

> Une présentation détaillée de la démarche, des architectures, des résultats et des limites est disponible dans [`Rapport.pdf`](./Rapport.pdf).

> **Note :** ce README a été rédigé avec l’assistance d’une intelligence artificielle, à partir du contenu du projet. Le rapport et le code ont été rédigés par les auteurs du projet.

---

## Objectifs

L'objectif principal est de réaliser une classification binaire :

- **pièce conforme** : aucune anomalie détectée ;
- **pièce non conforme** : présence probable d'une anomalie.

Pour chaque image, le modèle calcule une erreur de reconstruction, par exemple une MAE ou une MSE. Cette erreur est comparée à un seuil afin de déterminer si l'image doit être considérée comme normale ou anormale.

Le projet étudie également la localisation visuelle des zones importantes pour la reconstruction et pour l'erreur, à l'aide de cartes d'erreur MAE et MSE et Grad-CAM.

L'identification précise du type d'anomalie constitue un objectif secondaire. Elle n'est pas au cœur de l'approche présentée ici, qui reste centrée sur la détection binaire.

---

## Jeux de données

### MVTec AD

MVTec AD contient plusieurs catégories d'objets et de textures industrielles. Pour chaque catégorie :

- `train/good` contient uniquement des images sans anomalie ;
- `test/good` contient des images sans anomalie ;
- `test/<type_anomalie>` contient des images défectueuses ;
- `ground_truth/<type_anomalie>` contient les masques des anomalies.

Les catégories étudiées comprennent notamment `bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`, `leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood` et `zipper`.

### RAD

Le jeu de données RAD contient des photographies d'une plaque métallique perforée sur laquelle différents objets étrangers peuvent être déposés. Dans le projet, ces images sont regroupées sous la catégorie `metal_plate`.

Les images normales servent à l'entraînement, tandis que les images contenant un objet étranger sont utilisées pour l'évaluation.

Les jeux de données ne sont pas inclus dans le dépôt. Leur emplacement doit être défini localement dans la variable d'environnement `PATH_DATASET`.

---

## Principe de détection

Le pipeline utilisé est le suivant :

1. chargement et prétraitement des images ;
2. entraînement d'un modèle sur les seules images `good` ;
3. reconstruction des images d'entraînement et de test ;
4. calcul d'un score d'erreur par image ;
5. comparaison de ce score à un seuil ;
6. évaluation à l'aide de la courbe ROC, de l'AUC, du TPR, du FPR et de la matrice de confusion.

Le seuil peut être déterminé à partir d'un percentile des erreurs observées sur les images d'entraînement. Pour la démonstration Streamlit, lorsqu'un fichier ROC est disponible, le seuil retenu est celui qui maximise l'indice de Youden :

```text
Youden = TPR - FPR
```

En cas d'égalité, le seuil ayant le plus faible taux de faux positifs est privilégié.

---

## Architectures étudiées

Plusieurs familles de modèles sont disponibles.

### Auto-encodeur dense

L'image est aplatie, compressée dans un espace latent par plusieurs couches denses, puis reconstruite par un décodeur symétrique.

### Auto-encodeur convolutionnel

L'encodeur utilise des convolutions et des opérations de pooling afin d'extraire les motifs locaux de l'image. Le décodeur restaure progressivement les dimensions spatiales avec des convolutions et du sur-échantillonnage.

### Architectures hybrides

Deux combinaisons ont été testées :

- `conv_dense` : encodeur convolutionnel et décodeur dense ;
- `dense_conv` : encodeur dense et décodeur convolutionnel.

### Apprentissage par transfert

Les modèles `convtl` et `convtl_dense` utilisent **EfficientNetB0 pré-entraîné sur ImageNet** comme encodeur :

- `convtl` utilise un décodeur convolutionnel ;
- `convtl_dense` utilise un décodeur dense.

### Auto-encodeurs variationnels

Deux variantes de VAE ont également été étudiées sur les données RAD :

- un VAE convolutionnel classique ;
- un VAE avec perte perceptuelle calculée à partir d'un réseau VGG16 pré-entraîné.

---

## Structure générale du dépôt

La structure exacte peut évoluer, mais les principaux répertoires sont organisés de la manière suivante :

```text
.
├── dense_autoencoder/       # auto-encodeur dense historique
├── generic_model/          # auto-encodeurs configurables et modèles de transfert
├── VAE_1/                  # VAE convolutionnel pour RAD
├── VAE_transfer/           # VAE avec perte perceptuelle et Grad-CAM
├── output/                 # modèles, mesures et figures générés
├── streamlit_app.py        # application de démonstration
├── requirements.txt
├── requirements-lock.txt
├── sample.env
├── Rapport.pdf
└── README.md
```

### `dense_autoencoder/`

Première implémentation d'un auto-encodeur dense simple.

Principaux scripts :

- `autoencode.py` : script principal d'entraînement et d'évaluation ;
- `autoencoder_load_images.py` : chargement des images avec OpenCV ;
- `autoencoder_model.py` : création et compilation du modèle ;
- `autoencode_figures.py` : production des graphiques d'évaluation.

Cette première version charge les images en mémoire et travaille sur une catégorie à la fois.

### `generic_model/`

Implémentation principale et configurable du projet.

Principaux scripts :

- `autoencode.py` : entraînement, chargement d'un modèle sauvegardé et évaluation ;
- `autoencoder_model.py` : définition et chargement des différentes architectures ;
- `autoencode_data_augment.py` : augmentation de données à la volée ;
- `autoencode_figures.py` : génération des graphiques et visualisations ;
- `preprocessing_screw.py` : prétraitement spécifique des images de vis ;
- `vae_transfer_model.py` : définition du VAE avec apprentissage par transfert.

Le notebook `Analyse_train_results.ipynb` se trouve à la racine du dépôt. L'historique des configurations et des métriques est enregistré dans `output/generic_model/0_train_results.csv`.

Les principaux paramètres sont regroupés dans la section `SETTINGS` du script d'entraînement. Ils permettent notamment de modifier :

- la taille des images ;
- la taille des batchs ;
- le passage en niveaux de gris ;
- les augmentations de couleur et de position ;
- le type de modèle ;
- le nombre de couches ré-entraînées ;
- la fonction de perte ;
- le score d'erreur utilisé ;
- le percentile servant à définir le seuil.

### `VAE_1/` et `VAE_transfer/`

Ces répertoires contiennent les expériences menées sur RAD :

- `VAE_1/` implémente le VAE convolutionnel classique ;
- `VAE_transfer/` implémente le VAE avec perte perceptuelle et les visualisations Grad-CAM.

### Scripts d'analyse et de préparation des données

Le dépôt contient également plusieurs scripts consacrés à la préparation et à l'analyse des jeux de données, notamment :

- `list_images_project.py` : extraction des métadonnées de MVTec AD ;
- `list_images_project_rad.py` : extraction des métadonnées de RAD ;
- `clean_metadata.py` ou script équivalent : nettoyage et fusion des métadonnées ;
- `augmentation_MVTec.py` : génération d'images augmentées pour MVTec ;
- `augmentation_RAD.py` : génération d'images augmentées pour RAD.

---

## Répertoire `output`

Les modèles, métriques et graphiques sont enregistrés automatiquement dans le répertoire `output`.

Pour l'auto-encodeur configurable, chaque entraînement est stocké dans un dossier dont le nom reprend les principaux hyperparamètres, par exemple :

```text
64-8-True-True-True-conv_dense-0-mae-mse
```

Ce nom peut contenir successivement :

```text
resized_dimension
batch_size
grayscale
color_augmentation
move_augmentation
model_type
retrain_layers
loss
error_score
```

Chaque sous-dossier contient les fichiers associés à une ou plusieurs catégories. Exemple pour `metal_plate` :

```text
metal_plate_autoencoder.keras
metal_plate_classification_report.txt
metal_plate_histogramme_erreurs.png
metal_plate_history_plot.png
metal_plate_images_reconstruites_test.png
metal_plate_images_reconstruites_train.png
metal_plate_images_reconstruites_train_augmented.png
metal_plate_matrice_confusion.png
metal_plate_parameters.txt
metal_plate_roc_curve.png
metal_plate_roc_curve.txt
```

Les modèles `.keras` peuvent être volumineux et ne sont pas nécessairement versionnés dans GitHub.

---

## Graphiques et fichiers générés

### Historique d'entraînement

Le fichier `*_history_plot.png` montre l'évolution de la perte sur les jeux d'entraînement et de validation au fil des epochs. Il permet notamment de repérer une stagnation ou un début de sur-apprentissage.

Ce graphique n'est généré que lorsqu'un entraînement est réellement exécuté.

### Histogramme des erreurs

Le fichier `*_histogramme_erreurs.png` compare la distribution des erreurs de reconstruction pour :

- les images d'entraînement sans anomalie ;
- les images de test sans anomalie ;
- les images de test avec anomalie.

Le seuil de décision est affiché sur le graphique.

### Courbe ROC

Le fichier `*_roc_curve.png` représente le taux de vrais positifs en fonction du taux de faux positifs pour l'ensemble des seuils possibles.

Le fichier `*_roc_curve.txt` contient les valeurs numériques utilisées :

```text
threshold,fpr,tpr
```

L'AUC permet de comparer les modèles indépendamment d'un seuil particulier.

### Matrice de confusion

Le fichier `*_matrice_confusion.png` montre la répartition des :

- vrais négatifs ;
- faux positifs ;
- faux négatifs ;
- vrais positifs.

### Classification report

Le fichier `*_classification_report.txt` contient les métriques calculées pour plusieurs seuils, notamment :

- précision ;
- rappel ;
- F1-score ;
- accuracy ;
- spécificité et FPR déduits des résultats.

### Images reconstruites

Les fichiers `*_images_reconstruites_*.png` permettent de comparer visuellement :

- l'image originale après prétraitement ;
- la carte Grad-CAM, lorsqu'elle est disponible ;
- l'image reconstruite ;
- la carte d'erreur MAE ;
- la carte d'erreur MSE.

Des exemples sont générés pour les images d'entraînement, les images augmentées et les images de test.

### Paramètres

Le fichier `*_parameters.txt` conserve les paramètres utilisés lors de l'entraînement afin de rendre les résultats plus faciles à reproduire et à comparer.

---

## Prétraitement spécifique des vis

Les images de la catégorie `screw` présentent des orientations variables. Un prétraitement OpenCV a donc été développé afin de :

1. détecter le contour principal de la vis ;
2. estimer son orientation ;
3. l'aligner horizontalement ;
4. placer la tête à gauche ;
5. recentrer et redimensionner l'image.

Les images ainsi traitées peuvent être évaluées sous la catégorie `screw_preprocessed`, afin de comparer leurs performances avec celles des images originales.

---

## Application de démonstration Streamlit

Le fichier [`streamlit_app.py`](./streamlit_app.py) fournit une interface de démonstration interactive.

L'application comporte deux onglets.

### Onglet `Analyse`

Cet onglet permet de :

1. téléverser une ou plusieurs images ;
2. afficher les images sous forme de grille ;
3. sélectionner le modèle à utiliser pour chaque image ;
4. lancer l'analyse ;
5. afficher les résultats de reconstruction et de classification.

Le modèle est proposé automatiquement lorsque le nom du fichier contient le nom d'une catégorie connue. La sélection peut être modifiée manuellement avant de lancer l'analyse.

Pour chaque image, l'application :

- charge le meilleur modèle disponible pour la catégorie ;
- applique le redimensionnement et, si nécessaire, la conversion en niveaux de gris ;
- applique le prétraitement spécifique aux vis ;
- reconstruit l'image ;
- calcule son erreur ;
- compare cette erreur au seuil du modèle ;
- affiche la reconstruction, les cartes d'erreur et le résultat de la détection.

L'application regroupe les images utilisant un même modèle afin de limiter les chargements successifs. Elle libère également les modèles et la mémoire TensorFlow entre les traitements, ce qui est utile lorsque plusieurs modèles sont analysés dans une même session.

### Onglet `Modèles`

Cet onglet présente les résultats enregistrés dans `output/generic_model/0_train_results.csv`.

Il permet de :

- consulter les hyperparamètres et la ROC AUC des meilleurs modèles ;
- filtrer l'affichage par catégorie ;
- afficher le meilleur score par catégorie ;
- comparer les différentes architectures avec une heatmap des meilleures ROC AUC obtenues.

---

## Lancer la démonstration

### 1. Installer les dépendances

Le projet utilise principalement :

- Python ;
- TensorFlow / Keras ;
- Streamlit ;
- NumPy ;
- pandas ;
- Matplotlib ;
- Seaborn ;
- Pillow ;
- OpenCV ;
- python-dotenv.

Installer les dépendances à partir du fichier fourni dans le dépôt, par exemple :

```bash
pip install -r requirements.txt
```

### 2. Configurer le chemin des données

Créer un fichier `.env` à la racine du projet :

```env
PATH_DATASET=/chemin/vers/les/datasets
```

### 3. Vérifier les modèles disponibles

L'application attend les résultats et modèles dans :

```text
output/generic_model/
```

Le fichier suivant doit notamment être présent :

```text
output/generic_model/0_train_results.csv
```

Les chemins des modèles sont reconstruits automatiquement à partir des hyperparamètres enregistrés dans ce CSV.

### 4. Démarrer Streamlit

```bash
streamlit run streamlit_app.py
```

L'application s'ouvre ensuite dans le navigateur local.

---

## Évaluation et interprétation

La principale métrique utilisée pour comparer les architectures est la **ROC AUC**, car elle permet d'évaluer la capacité du modèle à classer les anomalies indépendamment du seuil choisi.

Le TPR et le FPR sont également enregistrés, mais ils dépendent directement du seuil retenu.

Les résultats montrent que :

- les performances varient fortement selon les catégories ;
- les modèles avec apprentissage par transfert sont souvent parmi les meilleurs ;
- certaines catégories, comme les vis, peuvent mieux fonctionner avec un modèle convolutionnel sans transfert et un traitement en niveaux de gris ;
- les textures bénéficient souvent d'un encodeur pré-entraîné associé à un décodeur dense ;
- aucun modèle unique ne domine systématiquement toutes les catégories.

Ces résultats doivent être interprétés avec prudence : les jeux de test contiennent souvent une proportion d'anomalies très différente d'un cas industriel réel, et le coût relatif des faux positifs et faux négatifs n'est pas connu.

---

## Limites et pistes d'amélioration

Les principales limites identifiées sont :

- le faible nombre d'images disponibles pour certaines catégories ;
- la diversité importante des pièces, textures, orientations et conditions d'éclairage ;
- des jeux de test peu représentatifs de la fréquence réelle des anomalies ;
- un seuil de décision difficile à définir sans données métier ;
- des scores MAE et MSE globaux qui peuvent diluer les petites anomalies localisées.

Plusieurs pistes d'amélioration sont envisagées :

- tester davantage de tailles d'espaces latents et de profondeurs de réseaux ;
- mesurer les erreurs sur des zones locales plutôt que sur l'image entière ;
- utiliser des approches spécialisées comme PatchCore ;
- réaliser plusieurs entraînements par configuration afin de mesurer la variance des résultats ;
- utiliser davantage de données de validation ;
- entraîner un modèle supervisé pour identifier le type précis d'anomalie.

---

## Rapport

Le document [`Rapport.pdf`](./Rapport.pdf) présente de manière détaillée :

- l'analyse des jeux de données ;
- les scripts de préparation ;
- les différentes architectures ;
- les choix de métriques ;
- les résultats par catégorie ;
- la comparaison statistique des VAE avec le test de DeLong ;
- les limites et les axes d'amélioration.

Le README fournit une vue d'ensemble du dépôt et des principaux points d'entrée. Le rapport reste la référence pour l'analyse complète du projet.
