import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
image_path = Path(os.getenv("PATH_DATASET"))
csv_file = Path(__file__).parent.parent / 'image_list_clean.csv'

from torch.utils.data import Dataset, WeightedRandomSampler
import pandas as pd
import numpy as np
from PIL import Image

df = pd.read_csv(csv_file, dtype={"file":str})
# Pour toutes les lignes avec type = 'test' et quality != 'good', 
# je veux les grouper par 'category' et 'quality' et 
# en sélectionner une proportion test_size pour lesquelles 
# je vais ajouter une colonne "train" qui contiendra True pour 
# les lignes qui seront utilisées pour le train et False pour celles 
# qui ont été sélectionnées pour le test.
test_size = 0.2
random_state = 42
df['train'] = True
df_anomaly = df[(df['type'] == 'test') & (df['quality']!='good')].copy()

# Grouper par 'category' et 'quality'
grouped_df = df_anomaly.groupby(['category', 'quality'])['train'].count()

np.random.seed(random_state)

# Ajouter la colonne 'train' avec True pour les lignes de train et False pour celles de test
for (category, quality), count in grouped_df.items():
    # Tableau de grouped_df.loc[(category, quality)] valeurs dont 
    # 0.2 x grouped_df.loc[(category, quality)] False et le reste True
    train_test_values = np.array([True] * count)

    nb_test = int(test_size * count)
    train_test_values[:nb_test] = False
    np.random.shuffle(train_test_values)

    #print(f"Category: {category}, Quality: {quality}")
    #print(f"   taille = {grouped_df.loc[(category, quality)]}, valeurs = {train_test_values}")

    df.loc[(df['type'] == 'test') & (df['category'] == category) & (df['quality'] == quality), 'train'] = train_test_values

df['image_path'] = df.apply(
    lambda row: (image_path / row['category'] / row['type'] / row['quality'] / f"{row['file']}.png").absolute(),
    axis=1,
)

class AnomalyDataset( Dataset ):
    def __init__(self, category, train=True, binary=False, transform = None, df=df):
        df = df[(df['category'] == category) & (df["train"]==train)].reset_index(drop=True)
        self.X = df.drop(['quality'], axis=1)

        self.binary=binary
        if binary:
            self.y = df['quality'] != 'good'
            self.classes = ['good', 'anomaly']
        else:
            self.y = df['quality']
            self.classes = sorted(df["quality"].unique())
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.targets = np.array([self._label_to_idx(label) for label in self.y])

        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open( self.X.iloc[idx]['image_path'] ).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        
        if self.binary:
            label = float(self.y.iloc[idx])
        else:
            label = self.class_to_idx[self.y.iloc[idx]]

        return img, label

    def __len__(self):
        return len(self.X)

    def _label_to_idx(self, label):
        if self.binary:
            return int(label)
        return self.class_to_idx[label]

    def class_counts(self):
        counts = np.bincount(self.targets, minlength=len(self.classes))
        return dict(zip(self.classes, counts))

    def make_weighted_sampler(self):
        """Ré-échantillonne les classes rares sans dupliquer les fichiers."""
        class_counts = np.bincount(self.targets, minlength=len(self.classes))
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[self.targets]

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
