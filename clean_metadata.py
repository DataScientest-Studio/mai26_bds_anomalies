import pandas as pd

df_mvtec = pd.read_csv('image_list.csv', dtype={"file":str})
df_rad = pd.read_csv('image_list_RAD.csv', dtype={"file":str})

# Suppression des fichiers en doublon dans le dataset RAD
# Il s'agit des noms de fichiers avec une parenthèse
df_rad = df_rad[~df_rad["file"].str.contains(r"\(", na=False)]

# Suppression des fichiers "good" qui sont dupliqués dans chaque catégorie
df_rad = df_rad[(df_rad["category"]=='bolt') | (df_rad["quality"]!='good')]

# Indication du type d'objet étranger dans la colonne "quality" plutôt que category
df_rad.loc[df_rad["quality"]=='defect', 'quality'] = df_rad.loc[df_rad["quality"]!='good', 'category']

# Remplacement de la colonne category par l'object metal_plate :
df_rad["category"] = "metal_plate"

# Ajout des colonnes indiquant les jeux de données avant la fusion : 
df_rad["origin"]="RAD"
df_mvtec["origin"]="MVTec"

# Concaténation des jeux de données
df = pd.concat([df_mvtec, df_rad], axis=0)

# Suppression des colonnes n'apportant aucune information
## L'extension est toujours ".png"
df = df.drop("extension", axis=1)

## Les masques n'ont toujours qu'une bande, le mode "L" et le type "G"
df = df.drop(["mask_type", "mask_bands", "mask_mode"],axis=1)

## Si bands == 3, le mode est toujours "RGB" et le type "C"
## Si bands == 1, le mode est toujours "L" et le type "G"
df = df.drop(["mode", "type_image"],axis=1)

## Si quality == "good", le masque est toujours vide (False). 
## Si quality != "good", le masque n'est jamais vide (True).
df = df.drop("mask", axis=1)

df.to_csv("image_list_clean.csv", index=False)