import numpy as np
import pandas as pd

df = pd.read_csv("../Data/train.csv")
df.fillna(-2, inplace=True)

label_columns = ['No Finding',
       'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
       'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
       'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
       'Support Devices']

# Labels in common with Kaggle:
# No Finding
# Pneumonia - 62 images
# Edema - 118 images
# Cardiomegaly - 141 images
# Consolidation - 226 images
# Pneumothorax - 271 images
# Atelectasis - 508 images

# Maybe similar?
# Pleural_Thickening - 176 images
# Effusion - 644 images


print(len(df))
# Sample 1000 for now
df = df.head(1000)

# TODO: Look at column 0 as well.  Starting at 1 for manual testing b/c it contains both 0s and 1s
for label in label_columns[1:]:
    df_type = df[['Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA', label]]

    # Drop unknowns
    df_type = df_type[(df_type[label] == 1) | (df_type[label] == 0)]

    # TODO: If we're dropping unknowns, use Random undersampler
    print(df_type)
    # print(df_type.groupby(label)['Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA'].sum())
    # print(df_type.sample(n=10, random_state = 0))

    # TODO: This should be removed, or somehow adapted. Only here so testing does not get out-of-hand
    break


# print(df.head())
# print(df.columns)

