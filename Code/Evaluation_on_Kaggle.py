import pandas as pd
import torch
import numpy as np
from Test_Model_On_Kaggle import predict
from KaggleDataset import LABELS
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score

label_columns = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices'
]

output_uzeros, labels_uzeros = predict(model_name="uzeros")
# proba_uzeros = [torch.nn.functional.softmax(out).numpy() for out in output_uzeros]
proba_uzeros = [torch.sigmoid(out).numpy() for out in output_uzeros]


output_uones, labels_uones = predict("uones")
# proba_uones = [torch.nn.functional.softmax(out).numpy() for out in output_uones]
proba_uones = [torch.sigmoid(out).numpy() for out in output_uones]

compare_df = pd.DataFrame({"Labels": LABELS})
compare_df["UZeros_AUC_Score"] = [0]*len(compare_df)
compare_df["UZeros_Precision_Score"] = [0]*len(compare_df)
compare_df["UOnes_AUC_Score"] = [0]*len(compare_df)
compare_df["UOnes_Precision_Score"] = [0]*len(compare_df)

def get_auc_score(row):
    # for uzeros
    y_predicts = [out[label_columns.index(row.Labels)] for out in proba_uzeros]
    y_actuals = [label[LABELS.index(row.Labels)].numpy() for label in labels_uzeros]
    row["UZeros_AUC_Score"] = roc_auc_score(y_actuals, y_predicts)
    row["UZeros_Precision_Score"] = precision_score(y_actuals, [i > 0.5 for i in y_predicts])
    # for uones
    y_predicts = [out[label_columns.index(row.Labels)] for out in proba_uones]
    y_actuals = [label[LABELS.index(row.Labels)].numpy() for label in labels_uones]
    row["UOnes_AUC_Score"] = roc_auc_score(y_actuals, y_predicts)
    row["UOnes_Precision_Score"] = precision_score(y_actuals, [i > 0.5 for i in y_predicts])
    return row

compare_df = compare_df.apply(get_auc_score, axis=1)
print(compare_df.iloc[:, 0:2])
print(compare_df.iloc[:, 2:])


