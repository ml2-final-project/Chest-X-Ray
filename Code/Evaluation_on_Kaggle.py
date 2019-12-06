import pandas as pd
import torch
import numpy as np
from Test_Model_On_Kaggle import predict
from KaggleDataset import LABELS
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
output = torch.from_numpy(np.array([[-2.7946, -2.8696, -2.0371, -0.1423, -3.4820, -1.6071, -2.7869, -3.7146,
         -1.7468, -2.4265, -0.6332, -4.3280, -3.2801,  0.1334]]))
ten = torch.from_numpy(np.array([]))

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
proba_uzeros = [torch.nn.functional.softmax(out).numpy() for out in output_uzeros]


# output_uones, labels_uones = predict("uones")
# proba_uones = [torch.nn.functional.softmax(out).numpy() for out in output_uones]

compare_df = pd.DataFrame({"Labels": LABELS})
compare_df["Uzeros"] = [0]*len(compare_df)
# compare_df["Uones"] = [0]*len(compare_df)

def get_auc_score(row):
    print(row)
    print(row.Labels)
    # for uzeros
    y_predicts = [out[label_columns.index(row.Labels)] for out in proba_uzeros]
    y_actuals = [label[LABELS.index(row.Labels)].numpy() for label in labels_uzeros]
    row["Uzeros"] = roc_auc_score(y_actuals, y_predicts)
    # for uones
    # y_predicts = [out[label_columns.index(row.Labels)] for out in proba_uones]
    # y_actuals = [label[label_columns.index(row.Labels)].numpy() for label in labels_uones]
    # row["Uones"] = roc_auc_score(y_actuals, y_predicts)
    return row

compare_df = compare_df.apply(get_auc_score, axis=1)
print(compare_df)

