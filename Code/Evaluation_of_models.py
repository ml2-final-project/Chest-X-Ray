import pandas as pd
import torch
import numpy as np
from Test_Model_On_Chexpert import predict
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
output = torch.from_numpy(np.array([[-2.7946, -2.8696, -2.0371, -0.1423, -3.4820, -1.6071, -2.7869, -3.7146,
         -1.7468, -2.4265, -0.6332, -4.3280, -3.2801,  0.1334]]))
print((torch.nn.functional.softmax(output,dim=0)>0.5))
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

selected_labels = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis','Pleural Effusion']
index_list = [target_labels.index(label) for label in selected_labels]

labels_uzeros, output_uzeros = predict("uzeros")
proba_uzeros = [torch.nn.functional.softmax(out) for out in output_uzeros]
labels_uones, output_ones = predict("uones")
proba_uones = [torch.nn.functional.softmax(out) for out in output_ones]


compare_df = pd.DataFrame({"Labels": selected_labels})
compare_df["Uzeros"] = [0]*len(compare_df)
compare_df["Uones"] = [0]*len(compare_df)

def get_auc_score(row):
    # for uzeros
    y_predicts = [out[target_labels.index(row["Labels"])] for out in proba_uzeros]
    y_actuals = [label[target_labels.index(row["Labels"])] for label in labels_uzeros]
    row["Uzeros"] = roc_auc_score(y_actuals, y_predicts)
    # for uones
    y_predicts = [out[target_labels.index(row["Labels"])] for out in proba_ones]
    y_actuals = [label[target_labels.index(row["Labels"])] for label in labels_ones]
    row["Uones"] = roc_auc_score(y_actuals, y_predicts)

comapre_df = compare_df.apply(get_auc_score)
print(compare_df)


