import pandas as pd
import torch
import numpy as np
from Test_Model_On_Chexpert import predict
from sklearn.metrics import accuracy_score, roc_auc_score,\
    roc_curve, precision_score, classification_report, f1_score, \
    recall_score
import matplotlib.pyplot as plt
from KaggleDataset import LABELS

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

# selected_labels = ['Cardiomegaly',
#                    'Edema',
#                    'Consolidation',
#                    'Atelectasis',
#                    'Pleural Effusion']

output_uzeros, labels_uzeros = predict("uzeros")
# proba_uzeros = [torch.nn.functional.softmax(out, dim=0).numpy() for out in output_uzeros]
proba_uzeros = [torch.sigmoid(out).numpy() for out in output_uzeros]


output_uones, labels_uones = predict("uones")
# proba_uones = [torch.nn.functional.softmax(out, dim=0).numpy() for out in output_uones]
proba_uones = [torch.sigmoid(out).numpy() for out in output_uones]


def initCols(model_names, col_names):
    fillValue = [0] * len(compare_df)
    for model_name in model_names:
        for col_name in col_names:
            compare_df[model_name + "_" + col_name] = fillValue


# compare_df = pd.DataFrame({"Labels": selected_labels})
compare_df = pd.DataFrame({"Labels": LABELS})
initCols(
    ["UZeros", "UOnes"],
    ["AUC", "Precision", "Recall", "Accuracy", "F1"]
)


def get_auc_score(row):

    def add_all_classifiers(name, y_actuals, y_predicts, y_predicts_probs):
        print(classification_report(y_actuals, y_predicts))
        row[name + "_AUC"] = roc_auc_score(y_actuals, y_predicts_probs)
        row[name + "_Precision"] = precision_score(y_actuals, y_predicts, average='weighted')
        row[name + "_Recall"] = recall_score(y_actuals, y_predicts, average='weighted')
        row[name + "_Accuracy"] = accuracy_score(y_actuals, y_predicts)
        row[name + "_F1"] = f1_score(y_actuals, y_predicts, average='weighted')

    # for uzeros
    print(row.Labels)
    y_predicts = [out[label_columns.index(row.Labels)] for out in proba_uzeros]
    y_actuals = [label[label_columns.index(row.Labels)].numpy() for label in labels_uzeros]
    add_all_classifiers("UZeros", y_actuals, np.array(y_predicts) > 0.5, y_predicts)
    # for uones
    y_predicts = [out[label_columns.index(row.Labels)] for out in proba_uones]
    y_actuals = [label[label_columns.index(row.Labels)].numpy() for label in labels_uones]
    add_all_classifiers("UOnes", y_actuals, np.array(y_predicts) > 0.5, y_predicts)

    return row


compare_df = compare_df.apply(get_auc_score, axis=1)
with pd.option_context('display.max_columns', None):
    print(compare_df)


