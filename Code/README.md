# Code

Summary of files, and sections pertaining to them.

| FileName                  | Section  | 
| ------------------------- | -------- | 
| CheXpertDataset.py        | [Custom Datasets](#custom-datasets)   |
| KaggleDataset.py          | [Custom Datasets](#custom-datasets)   |
| Evaluation_of_models.py   | [Model Evaluation](#model-evaluation) |
| Evaluation_on_Kaggle.py   | [Model Evaluation](#model-evaluation) |
| ImageTransforms.py        | [Transforms](#transforms) |
| Test_Model_On_Chexpert.py | [Model Evaluation](#model-evaluation) |
| Test_Model_On_Kaggle.py   | [Model Evaluation](#model-evaluation) |
| testing_common_utils.py   | [Model Evaluation](#model-evaluation) |
| training_common_utils.py  | [Training](#training)                 |
| training_team8_uones.py   | [Training](#training)                 |
| training_team8_uzeros.py  | [Training](#training)                 |

## Custom Datasets

The [CheXpertDataset.py](CheXpertDataset.py) file defines the custom Dataset object we used to construct pytorch DataLoaders for the CheXpertDataset.

The [KaggleDataset.py](KaggleDataset.py) file defines the custom Dataset object we used to construct pytorch DataLoaders for the KaggleDataset.

## Transforms

The majority of our custom image transform code is found in [ImageTransforms.py](ImageTransforms.py). Some transforms being used directly from torchvision, as provided, are setup in the [training_common_utils.py](training_common_utils.py) file.

Label transforms are defined in the [CheXpertDataset.py](CheXpertDataset.py), and [KaggleDataset.py](KaggleDataset.py) files, and initialized in the [training_common_utils.py](training_common_utils.py).

## Training

The primary training code, is found in [training_team8_uones.py](training_team8_uones.py) and the [training_team8_uzeros.py](training_team8_uzeros.py) files, which correspond with label preprocessing using the uones, and uzeros strategies respectively.

These two file invoke functions defined in [training_common_utils.py](training_common_utils.py), which exists to avoid duplication (and error).

## Model Evaluation

The primary testing code, is found in [Evaluation_of_models.py](Evaluation_of_models.py) and [Evaluation_on_Kaggle.py](Evaluation_on_Kaggle.py) files, which correspond with testing both the UZeros and UOnes based models on the Chexpert and Kaggle datasets respectively. 

These two files invoke functions defined in both [Test_Model_On_Chexpert.py](Test_Model_On_Chexpert.py) and [Test_Model_On_Kaggle.py](Evaluation_on_Kaggle.py). Those two files then invoke functions defined in the [testing_common_utils.py](testing_common_utils.py) file. All of these additional files exist to avoid duplication and generally reduce error-prone copy-pasting. 
