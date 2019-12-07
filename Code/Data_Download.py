# Please execute this script only once. The script download several gigabytes of data from AWS S3 bucket
import os

os.system("sh ../download_kaggle_data.sh")
os.system("sh ../download_stanford_data.sh")
os.system("sh ../download_trained_models.sh")

print("The data has been downloaded.")
print("The unzipped files are present in a directory with name: Data")