#!/bin/bash

wget -bqco download_kaggle_data.log "https://ml2-final-project-data.s3.amazonaws.com/sample.zip" -O sample.zip

echo "Download is proceeding in the background."
echo "Should a problem occur, rerunning this script, will resume the download."
echo "wget output is being directed to download_kaggle_data.log"
echo "use 'tail -f download_kaggle_data.log' to check progress"

if [ ! -d Data ]:
then
  mkdir Data
  if [ $? -ne 0]:
  then
    echo " Error creating the directory. Check your permissions"
    exit(1)
  fi
fi

unzip CheXpert-v1.0-small.zip -d Data/


if [ ! -d archive ]:
then
  mkdir archive
  if [ $? -ne 0]:
  then
    echo " Error creating the directory. Check your permissions"
    exit(1)
fi
mv sample.zip archive/