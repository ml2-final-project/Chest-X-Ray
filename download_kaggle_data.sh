#!/bin/bash

mkdir Data

wget -bqco download_kaggle_data.log "https://ml2-final-project-data.s3.amazonaws.com/sample.zip" -O Data/sample.zip

echo "Download is proceeding in the background."
echo "Should a problem occur, rerunning this script, will resume the download."
echo "wget output is being directed to download_kaggle_data.log"
echo "use 'tail -f download_kaggle_data.log' to check progress"
