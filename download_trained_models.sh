#!/bin/bash

if [ ! -d "Models" ]; then
	mkdir Models
fi

wget -bqc "https://ml2-final-project-data.s3.amazonaws.com/model_team8_uzeros_15epoch.pt" -O Models/model_team8_uzeros_15epoch.pt
wget -bqc "https://ml2-final-project-data.s3.amazonaws.com/model_team8_uzeros.pt" -O Models/model_team8_uzeros_v1.pt
wget -bqc "https://ml2-final-project-data.s3.amazonaws.com/model_team8_uones.pt" -O Models/model_team8_uones_v1.pt
wget -bqc "https://ml2-final-project-data.s3.amazonaws.com/model_team8_uzeros_v2.pt" -O Models/model_team8_uones_v2.pt

# images
wget -bqc "https://ml2-final-project-data.s3.amazonaws.com/loss_v_epochs_uones_v1.png" -O Models/loss_v_epochs_uones_v1.png
wget -bqc "https://ml2-final-project-data.s3.amazonaws.com/loss_v_epochs_uzeros_v1.png" -O Models/loss_v_epochs_uzeros_v1.png
wget -bqc "https://ml2-final-project-data.s3.amazonaws.com/loss_v_epochs_uzeros_15epoch.png" -O Models/loss_v_epochs_uzeros_15epoch.png
wget -bqc "https://ml2-final-project-data.s3.amazonaws.com/loss_v_epochs_uzeros_v2.png" -O Models/loss_v_epochs_uzeros_v2.png


echo "Download is proceeding in the background."
echo "Should a problem occur, rerunning this script, will resume the download."
